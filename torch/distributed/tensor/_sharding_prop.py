# mypy: allow-untyped-defs
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import chain, product
from typing import cast, Optional, Union

import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
    try_find_mesh_from_args,
)


aten = torch.ops.aten


def _length(obj) -> int:
    if obj is None:
        return 0
    if not isinstance(obj, Sequence):
        return 1
    return len(obj)


class LocalLRUCache(threading.local):
    def __init__(self, user_function: Callable) -> None:
        self.cache = lru_cache(None)(user_function)

    def __call__(self, *args, **kwargs) -> object:
        return self.cache(*args, **kwargs)

    def cache_info(self):
        return self.cache.cache_info()


class ShardingPropagator:
    def __init__(self) -> None:
        self.op_to_rules: dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.op_strategy_funcs: dict[
            OpOverload,
            Callable[[OpSchema], StrategyType],
        ] = {}
        # op map to save static argnum to decide to reuse sharding prop cache or
        # re-run sharding prop
        self.op_to_schema_info: dict[OpOverload, RuntimeSchemaInfo] = {}
        self.propagate_op_sharding = LocalLRUCache(
            self.propagate_op_sharding_non_cached
        )
        # op map to save indices of shape (and stride) args which may need to be
        # modified in sharding prop
        self.op_to_shape_and_stride_idx: dict[
            OpOverload, Union[int, tuple[int, int]]
        ] = {
            # new factory ops
            aten.new_empty.default: 1,
            aten.new_full.default: 1,
            aten.new_ones.default: 1,
            aten.new_zeros.default: 1,
            aten.new_empty_strided.default: (1, 2),
            # view ops
            aten.expand.default: 1,
            aten.reshape.default: 1,
            aten.view.default: 1,
            aten._unsafe_view.default: 1,
            aten.select_backward.default: 1,
            aten.slice_backward.default: 1,
        }

        # op map to save decomposition tables so that the op strategy can be generated from decomposition
        self.op_to_decompositions: dict[OpOverload, dict[OpOverload, Callable]] = {}

    def register_sharding_prop_rule(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[OpSchema], OutputSharding],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_op_decomposition(
        self,
        op_overload: OpOverload,
        decomposition_table: dict[OpOverload, Callable],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ) -> None:
        """
        Register a decomposition table for an operator.
        """
        self.op_to_decompositions[op_overload] = decomposition_table

        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_func: Callable[[OpSchema], StrategyType],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        Register a :class:`OpStrategy` generator for an operator.

        During the sharding propagation, DTensor wants to enumerate all
        acceptable sharding specs (:class:`OpSpec`) for an operator,
        and by "acceptable" we mean that the operator can be executed on
        the ``_local_tensor`` of DTensor args/kwargs (with ``OpSpec.input_specs``)
        and the output(s) constitute valid DTensor(s) (with ``OpSpec.output_specs``).

        ``strategy_func`` is the function that enumerates such acceptable specs
        for the operator ``op_overload``. One general approach to write ``strategy_func``
        is, if the operator has simple arguments structure (e.g. mm, bmm), first enumerating
        all sharding specs for the operands, and then filtering out the ones that
        are not valid. For example, for ``mm``, the operands are two 2D tensors, and
        if both ``input`` and ``mat2`` have sharding placements ``[Shard(0)]``, then this
        is not an acceptable ``input_specs``.

        Once we have a way to enumerate all acceptable sharding specs, we can use each
        of them to construct a :class:`OpSpec`. The ``OpSpec.input_specs`` directly comes
        from the sharding spec, and the ``OpSpec.output_specs`` is therefore determined
        (e.g. ``[Shard(1)]`` @ ``[Shard(0)]`` yields ``[Partial()]``). In addition,
        :class:`OpSpec` also contains ``redistribute_cost`` which records the redistribution
        cost from each :class:`OpSpec` in the source :class:`OpStrategy.strategies` to
        the target sharding spec, for each operand.

        The ``strategy_func`` should return a :class:`OpStrategy` which contains a list of
        all the :class:`OpSpec`s generated in the above.

        The optional ``schema_info`` tells which non-DTensor args/kwargs could affect the
        cache and whether ``pytree`` is needed to flatten the nested args. ``static_argnum``
        marks the starting index of the non-DTensor args that should be hashed into the
        sharding propagation hash key, and ``static_kwargkey`` marks the keys of the
        non-DTensor kwargs that should be hashed. ``needs_pytree`` should be used when
        the input arg has :class:`list` or :class:`dict` structure.

        For example, ``aten.cat.default`` op has a ``List[Tensor]`` argument ``tensors``
        and an ``int`` argument ``dim``. Because ``dim`` affects the sharding propagation
        result, we want to pass ``RuntimeSchemaInfo(static_argnum=1)`` because the argument
        index of ``dim`` is 1. Besides, we also want to set ``needs_pytree=True`` because
        ``tensors`` needs be flattened in sharding propagation. Another example is
        ``aten.histc.default``. ``histc`` has 4 arguments (self, bins, min, max) and the
        last two would affect sharding propagation along with the :class:`DTensor` argument
        ``self``. Since the argument index of ``min`` is 2, the `schema_info` should be
        `RuntimeSchemaInfo(static_argnum=2)`.
        """
        self.op_strategy_funcs[op_overload] = strategy_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def _propagate_tensor_meta_non_cached(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
        if op_schema.op == aten.equal.default:
            # data dependent ops can't be used for fake propagation
            return None

        # NOTE: We must call the tracing in fake tensor mode so that it avoids
        # materializing memory. Also disable the proxy mode tracing to prevent
        # these operators to be inserted in the fx graph.
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

        with FakeTensorMode(), disable_proxy_modes_tracing():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            fake_out = op_schema.op(*fake_args, **fake_kwargs)

        if isinstance(fake_out, torch.Tensor):
            return TensorMeta(
                shape=fake_out.shape, stride=fake_out.stride(), dtype=fake_out.dtype
            )

        elif isinstance(fake_out, (tuple, list)):
            tensor_meta_list: list[Optional[TensorMeta]] = []
            for fake_out_item in fake_out:
                if isinstance(fake_out_item, torch.Tensor):
                    tensor_meta_list.append(
                        TensorMeta(
                            shape=fake_out_item.shape,
                            stride=fake_out_item.stride(),
                            dtype=fake_out_item.dtype,
                        )
                    )
                else:
                    tensor_meta_list.append(None)
            return (
                tuple(tensor_meta_list)
                if isinstance(fake_out, tuple)
                else tensor_meta_list
            )
        else:
            # if fake is not a tensor or tuple of tensor, return as none
            return None

    @lru_cache  # noqa: B019
    def _propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Cached version of _propagate_tensor_meta_non_cached
        This is a private API. Use propagate_tensor_meta instead.
        """
        return self._propagate_tensor_meta_non_cached(op_schema)

    def propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas. This is a public API that should be
        used if cache should be used.
        """
        if _are_we_tracing():
            return self._propagate_tensor_meta_non_cached(op_schema)
        else:
            return self._propagate_tensor_meta(op_schema)

    def _create_output_spec_with_new_tensor_meta(
        self,
        op: OpOverload,
        output_specs: OutputSpecType,
        output_tensor_meta: Union[None, TensorMeta, Sequence[Optional[TensorMeta]]],
    ) -> OutputSpecType:
        """
        Wrap the output_specs with the tensor metadata from the output.
        """

        if isinstance(output_specs, DTensorSpec):
            if not isinstance(output_tensor_meta, TensorMeta):
                # Either error due to ShardingPropagator or due to incorrect OutputSpec
                if not isinstance(output_tensor_meta, (tuple, list)):
                    raise ValueError(
                        "ShardingPropagator error: output does not have an associated "
                        "TensorMeta"
                    )
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has 1 output which does "
                    "not equal the "
                    f"number of op outputs: {len(output_tensor_meta)}."
                )
            return output_specs.shallow_copy_with_tensor_meta(output_tensor_meta)
        elif isinstance(output_specs, (tuple, list)):
            new_specs: list[Optional[DTensorSpec]] = []
            if not isinstance(output_tensor_meta, (tuple, list)) or len(
                output_specs
            ) != len(output_tensor_meta):
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has {len(output_specs)} "
                    "outputs which does not equal the "
                    f"number of op outputs {_length(output_tensor_meta)}."
                )

            for i, spec in enumerate(output_specs):
                if isinstance(spec, DTensorSpec):
                    output_tensor_meta_i = output_tensor_meta[i]
                    if not isinstance(output_tensor_meta_i, TensorMeta):
                        # NOTE: aten.convolution_backward.default is an exception and it
                        # needs extra handling because the first Tensor in the output
                        # tuple can be `None` if the input Tensor to convolution op has
                        # `requires_grad=False` (e.g. convolution layer is the first
                        # layer in the model). We explicitly allow its corresponding
                        # TensorMeta to be `None`.
                        if (
                            op == aten.convolution_backward.default
                            and i == 0
                            and output_tensor_meta_i is None
                        ):
                            assert isinstance(output_specs, list)
                            new_specs.append(None)
                            continue
                        else:
                            raise ValueError(
                                f"ShardingPropagator error: output {i} of {op.name()} "
                                "does not have an associated TensorMeta"
                            )

                    new_specs.append(
                        spec.shallow_copy_with_tensor_meta(output_tensor_meta_i)
                    )
                else:
                    new_specs.append(spec)

            return tuple(new_specs)
        else:
            assert output_specs is None
            return output_specs

    def _wrap_with_op_strategy(self, op_schema: OpSchema) -> OpStrategy:
        """
        wrap a op_schema that contains DTensorSpec to another op_schema that contains
        OpStrategy/TupleStrategy, the returned op_schema is then used for sharding
        strategy propagation on pytorch operators.
        """

        if op_schema.op in self.op_to_decompositions:
            decomposition_table = self.op_to_decompositions[op_schema.op]
            return self._generate_op_strategy_from_decomposition(
                op_schema, decomposition_table
            )
        else:
            return self.op_strategy_funcs[op_schema.op](
                self._spec_schema_to_strategy_schema(op_schema)
            )

    def _spec_to_strategy(self, spec: object) -> object:
        if isinstance(spec, DTensorSpec):
            return OpStrategy([OpSpec(spec)])
        elif (
            isinstance(spec, (list, tuple))
            and len(spec) > 0
            and isinstance(spec[0], DTensorSpec)
        ):
            # tensor list create tuple strategy
            tuple_strategy = [self._spec_to_strategy(s) for s in spec]
            tuple_strategy = cast(Sequence[StrategyType], tuple_strategy)
            return TupleStrategy(
                tuple(tuple_strategy) if isinstance(spec, tuple) else tuple_strategy
            )
        else:
            return spec

    def _spec_schema_to_strategy_schema(self, op_schema: OpSchema) -> OpSchema:
        # swap the args spec with args strategies
        args_op_strategy = [self._spec_to_strategy(i) for i in op_schema.args_schema]
        kwargs_op_strategy = {
            k: self._spec_to_strategy(v) for k, v in op_schema.kwargs_schema.items()
        }

        # construct a new OpSchema on args for strategy based propagation
        return OpSchema(
            op=op_schema.op,
            args_schema=tuple(args_op_strategy),
            kwargs_schema=kwargs_op_strategy,
        )

    def _prepare_op_graph(
        self,
        op_schema: OpSchema,
        decomposition_table: dict[OpOverload, Callable],
    ) -> Optional[torch.fx.GraphModule]:
        from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule

        # call tracing in fake tensor mode to avoid materializing memory
        with FakeTensorMode():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            gm = get_isolated_graphmodule(
                op_schema.op,
                fake_args,
                fake_kwargs,
                decomposition_table=decomposition_table,
            )

        return gm

    def _propagate_sharding_through_graph(self, gm, input_specs):
        # 1. for each call_function, generate an OpSchema and run sharding prop
        # 2. filter out those which needs_redistribute
        node_to_spec = {}
        placeholder_idx = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node_to_spec[node] = input_specs[placeholder_idx]
                placeholder_idx += 1
            elif node.op == "call_function":
                if (
                    not isinstance(node.target, OpOverload)
                    or node.target not in self.op_strategy_funcs
                ):
                    raise NotImplementedError(
                        "Generating sharding strategy via decomposition failed "
                        f"because {node.target} does not have a sharding strategy registered."
                    )

                # generate node output spec via sharding propagation
                node_input_arg_strategies = tuple(
                    self._spec_to_strategy(node_to_spec.get(arg, arg))
                    for arg in node.args
                )
                node_input_kwarg_strategies = {
                    k: self._spec_to_strategy(node_to_spec.get(v, v))
                    for k, v in node.kwargs.items()
                }
                node_input_strategy_schema: OpSchema = OpSchema(
                    op=node.target,
                    args_schema=node_input_arg_strategies,
                    kwargs_schema=node_input_kwarg_strategies,
                )

                node_output_strategy = self.op_strategy_funcs[node.target](
                    node_input_strategy_schema
                )
                assert isinstance(node_output_strategy, OpStrategy), (
                    "TupleStrategy is not supported in decomposed sharding propagation"
                )

                # select the first OpSpec with needs_redistribute=False
                # NOTE: there should be only one such OpSpec; revisit if we find exceptions
                node_output_spec = None
                needs_redistribute = False
                for strtg in node_output_strategy.strategies:
                    if strtg.input_specs is None:
                        assert isinstance(strtg.output_specs, DTensorSpec)
                    for idx, input_strategy in enumerate(
                        node_input_strategy_schema.args_strategy
                    ):
                        desired_spec = (
                            strtg.output_spec
                            if strtg.input_specs is None
                            else strtg.input_specs[idx]
                        )
                        if (
                            input_strategy.strategies[0].output_spec.placements
                            != desired_spec.placements
                        ):
                            needs_redistribute = True
                            break
                    if not needs_redistribute:
                        node_output_spec = strtg.output_spec
                        break

                if node_output_spec is None:
                    return None

                node_output_spec.tensor_meta = TensorMeta(
                    shape=node.meta["tensor_meta"].shape,
                    stride=node.meta["tensor_meta"].stride,
                    dtype=node.meta["tensor_meta"].dtype,
                )

                node_to_spec[node] = node_output_spec
            elif node.op == "output":
                output_node = node.args[0]
                graph_output_specs = [node_to_spec[node] for node in output_node]
                return graph_output_specs
            else:
                raise NotImplementedError(f"Unsupported node type: {node.op}")

    def _generate_op_strategy_from_decomposition(
        self, op_schema, decomposition_table
    ) -> OpStrategy:
        # TODO(lty): expand_to_full_mesh_op_strategy hit circular import if put outside
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy
        from torch.utils._pytree import tree_flatten

        mesh = try_find_mesh_from_args(op_schema.op, op_schema.args_schema)

        # generate all possible placements for each DTensor input
        all_possible_schema = []
        flat_args_schema, _ = tree_flatten(
            [op_schema.args_schema, op_schema.kwargs_schema]
        )
        for arg_spec in flat_args_schema:
            if isinstance(arg_spec, DTensorSpec):
                possible_placements = [Replicate()] + [
                    Shard(i) for i in range(arg_spec.ndim)
                ]
                possible_arg_specs = tuple(
                    DTensorSpec(mesh, (p,), arg_spec.tensor_meta)
                    for p in possible_placements
                )
                all_possible_schema.append(possible_arg_specs)
            else:
                all_possible_schema.append((arg_spec,))

        op_gm = self._prepare_op_graph(op_schema, decomposition_table)

        single_mesh_dim_strategies: list[PlacementList] = []
        # for each possible input placement combination, run sharding propagation through the graph
        for graph_input_specs in product(*all_possible_schema):
            graph_output_specs = self._propagate_sharding_through_graph(
                op_gm, graph_input_specs
            )
            if graph_output_specs is not None:
                input_placements: PlacementList = [
                    item.placements[0] if isinstance(item, DTensorSpec) else None
                    for item in graph_input_specs
                ]
                output_placements: PlacementList = [
                    item.placements[0] if isinstance(item, DTensorSpec) else None
                    for item in graph_output_specs
                ]
                output_input_placements: PlacementList = (
                    output_placements + input_placements
                )
                single_mesh_dim_strategies.append(output_input_placements)

        strategy_schema = self._spec_schema_to_strategy_schema(op_schema)

        assert len(single_mesh_dim_strategies) > 0, (
            f"No valid strategy found for {op_schema.op} via decomposition!"
        )

        return expand_to_full_mesh_op_strategy(
            mesh,
            strategy_schema,
            single_mesh_dim_strategies,
            input_index=len(op_schema.op._schema.returns),
        )

    def propagate(self, op_info: OpInfo) -> None:
        # We cannot use an lru cache if we know that inputs will have dynamic shapes,
        # because SymInts are not hashable.
        # This is generally ok because this only happens during tracing in torch.compile,
        # and tracing does not need to be as fast as eagermode DTensor usages.
        if _are_we_tracing():
            output_sharding = self.propagate_op_sharding_non_cached(op_info.schema)
        else:
            output_sharding = cast(
                OutputSharding, self.propagate_op_sharding(op_info.schema)
            )
        op_info.output_sharding = output_sharding

    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # special case op, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        if op_schema.op is aten._local_scalar_dense.default:
            return OutputSharding(None, op_schema)

        out_tensor_meta = self._propagate_tensor_meta_non_cached(op_schema)
        if (
            op_schema.op in self.op_strategy_funcs
            or op_schema.op in self.op_to_decompositions
        ):
            # wrap the op_schema with op strategy for sharding strategy propagation
            op_strategy = self._wrap_with_op_strategy(op_schema)

            if isinstance(op_strategy, OpStrategy):
                # single Op strategy
                output_strategy = self._select_strategy(op_strategy, op_schema)

                # check if we need to redistribute the input
                needs_redistribute = False
                # check if we want to use args value from redistribute_schema
                use_val_from_redistribute_schema = False
                expected_input_specs: list[DTensorSpec] = []

                # in case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if output_strategy.input_specs is None:
                    assert isinstance(output_strategy.output_specs, DTensorSpec)

                for idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        output_strategy.output_spec
                        if output_strategy.input_specs is None
                        else output_strategy.input_specs[idx]
                    )
                    expected_input_specs.append(
                        desired_spec.shallow_copy_with_tensor_meta(
                            input_spec.tensor_meta
                        )
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(expected_input_specs), {}
                    )
                    suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

                # shape and stride args need to be modified for
                # view ops and new factory ops, potentially
                if op_schema.op in self.op_to_shape_and_stride_idx:
                    assert isinstance(output_strategy.output_spec, DTensorSpec)
                    # It happens when the output has the same shape as the input
                    # and the input placements are not all Replicate().
                    if output_strategy.output_spec.is_sharded():
                        schema = suggestion_schema or op_schema
                        assert isinstance(out_tensor_meta, TensorMeta)
                        suggestion_schema = self._adjust_shape_and_stride_args(
                            out_tensor_meta, schema, output_strategy.output_spec
                        )
                        needs_redistribute = True
                        use_val_from_redistribute_schema = True

                # construct output spec for the op
                if op_schema.return_type_tuple_tensor_like():
                    # for ops that return multiple tensors and the output_specs is not
                    # a tuple, we use a tuple of that single output spec as the new
                    # output_specs
                    output_specs: OutputSpecType = output_strategy.output_specs
                    if isinstance(output_specs, DTensorSpec):
                        output_specs = tuple(
                            # create a new DTensorSpec with the same placement as the
                            # output_specs in output_strategy
                            DTensorSpec(
                                mesh=output_specs.mesh,
                                placements=output_specs.placements,
                                tensor_meta=output_specs.tensor_meta,
                            )
                            for _ in range(len(op_schema.op._schema.returns))
                        )
                elif (
                    op_schema.return_type_tensor()
                    or op_schema.return_type_list_tensor_like()
                ):
                    output_specs = output_strategy.output_specs
                else:
                    output_specs = None

                output_sharding = OutputSharding(
                    output_specs,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                    use_val_from_redistribute_schema=use_val_from_redistribute_schema,
                )
            elif isinstance(op_strategy, TupleStrategy):
                # tuple strategy output sharding processing
                # runtime select OpSpec for each TupleStrategy input arg
                selected_strategies: list[OpSpec] = []
                out_spec_list: list[DTensorSpec] = []
                for strategy in op_strategy.children:
                    assert isinstance(strategy, OpStrategy)
                    selected_strategy = self._select_strategy(strategy)
                    selected_strategies.append(selected_strategy)
                    out_spec_list.append(selected_strategy.output_spec)

                needs_redistribute = False
                suggestion_args: list[object] = []
                tensor_or_list_tensor_arg_idx = 0

                for arg in op_schema.args_schema:
                    if (
                        arg
                        and isinstance(arg, (list, tuple))
                        and isinstance(arg[0], DTensorSpec)
                    ):
                        expected_input_spec_list: list[DTensorSpec] = []
                        for idx, arg_spec in enumerate(arg):
                            expected_input_spec = selected_strategies[idx].input_spec(
                                tensor_or_list_tensor_arg_idx
                            )
                            expected_input_spec = (
                                expected_input_spec.shallow_copy_with_tensor_meta(
                                    arg_spec.tensor_meta
                                )
                            )
                            if arg_spec.placements != expected_input_spec.placements:
                                needs_redistribute = True
                            expected_input_spec_list.append(expected_input_spec)
                        suggestion_args.append(
                            tuple(expected_input_spec_list)
                            if isinstance(arg, tuple)
                            else expected_input_spec_list
                        )
                        tensor_or_list_tensor_arg_idx += 1

                    elif isinstance(arg, DTensorSpec):
                        expected_input_spec = selected_strategies[0].input_spec(
                            tensor_or_list_tensor_arg_idx
                        )
                        expected_input_spec = (
                            expected_input_spec.shallow_copy_with_tensor_meta(
                                arg.tensor_meta
                            )
                        )
                        if arg.placements != expected_input_spec.placements:
                            needs_redistribute = True
                        suggestion_args.append(expected_input_spec)
                        tensor_or_list_tensor_arg_idx += 1
                    else:
                        suggestion_args.append(arg)

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(suggestion_args), op_schema.kwargs_schema
                    )

                output_sharding = OutputSharding(
                    tuple(out_spec_list) if out_tensor_meta is not None else None,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                    use_val_from_redistribute_schema=False,
                )
            else:
                raise ValueError("Unsupported op strategy type")

            # associate the output sharding with the output tensor metadata
            new_output_spec = self._create_output_spec_with_new_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            output_sharding.output_spec = new_output_spec
            return output_sharding
        elif op_schema.op in self.op_to_rules:
            # propagate the sharding with rule
            sharding_prop_func = self.op_to_rules[op_schema.op]

            # step 1. there's sharding propagation rule, run
            # sharding propagation to get the output sharding
            try:
                output_sharding = sharding_prop_func(op_schema)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_schema}.\nError: {e}"
                ) from e

            # step 2. if can't get output_spec from sharding
            # propagation (i.e. no rules apply for input
            # placements), we return the output sharding
            # with schema suggestions, which can be used to
            # decide how to do redistribute on inputs
            if output_sharding.output_spec is None:
                if output_sharding.redistribute_schema is None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_schema}!"
                    )
                else:
                    # we do auto redistribute on inputs if necessary
                    # run sharding propagation again with suggested schema
                    propagation_res = sharding_prop_func(
                        output_sharding.redistribute_schema
                    )
                    # we set the output sharding with the new propagation result
                    # so that dispatching know both output_spec and redistribute_schema
                    # exist, which indicates a reshard is needed
                    output_sharding.output_spec = propagation_res.output_spec
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            new_output_spec = self._create_output_spec_with_new_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            output_sharding.output_spec = new_output_spec

            return output_sharding
        else:
            raise NotImplementedError(
                f"Operator {op_schema.op} does not have a sharding strategy registered."
            )

    def _select_strategy(
        self, strategy: OpStrategy, op_schema: Optional[OpSchema] = None
    ) -> OpSpec:
        if len(strategy.strategies) == 1:
            # short cut with only one possible OpSpec
            return strategy.strategies[0]

        op_spec_costs: list[float] = []
        no_redistribute_strategy_index: int = -1
        for strategy_idx, op_spec in enumerate(strategy.strategies):
            assert op_spec.redistribute_cost is not None, (
                "must set redistribute cost each OpSpec!"
            )
            redistribute_cost = sum(chain.from_iterable(op_spec.redistribute_cost))
            op_spec_costs.append(redistribute_cost)

            # If there's no redistribute cost, we record the index of the strategy
            # which doesn't need redistribute.
            # TODO: Currently this only applies to OpStrategy selection. Requires extra
            # logic to make it work for TupleStrategy, if needed.
            if op_schema is not None and redistribute_cost == 0:
                needs_redistribute = False
                for spec_idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        op_spec.output_spec
                        if op_spec.input_specs is None
                        else op_spec.input_specs[spec_idx]
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True
                        break

                if not needs_redistribute:
                    no_redistribute_strategy_index = strategy_idx

        # for eager execution, we just select the one with the minimal redistribute cost
        min_cost = min(op_spec_costs)
        if min_cost < 0:
            # If there's negative cost, we select the one with the minimal cost,
            # even if this means we need to redistribute, e.g. via local chunking.
            # E.g. this can happen for ops in self.op_to_shape_and_stride_idx
            # when the inputs / outputs are sharded.
            selected_strategy_index = op_spec_costs.index(min_cost)
        elif min_cost == 0 and no_redistribute_strategy_index != -1:
            # If there's no redistribute cost, we select the one with no redistribute.
            selected_strategy_index = no_redistribute_strategy_index
        else:
            selected_strategy_index = op_spec_costs.index(min_cost)

        return strategy.strategies[selected_strategy_index]

    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
    ) -> OpSchema:
        shape_stride_idx = self.op_to_shape_and_stride_idx[schema.op]
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx = shape_stride_idx
            stride_idx = None

        expected_input_schema = list(schema.args_schema)
        # adjust shape to be the same as that of the _local_tensor
        # of the DTensor input arg at index 0, which is inferred
        expected_input_schema[shape_idx], _ = compute_local_shape_and_global_offset(
            out_tensor_meta.shape, spec.mesh, spec.placements
        )

        # adjust the stride arg for aten.new_empty_strided.default
        if stride_idx:
            expected_input_schema[stride_idx] = compute_local_stride(
                out_tensor_meta.stride, spec.mesh, spec.placements
            )

        return OpSchema(schema.op, tuple(expected_input_schema), schema.kwargs_schema)
