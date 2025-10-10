# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.tensor._ops._decomposed_ops import register_op_decomposition
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


aten = torch.ops.aten


# a utility to remove op from the op_strategy_funcs dict and
# register decomposition for testing purposes
def _register_op_decomposition(op):
    DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(op, None)
    register_op_decomposition(op)


class DistDecomposedOpsTest(DTensorTestBase):
    # test op decomposition with multiple outputs
    @with_comms
    def test_decomposed_aminmax(self):
        device_mesh = self.build_device_mesh()

        x = torch.rand(4, 8, device=self.device_type, dtype=torch.float32)
        y1, y2 = torch.aminmax(x)

        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        y1_dt, y2_dt = torch.aminmax(x_dt)

        self.assertTrue(y1_dt.placements[0].is_replicate())
        self.assertTrue(y2_dt.placements[0].is_replicate())
        self.assertEqual(y1_dt.to_local(), y1)
        self.assertEqual(y2_dt.to_local(), y2)

        for dim in range(len(x.shape)):
            dist_x = distribute_tensor(x, device_mesh, [Shard(dim)])
            y1_dt, y2_dt = torch.aminmax(dist_x)
            self.assertEqual(y1_dt.placements[0], Partial("min"))
            self.assertEqual(y2_dt.placements[0], Partial("max"))
            self.assertEqual(y1_dt.full_tensor(), y1)
            self.assertEqual(y2_dt.full_tensor(), y2)

    # test op decomposition with possibly infeasible sharding input
    # i.e. when the input is sharded on the softmax_dim
    @with_comms
    def test_decompose_log_softmax_fwd(self):
        _register_op_decomposition(aten._log_softmax.default)
        assert (
            aten._log_softmax.default
            not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        )

        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 8, device=self.device_type)
        softmax_dim = 1
        y = torch.nn.functional.log_softmax(x, dim=softmax_dim)

        # replicated input
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        y_dt = torch.nn.functional.log_softmax(x_dt, dim=softmax_dim)

        self.assertTrue(y_dt.placements[0].is_replicate())
        self.assertEqual(y_dt.to_local(), y)

        # sharded input
        for shard_dim in range(len(x.shape)):
            x_dt = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            y_dt = torch.nn.functional.log_softmax(x_dt, dim=softmax_dim)
            # we can preserve the sharding, even when the input is sharded
            # on the softmax dim as long as we all-reduce partials (i.e., max, sum)
            # across shards prior to usage
            if shard_dim == softmax_dim:
                self.assertTrue(y_dt.placements[0].is_replicate())
            else:
                self.assertTrue(y_dt.placements[0].is_shard(dim=shard_dim))
            self.assertEqual(y_dt.full_tensor(), y)

    @with_comms
    def test_decompose_gelu(self):
        _register_op_decomposition(aten.gelu.default)
        assert (
            aten.gelu.default
            not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        )

        device_mesh = self.build_device_mesh()
        x = torch.rand(8, 8)
        y = torch.nn.functional.gelu(x)

        # replicated
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        y_dt = torch.nn.functional.gelu(x_dt)
        self.assertTrue(y_dt.placements[0].is_replicate())
        self.assertEqual(y_dt.full_tensor(), y)

        # sharded input - GELU is elementwise so should preserve all sharding patterns
        for shard_dim in range(len(x.shape)):
            x_dt = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            y_dt = torch.nn.functional.gelu(x_dt)

            # GELU is purely elementwise, so should preserve sharding on any dimension
            self.assertTrue(y_dt.placements[0].is_shard(dim=shard_dim))
            self.assertEqual(y_dt.full_tensor(), y)

    @with_comms
    def test_decompose__softmax_out(self):
        _register_op_decomposition(aten._softmax.out)
        assert (
            aten._softmax.out
            not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        )

        mesh = self.build_device_mesh()
        x = torch.rand(8, 8, device=self.device_type)
        softmax_dim = 1
        ref = torch.nn.functional.softmax(x, dim=softmax_dim)

        # replicated
        x_dt = distribute_tensor(x, mesh, [Replicate()])
        out_dt = distribute_tensor(torch.empty_like(x), mesh, [Replicate()])
        y_dt = aten._softmax.out(x_dt, softmax_dim, False, out=out_dt)
        self.assertTrue(y_dt.placements[0].is_replicate())
        self.assertEqual(y_dt.to_local(), ref)

        # sharded
        for shard_dim in range(x.ndim):
            x_dt = distribute_tensor(x, mesh, [Shard(shard_dim)])

            # choose the correct expected placement for the OUT buffer
            if shard_dim == softmax_dim:
                out_placement = [Replicate()]
            else:
                out_placement = [Shard(shard_dim)]

            out_dt = distribute_tensor(torch.empty_like(x), mesh, out_placement)
            y_dt = aten._softmax.out(x_dt, softmax_dim, False, out=out_dt)

            if shard_dim == softmax_dim:
                self.assertTrue(y_dt.placements[0].is_replicate())
            else:
                self.assertTrue(y_dt.placements[0].is_shard(dim=shard_dim))

            self.assertEqual(y_dt.full_tensor(), ref)


if __name__ == "__main__":
    run_tests()
