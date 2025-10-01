# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# from typing import cast, List, Optional, Sequence, Tuple

from typing import Callable, Mapping, Optional

import torch
from torch._decomp import decomposition_table as decomp_table, get_decompositions
from torch._ops import OpOverload
from torch.distributed._tensor.api import DTensor


aten = torch.ops.aten


def register_op_decomposition(
    op,
    decomposition_table: Optional[Mapping[OpOverload, Callable]] = None,
    schema_info=None,
):
    if op not in decomp_table:
        raise RuntimeError(
            f"{op} is not registered in torch._decomp.decomposition_table"
        )

    DTensor._op_dispatcher.sharding_propagator.register_op_decomposition(
        op,
        decomposition_table or get_decompositions((op,)),
        schema_info,
    )


register_op_decomposition(aten.aminmax.default)
