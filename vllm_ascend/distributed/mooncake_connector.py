# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible import shim.

Some deployments still reference the legacy module path
`vllm_ascend.distributed.mooncake_connector`. The implementation lives under
`vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector` in this version.
"""

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (  # noqa: F401
    MooncakeConnector,
)

# Legacy alias: older configs may request `MooncakeConnectorV1`.
MooncakeConnectorV1 = MooncakeConnector

