# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible import shim for layerwise Mooncake connector."""

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: F401
    MooncakeLayerwiseConnector,
)

# Legacy alias: older configs may request `MooncakeLayerwiseConnectorV1`.
MooncakeLayerwiseConnectorV1 = MooncakeLayerwiseConnector

