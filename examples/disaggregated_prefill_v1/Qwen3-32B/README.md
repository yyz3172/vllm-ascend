# 单机 4+4 卡 PD 分离（Qwen3-32B）

本目录为单机 A2（8 卡）上 4 卡 Prefill + 4 卡 Decode 的 Shell 模板，与上级目录的 Python 启动脚本配合使用。

## 文件说明

| 文件 | 用途 |
|------|------|
| `run_prefill_4cards.sh` | Prefill 进程用（TP=4，卡 0–3），由 `launch_prefill_single_node_4cards.py` 调用 |
| `run_decode_4cards.sh`  | Decode 进程用（TP=4，卡 4–7），由 `launch_decode_single_node_4cards.py` 调用 |

## 使用前修改

在两个 `.sh` 文件顶部修改：

- `nic_name`：网卡名（如 `eth0`）
- `local_ip`：本机 IP（多机或代理需填实际 IP）
- `MODEL_PATH`：模型路径（如 `/ds/models/Qwen3-32B`）

## 启动方式

在上级目录 `examples/disaggregated_prefill_v1` 下执行：

```bash
# 1. 先启动 Prefill
python launch_prefill_single_node_4cards.py

# 2. 再启动 Decode（另开终端或 nohup）
python launch_decode_single_node_4cards.py
```

详细说明见文档 [large_scale_ep.md](../../../docs/source/user_guide/feature_guide/large_scale_ep.md) 中「单机 A2 部署」一节。
