# 单机 PD 分离（Qwen3-32B）

本目录为单机 PD 分离的 Shell 模板，与上级目录的 Python 启动脚本配合使用。支持三种规格：

- **4+4 卡**：8 卡机，4 卡 Prefill + 4 卡 Decode（卡 0–3 / 4–7）
- **2+2 卡**：4 卡机，2 卡 Prefill + 2 卡 Decode（卡 0–1 / 2–3）
- **1+1 卡**：2 卡机，1 卡 Prefill + 1 卡 Decode（卡 0 / 1）

**2+2 卡**与**1+1 卡**的配置已统一在对应 shell 顶部的「配置区」，**直接执行对应 shell 即可**，无 Python 启动脚本。

## 文件说明

| 文件 | 用途 |
|------|------|
| `run_prefill_4cards.sh` | Prefill（TP=4，卡 0–3），由 `launch_prefill_single_node_4cards.py` 调用 |
| `run_decode_4cards.sh`  | Decode（TP=4，卡 4–7），由 `launch_decode_single_node_4cards.py` 调用 |
| `run_prefill_2cards.sh` | Prefill（TP=2，卡 0–1），直接执行 |
| `run_decode_2cards.sh`  | Decode（TP=2，卡 2–3），直接执行 |
| `run_prefill_1card.sh`  | Prefill（TP=1，卡 0），直接执行 |
| `run_decode_1card.sh`   | Decode（TP=1，卡 1），直接执行 |

## 使用前修改

- **4+4 卡**：在 `run_prefill_4cards.sh` / `run_decode_4cards.sh` 顶部修改 `nic_name`、`local_ip`、`model_path`（Decode 的 `$8` 由 launch 脚本传入）。
- **2+2 卡**：在 `run_prefill_2cards.sh` / `run_decode_2cards.sh` 顶部的 **「========== 配置区 ==========」** 内修改即可。
- **1+1 卡**：在 `run_prefill_1card.sh` / `run_decode_1card.sh` 顶部的 **「========== 配置区 ==========」** 内修改即可，包含 `nic_name`、`local_ip`、`model_path`、`dp_port`、`engine_port`、`visible_devices` 等。

## 启动方式

在上级目录 `examples/disaggregated_prefill_v1` 下执行。

**4+4 卡（8 卡机）：**

```bash
python launch_prefill_single_node_4cards.py
# 另开终端或 nohup
python launch_decode_single_node_4cards.py
```

**2+2 卡（4 卡机）：** 直接执行 shell（无 launch 脚本）

```bash
cd examples/disaggregated_prefill_v1/Qwen3-32B
bash run_prefill_2cards.sh
# 另开终端或 nohup
bash run_decode_2cards.sh
```

**1+1 卡（2 卡机）：** 直接执行 shell（无 launch 脚本）

```bash
cd examples/disaggregated_prefill_v1/Qwen3-32B
bash run_prefill_1card.sh
# 另开终端或 nohup
bash run_decode_1card.sh
```

**重要：** Qwen3-32B bf16 仅权重约 64 GiB，**单卡显存需 ≥64 GiB** 才能 1+1 部署。若单卡约 61 GiB 会报 `NPU out of memory`（在加载 lm_head 时 OOM）。此时请改用 **2+2 卡**（`run_prefill_2cards.sh` / `run_decode_2cards.sh`），或换用 Qwen3-8B 的 1+1 脚本。调低 `--max-model-len` 无法解决「权重装不下」的 OOM。

Proxy 启动方式相同（端口 9000/9010），详见文档 [large_scale_ep.md](../../../docs/source/user_guide/feature_guide/large_scale_ep.md) 中「单机 A2 部署」一节。
