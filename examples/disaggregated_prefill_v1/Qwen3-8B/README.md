# 单机 PD 分离（Qwen3-8B，1 卡 P + 1 卡 D）

本目录为 **Qwen3-8B** 单机 **1 卡 Prefill + 1 卡 Decode**（1P+1D）的 Shell 脚本。**所有配置统一在两个 shell 顶部的「配置区」**，直接执行对应 sh 即可，无需 Python 启动脚本。显存占用小，适合 2 卡及以上环境。

## 文件说明

| 文件 | 用途 |
|------|------|
| `run_prefill_1card.sh` | Prefill（TP=1，单卡），直接执行 |
| `run_decode_1card.sh`   | Decode（TP=1，单卡），直接执行 |

## 配置（仅改 shell 顶部配置区）

在 **`run_prefill_1card.sh`** 与 **`run_decode_1card.sh`** 顶部的 **「========== 配置区 ==========」** 内修改即可：

| 变量 | 说明 | Prefill 默认 | Decode 默认 |
|------|------|--------------|-------------|
| `nic_name` | 网卡名 | eth0 | 同左 |
| `local_ip` | 本机 IP | 172.17.0.2 | 同左 |
| `model_path` | Qwen3-8B 路径 | /root/autodl-tmp/models/Qwen3-8B | 同左 |
| `dp_port` | DP 通信端口 | 13395 | 13495（须与 Prefill 不同） |
| `engine_port` | vLLM 服务端口 | 9000 | 9010 |
| `visible_devices` | 使用的 NPU 卡号 | "0" | "1" |
| `transfer_engine_lib_path` | `libtransfer_engine.so` 所在目录（报错时必填） | 见下方 | 同左 |
| `python_lib_path` | 当前 Python 的 lib 目录（报 libpython3.11.so.1.0 时必填） | 见下方 | 同左 |

若启动时报 **`libtransfer_engine.so: cannot open shared object file`**，请在配置区填写 `transfer_engine_lib_path`，指向该 .so 所在目录（例如 Mooncake 编译产出目录或 Ascend 安装下的 `lib64`）。可用 `find /usr -name "libtransfer_engine.so" 2>/dev/null` 查找。

若报 **`libpython3.11.so.1.0: cannot open shared object file`**，请填写 `python_lib_path` 为当前运行 vLLM 的 Python 所在环境的 **lib 目录**（例如 venv：`/root/autodl-tmp/py_venv/vllm2/lib`；uv 管理的 Python：`/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib`）。可用 `find /root -name "libpython3.11.so.1.0" 2>/dev/null` 查找后取所在目录。

## 启动方式

直接执行 shell（无 launch 脚本）：

```bash
cd examples/disaggregated_prefill_v1/Qwen3-8B
bash run_prefill_1card.sh
# 另开终端或 nohup
bash run_decode_1card.sh
```

Proxy 启动方式与 2+2 / 4+4 相同（Prefill 端口 9000，Decode 端口 9010），详见 [large_scale_ep.md](../../../docs/source/user_guide/feature_guide/large_scale_ep.md)。
