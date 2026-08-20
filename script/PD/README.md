# vLLM-Ascend PD 分离部署脚本

本目录包含 vLLM-Ascend Prefill/Decode (PD) 分离部署的启动、性能测试与精度测试脚本。

## 目录结构

```
PD/
├── README.md              # 本文档
├── start_pd.sh            # PD 服务启停入口
├── start_bench.sh         # 性能压测脚本
├── start_eval.sh          # 精度评测脚本
├── pd_service_ctl.py      # PD 服务编排工具（被 start_pd.sh 调用）
└── 1P1_1D1/               # 1 Prefill + 1 Decode 拓扑配置
    ├── run_prefill.sh      # Prefill 实例启动脚本
    ├── run_decode.sh       # Decode 实例启动脚本
    └── pd_proxy.py         # 负载均衡代理
```

## 环境变量说明

以下环境变量在 `start_pd.sh` 中配置，控制 PD 服务的行为：

| 环境变量 | 示例值 | 说明 |
|---|---|---|
| `PROXY_PORT` | `9878` | 代理服务监听端口，对外提供 OpenAI 兼容 API |
| `MODEL_PATH` | `/root/l00856060/model/Qwen3-0.6B/` | 模型权重路径 |
| `ASCEND_RT_VISIBLE_DEVICES_PREFILL` | `3` | Prefill 实例使用的 NPU 设备 ID |
| `ASCEND_RT_VISIBLE_DEVICES_DECODE` | `4` | Decode 实例使用的 NPU 设备 ID |
| `VLLM_VENV` | `/root/devcommon/w30022782/env/.venv/` | vLLM Python 虚拟环境路径 |
| `TYPE` | `BASE` / `TURBOQUANT` | 量化模式：`BASE` 为标准精度，`TURBOQUANT` 启用 TurboQuant 量化 |

---

## start_pd.sh — PD 服务启停

通过 `pd_service_ctl.py` 编排 Prefill、Decode 及代理进程的启动与停止。

### 用法

```bash
# 启动 PD 服务（默认 BASE 模式）
bash start_pd.sh start

# 启动 PD 服务（TurboQuant 量化模式）
bash start_pd.sh start turboquant

# 停止 PD 服务
bash start_pd.sh stop
```

### 参数说明

| 参数 | 位置 | 说明 |
|---|---|---|
| `start` | `$1` | 操作类型：`start` 启动服务，其他值均执行 `stop` |
| `turboquant` | `$2`（可选） | 仅在 `start` 时生效，传入此值启用 TurboQuant 量化模式 |

### PD 模式

当前默认使用 `1P1_1D1` 拓扑（1 个 Prefill 实例 + 1 个 Decode 实例）。启动流程为：

1. 拉起 Prefill 实例（`run_prefill.sh`）
2. 拉起 Decode 实例（`run_decode.sh`）
3. 拉起负载均衡代理（`pd_proxy.py`），对外暴露 `PROXY_PORT` 端口

### 进程 PID 文件

服务启动后 PID 文件位于 `/tmp/` 下：

| 文件 | 内容 |
|---|---|
| `/tmp/vllm_prefill.pid` | Prefill 进程 PID |
| `/tmp/vllm_decode.pid` | Decode 进程 PID |
| `/tmp/vllm_proxy.pid` | 代理进程 PID |

---

## start_bench.sh — 性能压测

使用 `vllm bench serve` 对 PD 服务进行吞吐/延迟压测。**运行前需确保 PD 服务已启动。**

### 用法

```bash
# 先启动 PD 服务
bash start_pd.sh start

# 运行性能压测
bash start_bench.sh
```

### 参数说明

压测脚本内嵌参数如下，按需直接修改脚本：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend` | `vllm` | 后端类型 |
| `--model` | `/root/l00856060/model/Qwen3-0.6B/` | 模型路径（需与 PD 服务加载的模型一致） |
| `--port` | `9878` | 目标服务端口（对应 `PROXY_PORT`） |
| `--endpoint` | `/v1/completions` | API 端点 |
| `--dataset-name` | `random` | 数据集类型：`random` 为随机生成请求 |
| `--input-len` | `10` | 输入 token 长度 |
| `--output-len` | `200` | 输出 token 长度 |
| `--num_prompt` | `1000` | 发送请求总数 |

### 示例：修改压测参数

编辑 `start_bench.sh`，调整输入输出长度和并发数：

```bash
vllm bench serve \
  --backend vllm \
  --model /root/l00856060/model/Qwen3-0.6B/ \
  --port 9878 \
  --endpoint /v1/completions \
  --dataset-name random \
  --input-len 1024 \
  --output-len 512 \
  --num_prompt 5000
```

---

## start_eval.sh — 精度评测

使用 [evalscope](https://github.com/modelscope/evalscope) 对 PD 服务进行模型精度评测。**运行前需确保 PD 服务已启动。**

### 用法

```bash
# 先启动 PD 服务
bash start_pd.sh start

# 运行精度评测
bash start_eval.sh
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `/root/l00856060/model/Qwen3-0.6B/` | 模型路径（仅用于标识，不影响服务端） |
| `--api-url` | `http://127.0.0.1:9878/v1/chat/completions` | PD 服务 chat completions 接口地址 |
| `--api-key` | `EMPTY` | API Key（本地部署通常无需真实 Key） |
| `--datasets` | `gsm8k` | 评测数据集名称，支持 `gsm8k`、`mmlu`、`ceval` 等 |
| `--limit` | `10` | 评测样本数量（用于快速验证；正式评测可增大或移除此参数） |

### 示例：完整 MMLU 评测

编辑 `start_eval.sh`：

```bash
evalscope eval \
  --model /root/l00856060/model/Qwen3-0.6B/ \
  --api-url http://127.0.0.1:9878/v1/chat/completions \
  --api-key EMPTY \
  --datasets mmlu
```

---

## 典型使用流程

```bash
# 1. 按需修改 start_pd.sh 中的环境变量（模型路径、NPU 设备等）
vim start_pd.sh

# 2. 启动 PD 服务
bash start_pd.sh start

# 3. 运行性能压测
bash start_bench.sh

# 4. 运行精度评测
bash start_eval.sh

# 5. 测试完毕，停止服务
bash start_pd.sh stop
```
