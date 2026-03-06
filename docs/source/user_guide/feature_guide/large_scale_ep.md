# Distributed DP Server With Large Scale Expert Parallelism

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation in the large scale **Expert  Parallelism (EP)** scenario. To achieve better performance，the distributed DP server is applied in vLLM-Ascend. In the PD separation scenario, different optimization strategies can be implemented based on the distinct characteristics of PD nodes, thereby enabling more flexible model deployment. \
Take the deepseek model as an example, use 8 Atlas 800T A3 servers to deploy the model. Assume the ip of the servers start from 192.0.0.1, and end by 192.0.0.8. Use the first 4 servers as prefiller nodes and the last 4 servers as decoder nodes. And the prefiller nodes deployed as master node independently, the decoder nodes set 192.0.0.5 node to be the master node.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements:

- The physical machines must be located on the same WLAN, with network connectivity.
- All NPUs must be interconnected. For the Atlas A2 generation, intra-node connectivity is via HCCS, and inter-node connectivity is via RDMA. For the Atlas A3 generation, both intra-node and inter-node connectivity are via HCCS.

### Verification Process:

:::::{tab-set}
::::{tab-item} A3

1. Single Node Verification:

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

```bash
 # Check the remote switch ports
 for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done
 # Get the link status of the Ethernet ports (UP or DOWN)
 for i in {0..15}; do hccn_tool -i $i -link -g ; done
 # Check the network health status
 for i in {0..15}; do hccn_tool -i $i -net_health -g ; done
 # View the network detected IP configuration
 for i in {0..15}; do hccn_tool -i $i -netdetect -g ; done
 # View gateway configuration
 for i in {0..15}; do hccn_tool -i $i -gateway -g ; done
 # View NPU network configuration
 cat /etc/hccn.conf
```

2. Get NPU IP Addresses

```bash
for i in {0..15}; do hccn_tool -i $i -vnic -g;done
```

3. Get superpodid and SDID

```bash
for i in {0..15}; do npu-smi info -t spod-info -i $i -c 0;npu-smi info -t spod-info -i $i -c 1;done
```

4. Cross-Node PING Test

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address x.x.x.x;done
```

::::

::::{tab-item} A2

1. Single Node Verification:

Execute the following commands on each node in sequence. The results must all be `success` and the status must be `UP`:

```bash
# Check the remote switch ports
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
# Get the link status of the Ethernet ports (UP or DOWN)
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# Check the network health status
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# View the network detected IP configuration
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# View gateway configuration
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
# View NPU network configuration
cat /etc/hccn.conf
```

2. Get NPU IP Addresses

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g;done
```

3. Cross-Node PING Test

```bash
# Execute on the target node (replace 'x.x.x.x' with actual npu ip address)
for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
```

::::

:::::

## Large Scale EP model deployment

### Generate script with configurations

In the PD separation scenario, we provide a optimized configuration. You can use the following shell script for configuring the prefiller and decoder nodes respectively.

:::::{tab-set}

::::{tab-item} Prefiller node

```shell
# run_dp_template.sh
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name="xxxx"
local_ip="xxxx"

# basic configuration for HCCL and connection
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256

# obtain parameters from distributed DP server
export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7

#pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

# enable the distributed DP server
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

# The w8a8 weight can obtained from https://www.modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8
# "--additional-config" is used to enable characteristics from vllm-ascend
vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
    --host 0.0.0.0 \
    --port $6 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --max-model-len 17000 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
      "kv_buffer_device": "npu",
      "kv_role": "kv_producer",
      "kv_parallel_size": "1",
      "kv_port": "20001",
      "engine_id": "0",
      "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
    --additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}'
```

::::

::::{tab-item} Decoder node

```shell
# run_dp_template.sh
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name="xxxx"
local_ip="xxxx"

# basic configuration for HCCL and connection
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024

# obtain parameters from distributed DP server
export VLLM_DP_SIZE=$1
export VLLM_DP_MASTER_IP=$2
export VLLM_DP_MASTER_PORT=$3
export VLLM_DP_RANK_LOCAL=$4
export VLLM_DP_RANK=$5
export VLLM_DP_SIZE_LOCAL=$7

#pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

# enable the distributed DP server
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

# The w8a8 weight can obtained from https://www.modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8
# "--additional-config" is used to enable characteristics from vllm-ascend
vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
    --host 0.0.0.0 \
    --port $6 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_r1 \
    --max-model-len 17000 \
    --max-num-batched-tokens 256 \
    --trust-remote-code \
    --max-num-seqs 28 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_parallel_size": "1",
        "kv_port": "20001",
        "engine_id": "0",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
        }' \
    --additional-config '{"enable_weight_nz_layout":true}'
```

::::

:::::

### Start Distributed DP Server for prefill-decode disaggregation

Execute the following Python file on all nodes to use the distributed DP server. (We recommend using this feature on the v0.9.1 official release)

:::::{tab-set}

::::{tab-item} Prefiller node

```python
import multiprocessing
import os
import sys
dp_size = 2 # total number of DP engines for decode/prefill
dp_size_local = 2 # number of DP engines on the current node
dp_rank_start = 0 # starting DP rank for the current node
# dp_ip is different on prefiller nodes in this example
dp_ip = "192.0.0.1" # master node ip for DP communication
dp_port = 13395 # port used for DP communication
engine_port = 9000 # starting port for all DP groups on the current node
template_path = "./run_dp_template.sh"
if not os.path.exists(template_path):
  print(f"Template file {template_path} does not exist.")
  sys.exit(1)
def run_command(dp_rank_local, dp_rank, engine_port_):
  command = f"bash ./run_dp_template.sh {dp_size} {dp_ip} {dp_port} {dp_rank_local} {dp_rank} {engine_port_} {dp_size_local}"
  os.system(command)
processes = []
for i in range(dp_size_local):
  dp_rank = dp_rank_start + i
  dp_rank_local = i
  engine_port_ = engine_port + i
  process = multiprocessing.Process(target=run_command, args=(dp_rank_local, dp_rank, engine_port_))
  processes.append(process)
  process.start()
for process in processes:
  process.join()
```

::::

::::{tab-item} Decoder node

```python
import multiprocessing
import os
import sys
dp_size = 64 # total number of DP engines for decode/prefill
dp_size_local = 16 # number of DP engines on the current node
dp_rank_start = 0 # starting DP rank for the current node. e.g. 0/16/32/48
# dp_ip is the same on decoder nodes in this example
dp_ip = "192.0.0.5" # master node ip for DP communication.
dp_port = 13395 # port used for DP communication
engine_port = 9000 # starting port for all DP groups on the current node
template_path = "./run_dp_template.sh"
if not os.path.exists(template_path):
  print(f"Template file {template_path} does not exist.")
  sys.exit(1)
def run_command(dp_rank_local, dp_rank, engine_port_):
  command = f"bash ./run_dp_template.sh {dp_size} {dp_ip} {dp_port} {dp_rank_local} {dp_rank} {engine_port_} {dp_size_local}"
  os.system(command)
processes = []
for i in range(dp_size_local):
  dp_rank = dp_rank_start + i
  dp_rank_local = i
  engine_port_ = engine_port + i
  process = multiprocessing.Process(target=run_command, args=(dp_rank_local, dp_rank, engine_port_))
  processes.append(process)
  process.start()
for process in processes:
  process.join()
```

::::

:::::

Note that the prefiller nodes and the decoder nodes may have different configurations. In this example, each prefiller node deployed as master node independently, but all decoder nodes take the first node as the master node. So it leads to difference in 'dp_size_local' and 'dp_rank_start'

## 单机 A2 部署（4 卡 Prefill + 4 卡 Decode）

若只有一台 Atlas 800T A2 服务器（8 张 NPU），可以将 4 张卡作为 Prefill 节点、4 张卡作为 Decode 节点，在同一台机器上部署 DP 分离服务。做法是：

- 使用 **ASCEND_RT_VISIBLE_DEVICES** 将设备划分：Prefill 进程只使用卡 0–3，Decode 进程只使用卡 4–7。
- Prefill 侧：1 个 DP 引擎，`tensor-parallel-size=4`（4 卡）。
- Decode 侧：**1 个 DP 引擎，`tensor-parallel-size=4`**（单进程用满 4 卡 4–7），显存更充裕，适合 32B 等大模型。若使用 7B 等小模型且希望多副本，可自行改为 TP=1、dp_size=4（4 个单卡副本）。

### 1. 单机脚本与模板说明

单机 4+4 卡部署已拆分为 **Prefill / Decode 两套脚本**，无需共用一份 `run_dp_template.sh`：

| 角色 | 启动入口 | Shell 模板 |
|------|----------|------------|
| Prefill（4 卡） | `launch_prefill_single_node_4cards.py` | `Qwen3-32B/run_prefill_4cards.sh` |
| Decode（4 卡） | `launch_decode_single_node_4cards.py` | `Qwen3-32B/run_decode_4cards.sh` |
| Prefill（2 卡） | 直接执行 | `Qwen3-32B/run_prefill_2cards.sh` |
| Decode（2 卡） | 直接执行 | `Qwen3-32B/run_decode_2cards.sh` |
| Prefill（1 卡，Qwen3-8B） | 直接执行 | `Qwen3-8B/run_prefill_1card.sh` |
| Decode（1 卡，Qwen3-8B） | 直接执行 | `Qwen3-8B/run_decode_1card.sh` |

- **Prefill 模板**（`run_prefill_4cards.sh`）：`--tensor-parallel-size 4`，`kv_role: kv_producer`，`--enforce-eager`；`ASCEND_RT_VISIBLE_DEVICES` 由 Python 启动脚本设置，脚本内不写死。
- **Decode 模板**（`run_decode_4cards.sh`）：`--tensor-parallel-size 4`，`--nnodes 1`，`kv_connector_extra_config` 为 `"prefill": {"dp_size": 1, "tp_size": 4}, "decode": {"dp_size": 1, "tp_size": 4}`；**第 8 个参数 $8** 为 `ASCEND_RT_VISIBLE_DEVICES`（单副本时由启动脚本传入 `"4,5,6,7"`）。
- **2 卡配置**（仅 4 张卡环境）：使用 `run_prefill_2cards.sh` / `run_decode_2cards.sh`，TP=2，Prefill 用卡 0–1、Decode 用卡 2–3；`kv_connector_extra_config` 中 `tp_size` 均为 2。**直接执行上述 shell 即可**，无 Python 启动脚本。
- **1 卡 P + 1 卡 D（Qwen3-8B）**：使用 `Qwen3-8B/run_prefill_1card.sh` / `run_decode_1card.sh`，TP=1，Prefill 用卡 0、Decode 用卡 1；`kv_connector_extra_config` 中 `tp_size` 均为 1。**直接执行上述 shell 即可**，无 Python 启动脚本。显存占用小，适合 2 卡机。
- 使用前在对应 shell 中修改 `nic_name`、`local_ip`、`model_path`。

### 2. 单机 Prefill / Decode 启动脚本

- **Prefill**：直接使用 [launch_prefill_single_node_4cards.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/launch_prefill_single_node_4cards.py)，单进程调用 `Qwen3-32B/run_prefill_4cards.sh`。脚本逻辑见仓库该文件。
- **Decode**：直接使用 [launch_decode_single_node_4cards.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/launch_decode_single_node_4cards.py)，单进程调用 `Qwen3-32B/run_decode_4cards.sh`。脚本逻辑见仓库该文件。
- Shell 模板见 [Qwen3-32B/run_prefill_4cards.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/Qwen3-32B/run_prefill_4cards.sh) 与 [Qwen3-32B/run_decode_4cards.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/Qwen3-32B/run_decode_4cards.sh)。使用前在脚本内修改 `nic_name`、`local_ip`、`model_path`。

### 3. 启动顺序与 Proxy

1. 先在本机启动 **Prefill**：在 `examples/disaggregated_prefill_v1` 下执行 `python launch_prefill_single_node_4cards.py`（会调用 `Qwen3-32B/run_prefill_4cards.sh`）。
2. 再启动 **Decode**：在同一目录执行 `python launch_decode_single_node_4cards.py`（会调用 `Qwen3-32B/run_decode_4cards.sh`）。
3. 启动负载均衡代理（示例，单副本 Decode 仅 1 个端口）：

```shell
python load_balance_proxy_server_example.py \
  --port 8000 \
  --host 0.0.0.0 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 9000 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 9010
```

请求发往代理的 8000 端口即可。

**其他建议：**（1）先确认 Prefill 日志中出现 “Creating v1 connector … kv_producer” 且无报错后再起 Decode；（2）两进程建议在不同终端或后台运行（如 `nohup ... &`），便于分别查看日志；（3）若更换模型或卡数，需同步修改对应 shell 中的 `model_path`、`--tensor-parallel-size` 及 `kv_connector_extra_config` 中的 dp_size/tp_size。

### 4. 单机配置小结

**4 卡 Prefill + 4 卡 Decode（8 卡机）：**

| 项目           | Prefill（单机 4 卡） | Decode（单机 4 卡） |
|----------------|----------------------|----------------------|
| ASCEND_RT_VISIBLE_DEVICES | 0,1,2,3（由启动脚本设置） | 4,5,6,7（单进程 4 卡） |
| DP 引擎数       | 1                     | 1                     |
| tensor-parallel-size | 4                 | 4                     |
| 引擎端口         | 9000                  | 9010                  |

**2 卡 Prefill + 2 卡 Decode（4 卡机）：**

| 项目           | Prefill（单机 2 卡） | Decode（单机 2 卡） |
|----------------|----------------------|----------------------|
| ASCEND_RT_VISIBLE_DEVICES | 0,1（由启动脚本设置） | 2,3（单进程 2 卡） |
| DP 引擎数       | 1                     | 1                     |
| tensor-parallel-size | 2                 | 2                     |
| 引擎端口         | 9000                  | 9010                  |

**1 卡 Prefill + 1 卡 Decode（Qwen3-8B，2 卡机）：**

| 项目           | Prefill（单机 1 卡） | Decode（单机 1 卡） |
|----------------|----------------------|----------------------|
| ASCEND_RT_VISIBLE_DEVICES | 0（由启动脚本设置） | 1（单卡） |
| DP 引擎数       | 1                     | 1                     |
| tensor-parallel-size | 1                 | 1                     |
| 引擎端口         | 9000                  | 9010                  |
| 模型/模板       | Qwen3-8B，`Qwen3-8B/run_prefill_1card.sh` | `Qwen3-8B/run_decode_1card.sh` |

:::{note}
单机部署时 Prefill 与 Decode 共用同一台机器，需保证两个 shell 模板中的 `nic_name`、`local_ip` 一致；Decode 的 `engine_port`（9010）与 Prefill（9000）不重叠；**Decode 的 `dp_port`（13495）必须与 Prefill 的 `dp_port`（13395）不同**，否则会报 `EADDRINUSE`。若模型或显存有压力，可在对应 shell 中调低 `--max-model-len`、`--max-num-batched-tokens` 或 `--gpu-memory-utilization`。
:::

**32B 模型与显存：** 上述单机 Decode 配置默认 **TP=4、dp_size=1**（1 个副本用满 4 卡），模型分片到 4 张卡后每卡约 16GB 权重，显存更充裕，适合 Qwen3-32B 等 32B 级模型。若使用 7B 等小模型且希望 4 个单卡 Decode 副本，需自行改启动脚本与 `run_decode_4cards.sh`（多进程、TP=1）。

**Decode 仍 OOM 时：** 若 TP=4 仍报 “NPU out of memory”，可在 `run_decode_4cards.sh` 中降低 `--max-model-len`（如 4096）和 `--gpu-memory-utilization`（如 0.85），或评估 `--quantization ascend`（需对应量化权重并确认兼容）。

## Example proxy for Distributed DP Server

In the PD separation scenario, we need a proxy to distribute requests. Execute the following commands to enable the example proxy:

```shell
python load_balance_proxy_server_example.py \
  --port 8000 \
  --host 0.0.0.0 \
  --prefiller-hosts \
    192.0.0.1 \
    192.0.0.2 \
    192.0.0.3 \
    192.0.0.4 \
  --prefiller-hosts-num \
    2 2 2 2 \
  --prefiller-ports \
    9000 9000 9000 9000 \
  --prefiller-ports-inc \
    2 2 2 2\
  --decoder-hosts \
    192.0.0.5 \
    192.0.0.6 \
    192.0.0.7 \
    192.0.0.8 \
  --decoder-hosts-num \
    16 16 16 16 \
  --decoder-ports  \
    9000 9000 9000 9000 \
  --decoder-ports-inc \
    16 16 16 16 \
```

|Parameter  | meaning |
| --- | --- |
| --port | Proxy service Port |
| --host | Proxy service Host IP|
| --prefiller-hosts | Hosts of prefiller nodes |
| --prefiller-hosts-num | Number of repetitions for prefiller node hosts |
| --prefiller-ports | Ports of prefiller nodes |
| --prefiller-ports-inc | Number of increments for prefiller node ports |
| --decoder-hosts | Hosts of decoder nodes |
| --decoder-hosts-num | Number of repetitions for decoder node hosts |
| --decoder-ports | Ports of decoder nodes |
| --decoder-ports-inc | Number of increments for decoder node ports |

You can get the proxy program in the repository's examples, [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/v0.9.1-dev/examples/disaggregate_prefill_v1/load_balance_proxy_server_example.py)

## Benchmark

We recommend use aisbench tool to assess performance. [aisbench](https://gitee.com/aisbench/benchmark) Execute the following commands to install aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
```

You need to canncel the http proxy before assessing performance, as following

```shell
# unset proxy
unset http_proxy
unset https_proxy
```

- You can place your datasets in the dir: `benchmark/ais_bench/datasets`
- You can change the configurationin the dir :`benchmark/ais_bench/benchmark/configs/models/vllm_api` Take the ``vllm_api_stream_chat.py`` for examples

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="vllm-ascend/DeepSeek-R1-W8A8",
        model="dsr1",
        request_rate = 28,
        retry = 2,
        host_ip = "192.0.0.1", # Proxy service host IP
        host_port = 8000,  # Proxy service Port
        max_out_len = 10,
        batch_size=1536,
        trust_remote_code=True,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1024,
            ignore_eos=False,
        )
    )
]
```

- Take gsm8k dataset for example, execute the following commands  to assess performance.

```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf  --debug  --mode perf
```

- For more details for commands and parameters for aisbench, refer to  [aisbench](https://gitee.com/aisbench/benchmark)

## Prefill & Decode Configuration Details

In the PD separation scenario, we provide a optimized configuration.

- **prefiller node**

1. set HCCL_BUFFSIZE=256
2. add '--enforce-eager' command to 'vllm serve'
3. Take '--kv-transfer-config' as follow

```shell
--kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
      "kv_buffer_device": "npu",
      "kv_role": "kv_producer",
      "kv_parallel_size": "1",
      "kv_port": "20001",
      "engine_id": "0",
      "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
```

4. Take '--additional-config' as follow

```shell
--additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}'
```

- **decoder node**

1. set HCCL_BUFFSIZE=1024
2. Take '--kv-transfer-config' as follow

```shell
--kv-transfer-config
    '{"kv_connector": "MooncakeConnectorV1",
      "kv_buffer_device": "npu",
      "kv_role": "kv_consumer",
      "kv_parallel_size": "1",
      "kv_port": "20001",
      "engine_id": "0",
      "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
    }'
```

3. Take '--additional-config' as follow

```shell
--additional-config '{"enable_weight_nz_layout":true}'
```

### Parameters Description

1.'--additional-config'  Parameter Introduction:

- **"enable_weight_nz_layout"：** Whether to convert quantized weights to NZ format to accelerate matrix multiplication.
- **"enable_prefill_optimizations"：** Whether to enable DeepSeek models' prefill optimizations.
  <br>

3.enable MTP
Add the following command to your configurations.

```shell
--speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}'
```

### Recommended Configuration Example

For example，if the average input length is 3.5k, and the output length is 1.1k, the context length is 16k, the max length of the input dataset is 7K. In this scenario, we give a recommended configuration for distributed DP server with high EP. Here we use 4 nodes for prefill and 4 nodes for decode.

| node     | DP | TP | EP | max-model-len | max-num-batched-tokens | max-num-seqs |  gpu-memory-utilization |
|----------|----|----|----|---------------|------------------------|--------------|-----------|
| prefill  | 2  |  8 | 16 |     17000     |         16384          |      4       |    0.9    |
| decode   | 64 |  1 | 64 |     17000     |          256           |      28      |    0.9    |

:::{note}
Note that these configurations are not related to optimization. You need to adjust these parameters based on actual scenarios.
:::

## FAQ

### 1. Prefiller nodes need to warmup

Since the computation of some NPU operators requires several rounds of warm-up to achieve best performance, we recommend preheating the service with some requests before conducting performance tests to achieve the best end-to-end throughput.
