# FX 图与 ACL Graph：概念与全链路说明

> 分支：`v0.18.0-branch`
> 关联代码：`vllm_ascend/platform.py`、`vllm_ascend/compilation/acl_graph.py`、`vllm_ascend/attention/attention_v1.py`
> 关联文档：`ACL_Graph.md`（更偏使用）、`attention_turboquant_fusion_design.md`（turboquant 与图模式兼容性）

本文档把 **FX 图（编译期程序 IR）** 与 **ACL Graph（运行期设备执行图）** 两个概念讲清楚，并给出"源码 → Dynamo → fx 图 → 按 splitting_ops 切分 → 每片 ACLGraph 捕获 → 运行期重放"的完整链路。

---

## 1. 两个概念,别混淆

| | **FX 图** | **ACL/CUDA Graph** |
|---|---|---|
| 层次 | **编译期** 的程序 IR（Python 侧）| **运行期** 的设备执行图 |
| 内容 | 算子节点 + 数据依赖（可读、可改写）| 录好的 kernel 下发 + 固定地址（只能重放）|
| 作用 | 分析 / 变换 / 切分 / 接后端编译 | 消除 host 逐算子下发开销 |
| 关系 | 先有 fx 图 → 切分 → 每片在执行时再被 ACL Graph 捕获 | 是 fx piece 的运行期载体 |

一句话:**fx 图是"程序结构的可改写表示",ACL Graph 是"设备执行的录制重放";piecewise 先用 fx 图把模型按注意力切片,再把每片用 ACL Graph 捕获加速。**

---

## 2. FX 图(torch.fx)

### 2.1 定义

FX 图是把一个 `nn.Module` 的 `forward` 捕获成的、由算子节点组成的有向无环图(DAG)。它用数据结构显式表达"forward 依次做了哪些操作、数据如何流动",从而可被遍历、改写、切分、交给自定义后端编译。

### 2.2 三层结构

```
GraphModule  ──持有──►  Graph  ──持有──►  [Node, Node, ...]
 (可执行模块)            (DAG)              (单个算子节点)
```

- `GraphModule`：特殊的 `nn.Module`，持有一张 `Graph`，并能根据图自动生成等价 `forward` 源码（`recompile()`），可直接执行。
- `Graph`：节点的拓扑有序集合，表达数据依赖。
- `Node`：一个操作，四要素 `op` / `target` / `args` / `kwargs`；`args` 引用其它 Node 即构成边。

Node 的 6 种 `op`：

| op | 含义 | 例 |
|---|---|---|
| `placeholder` | forward 输入 | `x` |
| `get_attr` | 取参数/buffer | `self.weight` |
| `call_function` | 自由函数 | `torch.relu` |
| `call_method` | 张量方法 | `x.view(...)` |
| `call_module` | 子模块 | `self.linear(x)` |
| `output` | 返回值 | `return ...` |

### 2.3 例子

```python
class M(nn.Module):
    def __init__(self): super().__init__(); self.lin = nn.Linear(4, 4)
    def forward(self, x):
        a = self.lin(x); b = torch.relu(a); return b + 1
```

`symbolic_trace(M())` 的图（`print_tabular()`）：

```
opcode         name    target          args          kwargs
placeholder    x       x               ()            {}
call_module    lin     lin             (x,)          {}
call_function  relu    torch.relu      (lin,)        {}
call_function  add     operator.add    (relu, 1)     {}
output         output  output          (add,)        {}
```

GraphModule 据此生成的 `forward`：

```python
def forward(self, x):
    lin = self.lin(x); relu = torch.relu(lin); add = relu + 1; return add
```

### 2.4 怎么捕获出来

- **`symbolic_trace`**：用 `Proxy` 假张量喂 forward，算子被拦截记成 Node；遇数据相关控制流会失败/特化。
- **TorchDynamo（`torch.compile` 走这条）**：字节码层追踪，能处理任意 Python，遇不可入图处插 **graph break** 切成多张 fx 子图。vLLM 用此路（`CompilationMode.VLLM_COMPILE`）。

> 动态 shape 可用 SymInt 表达；但 **数据相关 shape（如 `unique()`/`bt[valid]`，unbacked SymInt）** 会触发 graph break——这正是 turboquant `decode_compact` 难入图的根因。

---

## 3. ACL Graph(昇腾上的 cudagraph)

### 3.1 解决什么问题

LLM 推理每 token 近千个算子,**device 跑得快但 host 逐个 launch 慢**,造成 host bound、device 空闲。ACL Graph 把"一次 forward 的全部算子下发"录成图,**单次 host 调用重放**,消除逐算子下发开销(`ACL_Graph.md`)。

### 3.2 捕获与重放 + 地址固定

图录的是"算子 + 它当时用的**绝对显存地址** + 顺序",不是数据。重放时 device 读写同一批固定地址。因此**所有地址必须固定**,分两类:

| 数据 | 如何固定 |
|---|---|
| 模型输入 / KV cache / output | runner 预分配**持久 buffer**,每步 `copy_` 新内容进去（`model_runner_v1.py:801`）；KV cache 启动时一次分配 |
| 算子间中间张量(activations) | 捕获时从**全局图内存池** `get_global_graph_pool()` 分配,地址被烘焙进图（`acl_graph.py:84/156`）|

> 中间张量不每步拷贝——它们由图内存池在捕获时定址,重放时 kernel 写/读同一池内地址。

### 3.3 静态化:Padding + Bucketing

一张图只对应一个 shape。请求 shape 多变 → 捕获**多张不同 shape 的图(bucket)**,输入 pad 到最近的图;超阈值大 shape 直接 eager。代码里 `concrete_aclgraph_entries: dict[BatchDescriptor, ...]` 每个 shape 桶一套图（`acl_graph.py:91/123`）。

### 3.4 注意力的运行时 workspace

注意力算子需运行时 workspace,不能简单录死。`GraphParams`（`acl_graph.py:245`）按 token 数缓存 `workspaces/handles/events/attn_params`;捕获时 `graph_task_group_begin/end` 录成**可更新 task group**,重放前 `update_graph_params`（`attention_v1.py:424`）用 `graph_task_update_*` 回填 workspace/seq_lens/block_table。

---

## 4. 默认 graph 模式

判定（`platform.py`）：

```
enforce_eager                         → NONE          (:312)
xlite full / decode_only              → NONE / FULL_DECODE_ONLY  (:307/:310)
vLLM 默认 FULL_AND_PIECEWISE          → 强制降级 PIECEWISE  (:354-355)
```

`platform.py:353` TODO：full graph 完整支持后默认值才会改成 full。**当前未 enforce_eager、未显式配置时,生效默认 = PIECEWISE。**

切分配置：

| 模式 | 配置（platform.py） | 注意力 |
|---|---|---|
| `PIECEWISE`（:369） | `set_splitting_ops_for_v1(...)` + `splitting_ops += ["vllm::mla_forward"]`，`use_inductor=False` | **eager** |
| `FULL` / `FULL_DECODE_ONLY`（:389） | `splitting_ops = []`（不切） | **进图** |
| `FULL_AND_PIECEWISE` | 降级为 PIECEWISE | eager |
| `NONE` | 关图 | eager |

---

## 5. 全链路流程图

```
┌── ① 源码: 模型 forward (Python) ──────────────────────────────────────────┐
│   for layer in layers:                                                    │
│       h = h + attn(norm1(h))     # attn 是注册 custom op                   │
│       h = h + mlp(norm2(h))                                               │
└───────────────────────────────────────────────────────────────────────────┘
        │  @support_torch_compile (CompilationMode.VLLM_COMPILE)
        ▼
┌── ② TorchDynamo: 字节码追踪 → 捕获 fx 图 ─────────────────────────────────┐
│   算子记成 Node;不可入图处 graph break;数据相关 shape → break             │
└───────────────────────────────────────────────────────────────────────────┘
        ▼
┌── ③ fx 图 (GraphModule: Node DAG, 可改写) ────────────────────────────────┐
│   x → embed → norm1 →[attn]→ add → norm2 → mlp → add →…→ norm → lm_head    │
│                        ▲ splitting_ops 命中节点                            │
│   (platform.py:375 set_splitting_ops_for_v1 + :386 "vllm::mla_forward")    │
└───────────────────────────────────────────────────────────────────────────┘
        │  partition: 在每个 attn 节点处切开
        ▼
┌── ④ 按 splitting_ops 切分 → piece + 夹在中间的 eager 注意力 ──────────────┐
│   ┌piece0┐ [attn0] ┌piece1┐ [attn1] ┌piece2┐ … [attnK] ┌pieceN┐           │
│   │embed │  eager  │mlp   │  eager  │mlp   │    eager   │norm  │           │
│   │norm1 │         │norm  │         │norm  │            │lmhead│           │
│   └──────┘         └──────┘         └──────┘            └──────┘           │
│   每 piece = 独立 fx 子图; attn 节点不进 piece, 留作 eager                 │
└───────────────────────────────────────────────────────────────────────────┘
        │  每 piece 编译 + 外包 ACLGraphWrapper(runtime_mode=PIECEWISE)
        ▼
┌── ⑤ 每片 ACLGraph 捕获 (ACLGraphWrapper.__call__, acl_graph.py:109) ──────┐
│   按 batch_descriptor(shape 桶)分别捕获:                                   │
│   首次: input_addresses=[data_ptr]        (:139)                           │
│         with torch.npu.graph(g, pool=graph_pool):  (:156)                  │
│             output = piece(*args)         (:158 录制 kernel 序列)          │
│   注意力: runtime_mode 不匹配 → eager self.runnable(...)  (:114)           │
└───────────────────────────────────────────────────────────────────────────┘
        ▼
┌── ⑥ 运行期(每 decode step) ──────────────────────────────────────────────┐
│   runner: 本步输入 copy_ 进持久 buffer(地址不变,内容更新)                  │
│   piece0.replay() →[attn0 eager: decode/attn]→ piece1.replay() →…→ logits  │
│   (acl_graph.py:209 replay();  :190 校验输入地址未变)                      │
│   超捕获 size 的大 batch → runtime_mode=NONE → piece 也 eager (:114)        │
└───────────────────────────────────────────────────────────────────────────┘
```

各转换点：

| 阶段 | 输入 → 输出 | 关键代码 |
|---|---|---|
| ①→② | forward → Dynamo 追踪 | `@support_torch_compile` |
| ②→③ | 字节码 → fx GraphModule | TorchDynamo |
| ③→④ | fx 图 → 按注意力节点 partition | `platform.py:375/386` |
| ④→⑤ | 每 piece → 捕获 NPUGraph | `acl_graph.py:141/156` |
| ⑤→⑥ | 持久 buffer 更新 → 逐 piece replay + 注意力 eager | `acl_graph.py:209/114` |

两层地址固定：piece 间张量 + 模型输入靠**持久 buffer + copy_**；piece 内中间张量靠**图内存池烘焙**。

---

## 6. Piecewise vs Full graph

- **Full graph**：`splitting_ops=[]`，注意力也进图，最优；但注意力的动态 shape / 分页寻址 / workspace 难捕获，且容易 OOM（`platform.py:398` 警告）。
- **Piecewise**：注意力 eager，其余 piece 进图；host 仍每步下发注意力（开销小于全 eager）。full graph 跑不了 / 混合 prefill-decode（MLA）时的稳妥选择,也是当前默认。

---

## 7. turboquant 与图模式的关系(摘要)

- **默认 PIECEWISE → 注意力 eager → turboquant `decode_compact`(动态 shape)在 eager 下运行 → 兼容且正确。**
- **FULL / FULL_DECODE_ONLY**:注意力并入 piece 被捕获 → 触发两个问题:① compact 动态 shape 不可捕获;② 捕获路径 `full_graph_fia/pa`(`attention_v1.py:570/663`)**不解码 packed cache** → 输出错误。
- 详细的兼容性结论与开发项见 `attention_turboquant_fusion_design.md` 的 **Phase 0 / turboquant 与图模式兼容性**。
