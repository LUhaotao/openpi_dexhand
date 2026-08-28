# 测试同步需要

## KV Cache 问题

VLM 生成的 KV Cache 是一个由 `jax.Array` 组成的 pytree。`jax.Array` 背后的 device buffer
属于创建它的 JAX 进程、CUDA context 和具体 device，因此不能把 Python 对象或 GPU 指针直接交给
另一个 FM 进程。即使 VLM 和 FM 使用同一张 GPU，两个进程也拥有独立的 JAX runtime 和 CUDA
context；部署到两台物理机时则必然需要经过网络传输。

当前实现使用 `jax.device_get()` 将 KV Cache 从 VLM device 搬到 host，并递归转换为 NumPy 数组，
再使用 msgpack 通过 WebSocket 传输。该路径能够解除 JAX array 对原进程和 device 的绑定。但是当前实现还有两个问题：

1. VLM 将完整 KV Cache 返回给 client，client 保存后再将其发送给 FM，数据经过了不必要的 client
   中转。client 应当只承担控制面调度，不持有或转发 KV Cache payload。
2. 现有 msgpack encoder 不支持 NumPy `bfloat16`，所以代码将 BF16 KV Cache 转换成 FP32 后发送；
   FM 端只调用了 `jnp.asarray()`，没有显式恢复为模型使用的 BF16。这会使传输量和 host/device
   内存占用翻倍，并可能导致 FM 计算中的 dtype promotion。

目标数据流为：

```text
Client -> VLM: encode_prefix(images, prompt)
VLM -> Client: cache_id + cache_version
Client -> FM: refresh_prefix(vlm_endpoint, cache_id, cache_version)
FM -> VLM: 直接读取 KV Cache payload
FM: 校验并原子替换 active KV Cache
FM -> Client: refresh ack
Client -> FM: infer(latest_state, expected_cache_version)
```

VLM 需要将 JAX device array materialize
为保留 BF16 的 host bytes，并携带 pytree 结构、leaf path、shape、dtype、model id、cache version
和 prefix mask。FM 收到后按照自己的 device topology 使用 `jax.device_put()` 或显式指定
`jnp.bfloat16` 的 `jnp.asarray()` 重建 cache。VLM 的 device id、CUDA pointer 和 JAX sharding
对象不应成为传输协议的一部分。

PyTorch/DLPack 可以用于同进程、同 GPU 的零拷贝转换，但不能直接跨 WebSocket 或跨物理机。
同机器跨进程还需要额外的 CUDA IPC 和 buffer 生命周期管理，因此不作为当前同步协议的基础设计。

### NIXL 能力取舍

NIXL 面向多请求、异步、多 GPU、多节点和 RDMA 等复杂推理基础设施，因此包含内存注册、远端
descriptor、lease/heartbeat、完成通知、传输后端选择和故障恢复等保护。当前系统可以保证 VLM 和
FM 各自只使用单个 GPU，并且 cache 刷新完全同步，但不能保证两端 GPU 型号、网络设备或 RDMA/UCX
环境一致。第一版不应将 NIXL 作为运行前置条件，而应只保留与传输方式无关的正确性约束。

当前必须实现的保护包括：

1. VLM 和 FM 启动或刷新时校验 protocol version、model id、cache schema、dtype、leaf shape 和
   payload size，避免两端以不同结构解释同一份 cache。
2. 使用 `cache_id + cache_version` 关联一次刷新；重复请求同一版本时内容必须保持不变。
3. FM 使用 active/staging 双缓冲。新 cache 完整接收、校验并放入本地 device 后才能原子替换
   active cache，不能让推理读取正在写入或只传输了一部分的数据。
4. FM 的 refresh ACK 必须表示新 cache 已经可以用于推理，而不只是网络接收已经结束。传输失败或
   超时后不得激活 staging cache。
5. 限制最大 payload、leaf 数量、允许的 dtype 和 shape，并记录传输字节数、耗时和失败原因。

当前不需要引入异构 tensor parallel、mesh/sharding 转换、多 NIC 聚合、自动 RDMA 路径选择、ETCD
服务发现、paged KV block、分层存储、LRU 淘汰、跨请求复用、异步 layer-wise 传输和多副本故障转移。
GPU id、CUDA pointer、JAX sharding 和 VLM 本地 device 信息也不应进入传输协议；每个 server 只负责
将 cache 放到自己的本地默认 GPU。

传输层应保留一个小型 `KVTransport` 抽象。第一版实现框架无关的 `HostTcpTransport`，使用
`JAX BF16 -> host bytes -> TCP/WebSocket -> JAX BF16` 完成 VLM 到 FM 的直接传输。后续只有在
实测确认 host copy 或 TCP 是主要瓶颈，并且部署环境能够提供 NIXL/UCX/RDMA 时，再增加
`NixlTransport`；调度、cache version、双缓冲和 ACK 语义保持不变。


## state 语义问题

高低频拆分后的语义边界应当固定为：

```text
VLM prefix: images + language
FM suffix: latest state + noisy/streaming actions + timestep
```

state 必须属于 FM，因为后续 streaming inference 需要让高频本体状态及时影响动作去噪过程，而
VLM KV Cache 可以按较低频率刷新。每次 FM 推理都应接收最新 state，并使用当前 active cache。

当前 pi0.5 的 state 语义不满足该边界。`TokenizePrompt(discrete_state_input=True)` 会先将归一化后的
state 离散化并编码进 language prefix，因此 state 随 KV Cache 一起被冻结。使用旧 cache 调用 FM
时，即使请求携带了新 observation，新的 state 也不会进入 FM 的去噪网络。它最多参与输出端从
delta action 恢复 absolute action，不能为动作生成提供新的状态条件。

不能只把 `discrete_state_input` 改为 `False`。当前 pi0.5 的 `embed_suffix()` 不读取 state，也没有
pi0 路径中的 `state_proj`；单独关闭 state tokenization 会使模型完全忽略 state。正确改动需要同时：

1. 从 VLM prompt tokenization 中移除 state，使 prefix 只由图像和任务语言决定。
2. 在 pi0.5 FM suffix 中增加连续 state projection/state token，并设置正确的 attention mask 和
   position。
3. 在 FM 侧完成与训练一致的 state normalization，并确保 streaming 训练和推理使用相同结构。
4. 使用同一次 FM 请求的 state 完成 output transform，将 Franka 前 6 维恢复成 absolute target；
   XHand 12 维继续保持 absolute target。
5. 在 FM 请求中携带 `expected_cache_version`，避免 cache 刷新失败或重连后使用错误版本生成动作。
