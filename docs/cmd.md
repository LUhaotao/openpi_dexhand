# 记录指令

### 真机推理

我们的推理有很多，单进程同步，多进程同步，多进程异步streaming

server：

```bash
  python scripts/serve_policy.py \
    --multi-process \
    --port 8000 \
    --vlm-port 8001 \
    policy:checkpoint \
    --policy.config pi05_franka_xhand_flower_streaming \
    --policy.dir=/data/dex_ws/pi05_checkpoints/pi05_franka_xhand_flower_zhb_right_600
```

client：

```bash
from scripts.multi_process_client import MultiProcessClient

client = MultiProcessClient(
    host="127.0.0.1",
    fm_port=8000,
    vlm_port=8001,
)

client.update_vlm(observation)
result = client.infer_fm(observation)

```


### fork env 测试延迟

思路上是给模型全部0输入

server：
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_policy.py \
  --multi-process \
  --port 8000 \
  --vlm-port 8001 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /home/rui/data/openpi_dexhand/ckpt/pi05_franka_xhand_flower_zhb_right_600/flower_zhb_right_600/29999
```


client：
```bash
.venv/bin/python scripts/test_multi_process_latency.py \
  --mode fm \
  --environment franka_xhand_continuous_state \
  --num-steps 10 \
  --noise-tokens 1 \
  --warmup 3 \
  --runs 10

# 这里warmup是指先运行3次推理
# num-steps表示前向10次，vlm没有实装num-steps
# runs表示总共运行多少次计算均值
# noise-tokens表示去噪的token数
```

### fake client 测试延迟

`fake_client.py` 使用固定 shape 的全零 observation，不需要在客户端安装 JAX 或启动仿真环境。
VLM latency 包括请求上传、prefix 推理、KV cache materialize 以及在 VLM 侧保留为可拉取 cache 的时间；
FM latency 只测已有 active KV cache 上的一次运行态 forward；ALL latency 测量
`VLM encode_prefix → FM 拉取并激活 KV cache → FM forward` 的完整运行态链路。

先在 `scripts/test_fake_client_latency.sh` 顶部设置 VLM 和 FM 两个远端 endpoint：

```bash
VLM_HOST=10.0.0.11
VLM_PORT=8001
FM_HOST=10.0.0.11
FM_PORT=8000
```

启动对应 server 后运行：

```bash
bash scripts/test_fake_client_latency.sh
```

默认会依次测试 VLM、FM、ALL。可以通过修改 bash 脚本中的环境变量调整 warmup、runs、`num_steps`、
`noise_tokens`、streaming chunk size 和 observation profile。FM 默认使用已初始化的 `stream_infer`；
对于非 streaming checkpoint，可以将 `FAKE_FM_MODE=infer`。将 `FAKE_NOISE_TOKENS` 设为空时，
infer 模式使用完整 action horizon。

FM 的初始 cache setup 和 streaming seed 在正式计时前完成；计时过程中不清理 cache。ALL 的每个样本
都会同步等待 FM 激活本次 VLM 生成的新 cache，确保 cache 拉取延迟被计入。

### UniVTAC streaming（两张 GPU）

在推理机器上分别启动两个 role。VLM 与 FM 在同一台机器时，FM 的 `--vlm-host` 保持 `127.0.0.1`：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_policy.py \
  --multi-process-role vlm \
  --port 8001 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /path/to/streaming_checkpoint
```

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_policy.py \
  --multi-process-role fm \
  --port 8000 \
  --vlm-host 127.0.0.1 \
  --vlm-port 8001 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /path/to/streaming_checkpoint
```

在 UniVTAC 机器中，将 `policy/streaming_openpi/deploy.yml` 的 `host` 改为推理机器 IP，然后运行：

```bash
bash eval_policy.sh lift_bottle demo streaming_openpi/deploy 0
```

checkpoint 必须有与 UniVTAC Panda 8-D qpos 一致的 action space；否则要在 `deploy.yml` 显式声明已验证的 `action_indices`。

单进程推理（单进程推理要和单进程训练权重配合使用）：

```bash
  cd /path/to/openpi_dexhand

  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_franka_xhand_flower_streaming \
    --policy.dir /path/to/streaming_checkpoint
```

UniVTAC client命令：

```bash


```

### 训练
```bash
# 有一个脚本可以用
# train_pi05_dex.sh


```
