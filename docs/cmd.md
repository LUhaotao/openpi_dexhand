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
