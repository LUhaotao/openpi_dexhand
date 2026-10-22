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

在推理机器上分别启动两个 role。VLM 编码观测时会向 FM 查询当前 chunk 编号，因此 VLM 也要配置 FM 地址；同机部署时使用 `127.0.0.1:8000`。FM 侧仍通过 `--vlm-host/--vlm-port` 拉取 VLM 生成的 KVCache：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_policy.py \
  --multi-process-role vlm \
  --port 8001 \
  --fm-host 127.0.0.1 \
  --fm-port 8000 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /path/to/streaming_checkpoint
```

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_policy.py \
  --multi-process-role fm \
  --port 8000 \
  --save-gate-logs \
  --vlm-host 127.0.0.1 \
  --vlm-port 8001 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /path/to/streaming_checkpoint

# --save-gate-logs \ 用来保存门控值的，保存路径是 log/exp_name exp_name通过policy.dir解析
```

在 UniVTAC 机器中，将 `policy/streaming_openpi/deploy.yml` 的 `host` 改为推理机器 IP，然后运行：

```bash
bash eval_policy.sh lift_bottle demo streaming_openpi/deploy 0
```

checkpoint 必须有与 UniVTAC Panda 8-D qpos 一致的 action space；否则要在 `deploy.yml` 显式声明已验证的 `action_indices`。

### Runtime 管理的 VLM/FM（两张 GPU）

这是新增的统一 Runtime 部署入口，不影响上面的单进程和 `--multi-process` server。默认 `--mode process` 时，Runtime 只暴露一个端口并拉起两个私有 worker；`--vlm-cuda-device` 和 `--fm-cuda-device` 使用主机上的 CUDA 编号。KV Cache 使用本机 shared memory 传递，worker WebSocket 只传控制事件和 descriptor：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/serve_runtime.py \
  --port 8000 \
  --vlm-cuda-device 0 \
  --fm-cuda-device 1 \
  --streaming-horizon 50 \
  --streaming-watermark 40 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /path/to/streaming_checkpoint
```

UniVTAC 使用 `policy/openpi/deploy_runtime.yml`，只连接 Runtime 的 `8000` 端口。每个控制周期同步发送一帧 observation 并等待一个 action；推理消息里只有 observation/action。Runtime 在服务端内部维护 execution id、FM chunk action queue 和 VLM refresh waterline，client 无需发送 cache/session/execution 调度字段。`reset_stream` 仅在 episode 开始时作为生命周期控制消息发送。Runtime 的两个 worker 使用随机分配的本机私有端口，原有 VLM/FM 公网端口协议仍然保留。

单进程也可使用相同 Runtime API，将 backend 切到 `local`。此模式只加载一个本地 JAX policy，KV Cache 直接作为 JAX pytree 传递，不使用 worker WebSocket 或 shared memory：

```bash
.venv/bin/python scripts/serve_runtime.py \
  --mode local \
  --cuda-device 0 \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_streaming \
  --policy.dir /path/to/streaming_checkpoint
```

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


### 虚拟实验

```bash 
uv run python -m exp_scripts.jax_obs_transition_probe \
  --checkpoint /public/node01/users/lvrui/workspace/openpi_dexhand/checkpoints/pi05_franka_xhand_flower_streaming/pi05_franka_xhand_flower_streaming_training_delay/19999 \
  --dataset /public/node01/users/lvrui/datasets/lerobot/flower_xhand_franka \
  --config-name pi05_franka_xhand_flower_streaming \
  --experiment all \
  --seed 42 \
  --max-pairs 32 \
  --frame-gap 20 \
  --num-noise-samples 1 \
  --save-traces \
  --output-dir results/pi05_franka_xhand_flower_streaming_training_delay_anchor


uv run python -m exp_scripts.jax_obs_transition_probe   --checkpoint /public/node01/users/lvrui/workspace/openpi_dexhand/checkpoints/pi05_franka_xhand_flower_zhb_right_600   --dataset /public/node01/users/lvrui/datasets/lerobot/flower_xhand_franka   --config-name pi05_franka_xhand_flower_v2   --experiment all   --seed 42   --max-pairs 32   --frame-gap 20   --num-noise-samples 1   --save-traces   --output-dir results/pi05_franka_xhand_flower_scalar_anchor

```

### 工具脚本

生成只包含 `state` 和 `actions` 的 `norm_stats.json`：

```bash
cd /public/node01/users/lvrui/workspace/openpi_dexhand

# config_name 使用 src/openpi/training/config.py 中已有的 JAX 配置名
.venv/bin/python scripts/compute_norm_stats.py \
  pi05_UniVTAC_pull_out_key
```

脚本会根据配置中的数据集、`action_horizon`、repack mapping 和数据 transform
遍历数据集，并把结果写入配置对应的 assets 目录。实际写入位置会在命令输出的
`Writing stats to:` 中显示。需要限制扫描数据量时可以增加：

```bash
.venv/bin/python scripts/compute_norm_stats.py \
  pi05_UniVTAC_pull_out_key \
  --max-frames 100000
```

将旧的增量 `model_config` 展开为新的完整 JSON 配置：

```bash
cd /public/node01/users/lvrui/workspace/openpi_dexhand

.venv/bin/python scripts/expand_config.py \
  /path/to/checkpoint/model_config/pi0_config.json \
  --defaults configs/default.json \
  --output configs/experiments/pi05/my_experiment.json
```

输入也可以是旧的增量实验 JSON。脚本会使用 `configs/default.json` 补齐缺少的
模型、数据、`repack_mapping`、优化器和训练字段，并输出完整 JSON。旧的
`pi0_config.json` 只包含模型字段时，转换后需要根据实际实验补充 `name`、
`exp_name`、数据集路径和 `weight_loader.params_path`。
