# 记录指令

### 推理

server：

```bash
  python scripts/serve_policy.py \
    --multi-process \
    --port 8000 \
    --vlm-port 8001 \
    policy:checkpoint \
    --policy.config pi05_franka_xhand_flower_v2 \
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
.venv/bin/python scripts/serve_policy.py \
  --multi-process \
  --port 8000 \
  --vlm-port 8001 \
  policy:checkpoint \
  --policy.config pi05_franka_xhand_flower_v2 \
  --policy.dir /home/rui/data/openpi_dexhand/ckpt/pi05_franka_xhand_flower_zhb_right_600/flower_zhb_right_600/29999
```


client：
```bash
.venv/bin/python scripts/test_multi_process_latency.py \
  --mode fm \
  --environment franka_xhand \
  --num-steps 10 \
  --noise-tokens 1 \
  --warmup 3 \
  --runs 10
# 这里warmup是指先运行3次推理
# runs表示前向10次，vlm没有实装runs
# num-steps表示总共运行多少次计算均值
# noise-tokens表示去噪的token数
```