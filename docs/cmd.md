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