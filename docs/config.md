# model config

为了适应实验需要，我们将openpi的Pi0Config保留成了config快照，以保持训推一致性，config默认值如下所示：
```bash
## 模型
"pi05": false # 以下配置pi05启动时才生效
"discrete_state_input": true # 默认使用离散state，可以使用false变成将状态通过mlp进入fm
## streaming
"streaming": false # 以下配置streaming启动时才生效
"action_horizon": 50 
"streaming_chunk_size": 5 
"streaming_constant_weight": 0.2 # 所有horizon使用同样的timestep
"streaming_chunk_wise_weight": 0.8 # chunk使用同样的timestep
"streaming_token_wise_weight": 0.0 # 每一个token使用不同的timestep
"streaming_attention_mode": "bidirectional" # 所有token使用双向注意力（还可以选择mask和casual，chunk之间隔断注意力，chunk之间使用因果注意力）
## tactile
"use_tactile": false # 以下配置tactile启用时才生效
```

目前我们的config主要建立在JAX+pi05线路上，其他线路没有改动

