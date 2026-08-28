继续深入逐行讲解

1. 模型载入怎么做 这里是初始化权重路径，不是初始化权重
checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
    config.checkpoint_dir,
    keep_period=config.keep_period,
    overwrite=config.overwrite,
    resume=config.resume,
)

初始化权重在
train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
续训的时候先初始化模型然后用这个加载权重
train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

2. dataloader怎么做
data_loader = _data_loader.create_data_loader(
    config,
    sharding=data_sharding,

dataloader数据transform这里封装比较深，具体做了哪些处理？

3. 执行一次训练具体怎么做
train_state, info = ptrain_step(train_rng, train_state, batch)

小问题：

1. ema用来干啥

2. batch、worker配置参考，batch我看到是总的，num_worker是总的还是单个GPU的

3. 验证能不能跑是不是应该跑完一整轮（step=1）

4. TrainConfig里面很多配置项可以设置，但是我们没有设置是吧，比如续训得那个checkpoint_dir啥的

pi-r2 的阅读问题：

这个关键是他的噪声调度，也就是diffusion-forcing给每个token不同的noise_timestep

1. 为啥线性ramp - 这个方法会不会惯性更大，考虑非线性ramp

2. 为啥训练推理用一样的噪声调度 - 为啥训练的时候不随机调度噪声？这样鲁棒性直观上更好一点

复现的问题是惯性更大 —— 开始的时候手指会到抓握状态，然后腕部就无法正常运行了，这是因为状态过拟合严重（抓握后无法调整）、惯性太大（RTC类方法），KV Cache的刷新频率低过拟合很正常，惯性大是RTC类方法的常态