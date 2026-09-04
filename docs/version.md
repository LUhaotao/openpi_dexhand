# 介绍

该文件用于进行版本及相关实验记录

## version 0.1.0

实现**训练时streaming噪声调度**：

- [*] 修改AdaRMS等Embed及归一化以适配diffusion forcing

- [*] 增加streaming噪声调度逻辑

只实现了streaming，没有实现RTC的干净前缀，也没有给时间步增加随机噪声的trick

## version 0.1.1

实现**高低频划分**：

- [*] 拆分原openpi server逻辑，允许vlm、fm分别多进程运行，设计KV Cache server存储

- [*] 配置对应的client调度方案，fm-vlm两个同步推理（双进程一样会抢占单个GPU，最好的办法是服务器跑VLM，本地跑FM，或者本地做显卡划分，这样很麻烦）

- [*] 推理延迟测试接口，**实际测试单个token和多个token的提速不明显，最好降低前向次数或者配合其他fm加速措施**

### version 0.1.1 fix

- [*] 将状态通过一个mlp转换到fm一侧，无论训练还是推理时，和噪声token一同前向，**这里后期最好进行一个是否滑动窗口的实验**（使用action同样尺寸的encoder，一个简单的线性层，可以考虑是否切换成**mlp**）

目前这里state归一化直接引入了一个额外的干净timestep emb（参考training-time RTC方式），这会导致AdaRMS归一化偏向近端action带来问题，需要考虑别的注入方式-**AdaLN**，或者考虑**使用Mask提供clean action语义**

## version 0.1.2

实现**推理时streaming噪声调度**：

- [*] 允许streaming与滑动窗口

- [] 部分导入与异步推理，需要配合云端推理使用，**部分导入不是必须而是优化选项**，使用UniVTAC配合测试云端推理

## version 0.1.2 fix

- [*] 补充fm server侧异步刷新kvcache

## 阶段总结

self-forcing专门提到diffusion forcing预先去噪未来状态带来的问题 —— 新增控制对未来输出**控制能力不够**，self-forcing将这种情况描述为 **premature commitment（早熟）**，认为其**降低了生成的交互性**，在我们实际使用中同样发现diffusion forcing类方法推理的**一致性太强**，导致**惯性特别大**

我们的目标是希望快速的本体状态、力矩、触觉等信息能够对输出产生即时控制，但是当前的方法即时控制效果弱，思路：

1. 分割语义空间，在较远区域增加图像权重，较近区域增加快速反馈权重（Gate，事件）

2. 切换self-forcing类方法（延迟问题，需要看看self-forcing原文）

3. 直接对图像KV Cache刷新（同样是fm的延迟问题）

RTC-pir2 体现了 teacher-forcing 到 diffusion-forcing 的过程

另外我们这里要先接入触觉以创造主题

## version 0.1.3

- [*] 切换为flashVLA噪声调度，chunk内同样时间步缓解模式冲突，增加chunk后效果好很多，但是chunk同样有问题**1.如何使用即时Conditioned？2.diffusion forcing类方法如何将不同即时条件注入不同状态**

## version 0.1.3 fix

- [*] server warmup

## version 0.2.0

增加触觉进入fm部分：

- [] 初版使用UniVTAC的Marker方案

- [] MLP触觉encoder

- [] state回到prompt里面，触觉通过AdaLN注入

**encoder和注入方式**是主要实验，encoder可以暂时用MLP，注入方式分别尝试**AdaLN、FiLM、CA、Token拼接**

引入仿真作为实验环境：

- [*] 搭建UniVTAC仿真环境并测试推理时通路、延迟