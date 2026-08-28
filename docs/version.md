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

- [] 推理延迟测试接口

## ## version 0.1.2

实现**推理时streaming噪声调度**：

- [] 允许streaming与滑动窗口

## 阶段总结

self-forcing专门提到diffusion forcing预先去噪未来状态带来的问题 —— 新增控制对未来输出**控制能力不够**，self-forcing将这种情况描述为 **premature commitment（早熟）**，认为其**降低了生成的交互性**，在我们实际使用中同样发现diffusion forcing类方法推理的**一致性太强**，导致**惯性特别大**

我们的目标是希望快速的本体状态、力矩、触觉等信息能够对输出产生即时控制，但是当前的方法即时控制效果弱，思路：

1. 分割语义空间，在较远区域增加图像权重，较近区域增加快速反馈权重（Gate，事件）

2. 切换self-forcing类方法（延迟问题，需要看看self-forcing原文）

3. 直接对图像KV Cache刷新（同样是fm的延迟问题）

RTC-pir2 体现了 teacher-forcing 到 diffusion-forcing 的过程

另外我们这里要先接入触觉以创造主题

## version 0.1.2

增加触觉进入fm部分：

- [] 触觉encoder

- [] encoder结果和本体状态同样方式接入