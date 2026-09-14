# 记录一些实验

## 实验约定

要求Checkpoint命名如下：

```text
{base_policy}_{task}_{schedule}_{tactile}

base_policy —— pi05 GR00T ACT DP 等

task —— device + task，如 franka_xhand_flower UniVTAC_insert_HDMI

schedule —— streaming full 等

    在 schedule 下可以命名不同的调度方式，如 chunk_10 noise training_delay 等，但我们更建议通过 config 快照进行实验区分，而实验命名采用简单的方式声明即可

tactile —— 是否使用触觉，如 tactile_AdaLN 使用触觉且为 AdaLN 条件注入

```

## schedule 子命名



## tactile 子命名


## univtac 实验结果及分析

100 episode 64 batch 2 卡 4000 step

pi05_UniVTAC_insert_HDMI_streaming: 8/100

pi05_UniVTAC_insert_HDMI_streaming_mask: 

pi05_UniVTAC_insert_HDMI: 12/100

pi05_UniVTAC_insert_HDMI_streaming_tactile: 