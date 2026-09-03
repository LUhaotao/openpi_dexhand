# 此处记录 3rd 内可以使用的一些信息

## 算法层面

### 拆分触觉语义和动作信息

1. https://arxiv.org/pdf/2608.19574，HiTac-WAM:，这一篇论文拆分了接触、形变、滑移风险，接触和滑移都是事件，形变是作用力相关，**可以参考这篇论文的拆分思路**

### 触觉信息注入方式

1. 

### benchmark

1. https://github.com/univtac/UniVTAC，UniVTAC

### baseline

开源的触觉模型不多：

1. RDP

2. FTP-1

## 基础设施

### 流式推理

1. https://github.com/hku-sail/StreamPI，streamPI，这一篇是做记忆的，但是做了**滑动KVCache，basecode还是openpi**，后续要上滑动KVCache的时候参考

2. https://github.com/pi-r2-flow/Isaac-GR00T.git，Pi-R2，实现了**diffusion forcing的滑动窗口**

3. https://arxiv.org/pdf/2608.25798v1，TacForcing:，这一篇是最为接近我们想法的，diffusion forcing + 触觉，他的状态使用在了离散部分，保持pi05的原始信息，但是没有开源**我们要加速了**，这一篇在UniVTAC上面跑了，我们也跑这个，这篇论文给出了触觉注入方式**动作 token 作为 query，触觉 token 作为 key**，同时只允许当前即将执行token看触觉

4. https://github.com/zhuoqun-chen/rnrdp，RNR-DP，在训练时使用独立随机噪声，其提到**streaming噪声提高平滑性，随机噪声提高独立去噪能力**，使用额外的训练步训练独立去噪会有附加成本

5. https://github.com/z-lab/flashvla，flashVLA，比较重要的是使用的chunk内同样时间步，chunk间不同时间步，我们的问题很有可能是一个**action chunk解决的模式冲突问题**

### 模型高低频

1. https://github.com/pi-r2-flow/pi-r2-flow，Pi-R2，主要拆分了vlm-fm，和我们很适配但是base code不一样

2. https://github.com/xiaoxiaoxh/reactive_diffusion_policy，RDP，比较早了而且是DP不是大模型