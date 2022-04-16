Week 01【2022年4月16日22点17分】
宅码分享: 
期数: 第6期
论文：Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting
关键词：时序、Transformer、AAAI2021
分享者：艾宏峰
分享时间：4月15日周五晚9点（已结束）

---

# 【时序】Informer：对长序列预测更高效的Transformer

Original Ai [宅码](javascript:void(0);) *2022-04-15 22:53*

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqXpuIgIVaR9FibkHTMAKLOMeMAMYlYSYVOM22V5WTHDY0KkYb5dA96yw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**论文：**AAAI2021 | Informer: Beyond efficient transformer for long sequence time-series forecasting [1]

**作者：**Zhou H, Zhang S, Peng J, et al.

**机构：**北航、UC伯克利、Rutgers大学等

**录播：**https://www.bilibili.com/video/BV1RB4y1m714?spm_id_from=333.999.0.0

**代码：**https://github.com/zhouhaoyi/Informer2020

**引用量**：162

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGq7Y5S0M2jFxdFNW2GIvpVOAvLb7Oj2Z4wqelXnZSujaKQmNdBP4cgpg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

Informer是AAAI2021的最佳论文，主要是针对**长时序预测任务（Long sequence time-series forecasting，LSTF）**，改进了Transformer。



**一、历史瓶颈**

图1展示了电力转换站的小时级温度预测，其中短期预测是0.5天（12个点），而长期预测是20天，480个点，其中当预测长度大于48个点时，整体表现会有明显的差异，LSTM的MSE和推理速度都在往很坏的方向发展。



![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqRwWyJA9HU4ROaT6zNL8QT9gkCmia0J7Z7xAUd8ia5GsaFxQsCHibtvCgw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图1：LSTM在不同预测长度上性能和准确性的分析



为了满足LSTF的挑战，模型需要有：能够有效地捕捉长序列输入和输出的相互依赖性。最近，Transformer在捕捉长期依赖性上，明显比RNN有着卓越的表现。但存在以下瓶颈：

\1. **自关注机制的二次计算复杂度高**：自关注机制的算子，即点积，导致时间复杂度和每层内存消耗量为  ![image-20220416220449268](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220449268.png)；

\2. **长序列输入下堆叠层的内存瓶颈**：堆叠J层encoder/decoder层让内存使用率为 ![image-20220416220502343](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220502343.png)，限制模型去接受更长的输入；

\3. **预测长输出时推理速度慢**：原始Transformer是动态解码，步进式推理很慢。



**二、论文贡献**

本文贡献如下：

\1. 提出informer，能成功提高在LSTF问题上的预测能力，验证了Transformer模型的潜在价值，能捕获长序列时间序列输出和输入之间的个体长期相关性；

\2. 提出**ProbSparse自相关机制**，使时间复杂度和内存使用率达到![image-20220416220522623](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220522623.png) ；

\3. 提出**自相关蒸馏操作**，在J个堆叠层上突出关注分高的特征，并极大减少空间复杂度到![image-20220416220546752](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220546752.png)，这帮助模型接收长序列输入；

\4. 提出**生成式decoder**，直接一次性多步预测，避免了单步预测产生的误差累积。



**三、网络结构**

Informer的网络结构示意图如下：



![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqE4eSnjzQuwOJ2Eajc2fdAjDXyFGcZ8lvYzu64NRusgn80icCGDTxHFQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图2：网络结构示意图



在图2中：

● 左边：encoder接收大量长序列输入（绿色），Encoder里ProbSparse自关注替换了原自关注模块，蓝色梯形是自关注蒸馏操作，用于抽取主要关注，减少网络尺寸。堆叠层是用于增加鲁棒性。

● 右边：decoder接收长序列输入，用0填充预测部分的序列。它衡量特征map上加权关注成分，然后生成式预测橙色部分的输出序列。



**1. ProbSparse Self-attention**

ProbSparse Self-attention是Informer的核心创新点，我们都知道Transformer里，自关注是有query, key和value组成： ![image-20220416220654456](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220654456.png) ![image-20220416220713501](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220713501.png)能帮助拥有更稳定的梯度，这也可以是其它可能值，但这个是默认的，Transformer作者是担心对于大的Key向量维度会导致点乘结果变得很大，将softmax函数推向得到极小梯度的方向，因此才将分数除以Key向量维度开方值。关于Transformer模型，可以阅读我的历史文章[2]。另外， ![image-20220416220820030](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220820030.png)是非对称指数核函数![image-20220416220832671](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220832671.png) 。



但Transformer中attention分数是很稀疏的，呈长尾分布，只有少数是对模型有帮助的。

![Image](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/640)

图3：在ETTh1数据集上，4层原Transformer的自关注的softmax分数分布



作者在文中提到：**如果存在核心attention的点积pairs，那query的关注概率分布便会远离均匀分布。**意思是说：如果 ![image-20220416220911549](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220911549.png)接近于平均分布![image-20220416220922271](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220922271.png) ，说明该输入V没什么值得关注的地方，因此衡量 ![image-20220416220940588](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220940588.png)的分布p和分布q（即均匀分布![image-20220416220949432](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416220949432.png) ）之间的差异，能帮助我们判断query的重要性，而KL散度便是衡量分布是否一致的一种方式（论文省略了一些推导过程，但好在身边有同事帮忙推导出来了，tql）：

![image-20220416221020860](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221020860.png)

公式(6)中第一项是Log-Sum-Exp（LSE），第二项是算术平均数，散度越大，概率分布越多样化，越可能存在关注重点。但上述公式会有两个问题：

● 点积对的计算复杂度是![image-20220416221036603](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221036603.png) ；

● LSE计算存在数值不稳定的风险，因为形式下，可能会数据溢出报错。



为了解决这两个问题，作者分别采取以下手段：

● **随机采样**![image-20220416221045198](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221045198.png) 个点积对计算![image-20220416221053066](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221053066.png) ；

● 用![image-20220416221102424](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221102424.png) 替换![image-20220416221113317](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221113317.png) （推导过程见论文附录）；



**2. Encoder**

Informer Encoder结构如下：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqTdl70aQtH0AicNXbNjcJDrRbwgq2mDtEzIPLoMzD0Gib34k3O2iclicqjw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图4：Encoder网络结构



Encoder的作用是Self-attention Distilling，由于ProbSparse自相关机制会带来冗余的V值组合，所以作者在设计Encoder时，采用蒸馏的操作不断抽取重点特征，从而得到值得重点关注的特征图。我们能从图4看到每个Attention Block内有n个头权重矩阵，整个Encoder可以由下面的公式所概况：  ![image-20220416221222703](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221222703.png)

![image-20220416221232224](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221232224.png)代表Attention Block，包括多头ProbSparse自相关， ![image-20220416221244749](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221244749.png)是1D-CNN（kernel width=3）， ![image-20220416221257681](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221257681.png)是一种激活函数，外面套了max-pooling层（stride=2），每次会下采样一半。这便节省了内存占用率。



**3. Decoder**

Decoder如图2所示，由2层相同的多头关注层堆叠而成，Decoder的输入如下： ![image-20220416221319007](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221319007.png) 

![image-20220416221333953](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221333953.png)是开始token， ![image-20220416221342458](%E3%80%90%E6%97%B6%E5%BA%8F%E3%80%91Informer%EF%BC%9A%E5%AF%B9%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E6%9B%B4%E9%AB%98%E6%95%88%E7%9A%84Transformer.assets/image-20220416221342458.png)是用0填充预测序列。在使用ProbSparse自相关计算时，会把masked的点积对设为负无穷，这样阻止它们参与决策，避免自相关。最后一层是全连接层，用于最终的输出。



**四、实验结果**

在4个数据集上，单变量长序列时序预测结果：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqs5nTDhYSNIqIddyh3PJg9X3QHwWLqyhEDYuoeDD6kZULOD0ribvpwRg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图5：单变量长序列时序预测结果（4个数据集）



Informer在3种变量下，的参数敏感度：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqxJooHAevIy1viaRqbHm3ZibEZvdIm52oXeTHS9iavHr5813yqfZ7MV1vg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图6：Informer的参数敏感度（3种变量）



ProbSparse自相关机制的消融实验：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqib7iallmIa5JdWoLkMZKPicyA8K8nusaYCPUYpCibFIBiaw62XK3jUk53OQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图7：ProbSparse自相关机制的消融实验



每层的复杂度：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqXuJ68wMTCfxvo1SJ4j7micOFmiavjjZ8e8aC2ZEpaSI3Eo7bStOpIB1Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图8：各模型复杂度对比



自关注蒸馏的消融实验：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqosNVpZYPlNsmMJBkibUcIqBeXfvMcN1nrwQ720luqloz4wWGo8ZE7Jw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图9：自关注蒸馏的消融实验



生成式decoder的消融实验：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqITGo8aicCw06o0H4pYVnHscTuTGpFMz2icd3hpmgDq38aT58hu7w65icQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图10：生成式decoder的消融实验



训练和测试阶段的运行时间：

![Image](https://mmbiz.qpic.cn/mmbiz_png/CcRHkXzUruUia50wOBDzSjvX6UeRJgRGqicfrOlibpO5vMQXYCqatViakqxicX1vaR3ficm6NKYlyWEtSQc9AzE6suEg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图11：训练和测试阶段的运行时间



**五、总结**

总的来说，Informer的亮点在于ProbSparse Self-attention和Encoder with Self-attention Distilling。前者采样点积对，减少Transformer的计算复杂度，让其能接受更长的输入。后者采用蒸馏的概念设计Encoder，让模型不断提取重点关注特征向量，同时减少内存占用。



据身边同事反馈，Informer的效果没有我们想象中那么好，也建议大家可以多试试。



**参考资料**

[1] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021, February). Informer: Beyond efficient transformer for long sequence time-series forecasting. In *Proceedings of AAAI.*

[2] [【务实基础】Transformer](http://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247486160&idx=1&sn=2dfdedb2edbca76a0c7b110ca9952e98&chksm=fa0414bbcd739dad0ccd604f6dd5ed99e8ab7f713ecafc17dd056fc91ad85968844e70bbf398&scene=21#wechat_redirect) - 宅码

