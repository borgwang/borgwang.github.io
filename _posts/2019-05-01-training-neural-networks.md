---
layout: post
title: 神经网络训练二三事
date: 2019-05-01
categories: DL
description: Karpathy 更博了！
---

<br>

[Andrew Karpathy](https://cs.stanford.edu/people/karpathy/) 最近更新了一篇博文：[A Recipe for Training Neural Networks ](http://karpathy.github.io/2019/04/25/recipe/)，阅读过后有不少的收获，推荐阅读。本文整理了原博文中比较有用的点，记录在这里。

<br>

### 目录

1. [分析数据](#1-分析数据)
2. [搭建一个完整的训练/验证 pipeline，得到一个 baseilne 模型](#2-搭建一个完整的训练验证-pipeline得到一个-baseilne-模型)
3. [产生一个更优的模型](#3-产生一个更优的模型)
4. [压榨最后的（验证集）性能提升](#4-压榨最后的验证集性能提升)
5. [其他建议](#5-其他建议)

---

### 1. 分析数据

- 在深入分析数据之前不要开始模型代码的编写，这个阶段应该花较多的时间去观察、分析和理解数据，了解数据各个维度的分布，肉眼寻找相关的 pattern。

- 尝试在数据特征维度上做一些过滤、搜索、排序等操作，并进行可视化。观察数据预处理的结果是否有问题等。

<br>

### 2. 搭建一个完整的训练/验证 pipeline，得到 baseilne 模型

在上一步对数据有了一定的感觉之后可以开始搭建训练/验证的 pipeline，构建这个 pipeline 的时候要确保万无一失，仔细实现以及检查。选定好验证集和评估方法并在后续的实验中保持固定不变。在此基础上可以使用简单的、经典的算法构建 baseline 模型。

这一步中有以下几点需要注意的：

- 固定好随机种子（保证实验可复现）

- 参数的初始化。主要网络最后一层参数的设置，如果回归值期望大概在 50 左右，那么将最后一层的 bias 设置在 50 左右；如果分类的概率大概是 0.1，那么将最后一层的 bias 调整至让网络初始能够输出 0.1 附近的参数。这样做的意义是引入一点人为先验，帮助网络参数初始在一个相对较好的位置，可以加速收敛。

- 设置一个 input-independent baseline。input-independent 是指将数据中的 x 使用 0 或者随机噪声代替，训练得到的一个 baseline 模型。与这个 baseline 模型对比可以看出你当前的模型是否确实从输入数据中提取到了有用的信息

- 设置一个人的 baseline。抽出一部分数据，人肉回归/分类，统计结果作为 baseline。留意在做决策的时候人的关注点在哪

- overfit 一个 batch。这是一个检查性的操作，通过加大模型的容量，在一小个 batch 的数据上面训练，看能不能将 training loss 能否降到非常接近 0。若不能则说明代码流程中可能存在某些 bug 需要仔细检查

- 验证数据的准确性的时候，从数据进入模型的入口进行检查（如 sklearn API 中的 `model.fit(x, y)`）

- 使用一个**固定的具有代表性的**测试集，通过不同的模型在测试集上的效果可以给你一些 intuition，哪些模型是 work 的，哪些模型不 work

<br>

### 3. 产生一个更优的模型

上个阶段我们已经对数据集有了一定的理解，有了一个完整的训练/验证 pipeline，给定一个模型可以产生一个可靠的指标体现模型的性能。我们有了几个 baseline 模型，一个是 input-independent 的模型，一些非常简单的 baseline，可能还有人工的 baseline。因此这个阶段就是在上个阶段的基础上去产生一个更优的模型。

这个阶段可以分成两步：

1. focus on training loss，先专注于降低训练集误差
   - 模型选型：在项目的前期不要使用太 fancy 的模型，优先选择经典的被验证过效果好的模型
   - <u>如果同时多个不同的特征/信号/技巧增加到模型中，不要一股脑全部放进去，尽量 one by one，并且同时记录每个操作带来的影响，保留那些给模型性能带来提升的操作</u>

2. focus on validation loss，**牺牲一部分训练误差来换取验证集误差**
   - 获取更多的数据（第一选择）、在没办法获得更多数据的情况下考虑数据增强
   - Pre-train 几乎没有坏处
   - 更小的 batch size
   - dropout/early-stopping
   - 一个大的 early-stopping 的模型通常要比一个训练更久的小模型泛化性能更好

<br>

### 4. 压榨最后的（验证集）性能提升

- 超参优化。random search 通常比 grid search 效果更好，因为不同的超参对于模型性能敏感度不一样，grid search 均匀搜索的问题是容易在一些不敏感的超参上花费大量的尝试机会，而在更 sensitive 的超参区域缺乏搜索

- 模型集成。模型集成几乎都能够保证模型有性能上的小幅度提升，但是相对应带来的是计算 cost 的增加。

<br>

### 5. 其他建议

- 在实际的项目中，不需要一开始就将代码写的十分通用（通常情况下你也做不到），而是先写一个针对具体问题的非常 specific 的，正确的 run 起来，后面再考虑泛化的事情。

 <br>
 <br>