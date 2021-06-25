---
layout: post
title: Optimization Methods in Deep Learning
date: 2017-4-19
update_date: 2019-09-08
categories: DL
description: SGD, Adam, RMSProp, NAG, Adadelta, Adagrad
---

<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006y8mN6gy1g6ritesh3aj316e0fw7wh.jpg">
    <img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g6ritesh3aj316e0fw7wh.jpg" width="100%" alt="optimize_method_head_img" referrerPolicy="no-referrer"/>
  </a>
</div>

<br>

本文对 Deep Learning 中常见的优化方法进行推导和小结，包括传统的梯度下降(GD、SGD、 minibatch-SGD)，引入惯性动量的 Momentum 和 NAG 方法、自适应学习率的 AdaGrad、 AdaDelta、Adam、RMSProp 方法，然后通过可视化对几种方法的性质进行分析比较。

<br>

### 目录

- [Gradient descent variants](#gradient-descent-variants)
- [Momentum](#momentum)
- [Nesterov Accelerated Gradient（NAG）](#nesterov-accelerated-gradientnesterovpdfnag)
- [Adagrad](#adagrad)
- [Adadelta](#adadelta)
- [RMSProp](#rmsprop)
- [Adam（Adaptive Moment Estimation）](#adamadaptive-moment-estimation)
- [小结](#%e5%b0%8f%e7%bb%93)
- [可视化](#可视化)
- [Final words](#final-words)

---

### Gradient descent variants

1. Gradient descent
  - 每次取所有训练数据计算一个梯度进行一次更新。
  - 优点：保证收敛（全局或局部最优） 缺点：计算量大
2. Stochastic gradient descent（SGD）
  - 每次只取一个训练数据计算梯度更新。
  - 优点：速度快； 缺点：梯度 variance 大
3. minibatch stochastic gradient descent
  - 前两种的折中，每次取一个 batch 的数据计算梯度，广泛应用于现代神经网络训练
  - 优点：比 SGD 更好的梯度稳定性和比 BDG 更少的计算量
  - 缺点：每次更新对所有的参数的更新都是等权重的；容易在鞍点被困住

<br>

### Momentum

Momentum 方法计算梯度的时候考虑到之前的梯度（惯性动量），使用[指数滑动平均](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average)（Exponential Moving Average）对当前梯度与过去梯度进行滑动平均，其公式如下：

  <div class="formula">
    $$ v_t=\gamma v_{t-1}+\eta\bigtriangledown_\theta J(\theta) $$
    $$ \theta=\theta-v_t $$
  </div>

>其中动量参数 $$ \gamma $$ 一般取 0.9

优点：每一步的更新考虑到之前的梯度，因此可以减少梯度的震荡、曲折，使朝着比较正确的方向前进，加速收敛。

<br>

### Nesterov Accelerated Gradient（NAG）

NAG 方法也是属于一种利用动量的方法，与 Momentum 的区别是：Momentum 参数更新是由当前参数的梯度 $$ \bigtriangledown J(\theta) $$ 加上之前的动量 $$ v_{t-1} $$ 计算得到，NAG 则是先计算基于之前的动量移动后得到的新位置的梯度 $$ \bigtriangledown J(\theta-\gamma v_{t-1}) $$ ，再将该梯度加上前面的动量 $$ v_{t-1} $$ 得到参数更新梯度。具体公式如下：

  <div class="formula">
    $$ v_t=\gamma v_{t-1}+\eta\bigtriangledown J(\theta-\gamma v_{t-1}) $$
    $$ \theta=\theta-v_t $$
  </div>

下图可以看到 NAG 的计算过程：

<div class="figure">
  <a href="/assets/figures/nag.png" data-lightbox="nag_optimization">
    <img src="/assets/figures/nag.png" width="60%" alt="nag_optimization" referrerPolicy="no-referrer"/>
  </a>
</div>

蓝色的箭头表示经典的 SGD 方法：一个箭头代表一次的参数更新

棕红绿箭头表示 NAG 方法的：棕色箭头代表上一步的动量 $$ v_{t-1} $$，先按照该动量走一步，红色箭头代表在当前位置的梯度 $$ \eta\bigtriangledown J(\theta-\gamma v_{t-1}) $$，最后 NAG 方法更新的方向就是棕红两个向量的和 — 绿色的箭头。

特点：NAG 和 Momentum 方法都使用了惯性，区别是如何使用。Momentum 是先计算当前位置的梯度再加上惯性，而 NAG 是先按照之前的惯性走一步，再计算新位置的梯度最后加起来。这种“前瞻”的特性使得 NAG 能够进一步减少梯度的震荡，加快收敛。

<br>

### [Adagrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

基本思想动态地对不同参数调整学习率，即对参数的更新不是等权重的。更新公式如下

  <div class="formula">
    $$ \theta_{t+1,i}=\theta_{t,i}-\frac{\eta}{\sqrt{G_{t,ii}+\epsilon}} g_{t,i} $$
  </div>

Adagrad 相比 SGD 学习率多了一个调整项，其中 $$ \sqrt{G_t} $$ 是一个对角线矩阵， $$ G_{t,ii} $$ 表示参数 $$ \theta_i $$ 过去累积的梯度平方和， $$ \epsilon $$ 是为了避免分母为 0。过去累计梯度平方和越大的学习率就越小，反之亦然。

```python
# ----- Adagrad -----
learning_rate = 0.01
G = 0
for i in range(steps):
    grad = ...		 # calculate grad
    G += grad ** 2
    delta = - (learning_rate / (G + eps) ** 0.5) * grad
    grad += delta
```

优点：根据不同的参数调节学习率，适用于处理稀疏梯度
缺点：随着训练的进行分母调整项会越来越大，使得学习率会越来越小，后期基本上很难学到新的东西

<br>

### [Adadelta](https://arxiv.org/abs/1212.5701)

Adadelta 是 Adagrad 的改进版本，为了解决 G 随时间单调递增的问题和需要设置全局学习率的问题。

Adagrad 调整项的分母是 $$ \sqrt{G_{t,ii}+\epsilon} $$ ，其中 $$ G_{t,ii} $$ 会不断累加，Adadelta 将其换成 $$ g^2 $$ 的 running average $$ E[g^2] $$。

<div class="formula">
  $$ \Delta \theta_t=\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}} g_t $$
</div>

其中$$ E[g^2]_t $$ 由指数滑动平均维护：

  <div class="formula">
    $$ E[g^2]_t=\gamma E[g^2]_{t-1}+(1-\gamma)g_t^2 $$
  </div>

这样解决了 Adagrad 由于调整项分母单调递增导致的后期学习率过小的问题。

观察到上面更新公式种存在两边单位不一致的问题，（左边单位是 theta，右边的单位是 1），将全局学习率 $$ \eta $$ 换成的单位：

  <div class="formula">
    $$ RMS[\Delta\theta]_t=\sqrt{E[\Delta_\theta]^2+\epsilon} $$
  </div>

其中$$ E[\Delta\theta]^2 $$ 同样采用指数滑动平均方式更新：

  <div class="formula">
    $$ E[\Delta\theta^2]_t=\gamma E[\Delta\theta^2]_{t-1}+(1-\gamma)\Delta\theta_t^2 $$
  </div>

计算 t 时刻的 $$ \Delta\theta $$ 时使用 t-1 时刻的 $$ RMS[\Delta\theta] $$ ，完整更新公式如下：

  <div class="formula">
    $$ \Delta\theta_t=\frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t} g_t $$
    $$ \theta_{t+1}=\theta_t-\Delta\theta_t $$
  </div>

伪代码如下：

```python
# ----- Adadelta -----
MS_delta = 0
MS_grad = 0
eps = 1e-8
decay = 0.9
for i in range(steps):
    grad = ...
    MS_grad = decay * MS_grad + (1-decay) * grad ** 2
    # calculate delta
    delta = -((MS_delta + eps) ** 0.5 / (MS_grad + eps) ** 0.5) * grad
    # update mean_square_delta
    MS_delta = decay * MS_delta + (1-decay) * delta
    # update params
    grad += delta
```

Adadelta 在实际应用中通常收敛速度较慢，不被经常使用，原因在于统一单位项 $$\Delta\theta_t=\frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t}$$ 在初期的时候分子项非常小，接近 0，因此导致初期的步长非常小，优化速度非常慢。

<br>

### [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
RMSProp 其实可以看作是 Adadelta 的一个特例（没有考虑单位统一，只维护一个 $$ g^2 $$ 的指数滑动平均 $$ E[g^2] $$ 用于解决学习率消失的问题，其更新公式为：

<div class="formula">
  $$ E[g^2]_t=\gamma E[g^2]_{t-1}+(1-\gamma)g_t^2 $$
  $$ \Delta\theta_t=\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}} g_t $$
</div>

伪代码如下：

```python
# ----- RMSProp -----
learning_rate = 0.001
decay = 0.9
for i in range(steps):
    grad = ... 		# calculate grad
    MS_grad = decay * MS_grad + (1-decay) * grad ** 2
    # calculate delta
    delta =   - ( learning_rate / (MS_grad + eps) ** 0.5  ) * grad
    grad += delta
```

RMSProp 是 Adagrad 的发展，也是 Adadelta 的变体，效果趋于两者中间，适合用于处理非平稳目标（对 RNN 效果好）

<br>

### [Adam](https://arxiv.org/abs/1412.6980)（Adaptive Moment Estimation）

Adam 可以看作是带有 momentum 的 RMSProp，Adam 维护两个近似均值 $$ E[g] $$ 和 $$ E[g^2] $$

<div class="formula">
  $$ m_t=\beta_1 m_{t-1}+(1-\beta_1) g_t $$
  $$ v_t=\beta_2 v_{t-1}+(1-\beta_2) g_t^2 $$
</div>

然后对这两个近似期望进行偏差校正（因为初始 $$ m_0, v_0 $$ 为 0）

<div class="formula">
  $$ \hat{m_t}=\frac{m_t}{1-\beta_1^t} $$
  $$ \hat{v_t}=\frac{v_t}{1-\beta_2^t} $$
</div>

最后更新公式如下：

<div class="formula">
  $$ \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat{v_t}+\epsilon}}\hat{m_t} $$
</div>

相比 RMSProp，Adam 不仅维护了梯度平方的滑动平均 $$ E[g^2] $$ 动态调整学习速率，同时维护梯度的指数滑动平均 $$ E[g] $$，相当于在 RMSProp 的基础上引入了 Momentum。

代码如下：

```python
# ----- Adam -----
m_decay = 0.9
v_decay = 0.999
learning_rate = 0.001
for i in range(steps):
    grad = ... 		# calculate grad
    m = m_decay * m + (1 - m_decay) * grad
    v = v_decay * v + (1 - v_decay) * grad ** 2
    m_ = m / (1-m_decay ** i)
    v_ = v / (1-v_decay ** i)
    # calculate delta
    delta = - (learning_rate / v_ ** 0.5 + eps) * m_
    grad += delta
```

<br>

### 小结

上述几种优化方法均由梯度下降法演变而来，主要从两个方面进行改进：**动量**和**自适应学习率**。

- Momentum 从动量改进（计算梯度加上动量）
- NAG 从动量改进（先加上动量再计算梯度）
- AdaGrad 从自适应学习率改进（使用历史梯度平方根调整学习率）
- RMSProp 从自适应学习率改进 （使用历史梯度平方根的指数衰减调整学习率）
- AdaDelta 在 RMSProp 基础上对梯度单位进行归一
- Adam 则是动量和自适应学习率两方面都进行改进。一方面借鉴了 RMSProp 用历史梯度平方根指数衰减调整学习率，同时也使用了指数衰减的动量调整梯度

<br>

### 可视化

对上文介绍的几种优化方法进行可视化。为了方便可视化，假设参数只有两个维度，模拟生成一个损失函数的平面，上述几种优化方法在该平面上的优化路径如下图

<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006y8mN6ly1g6rk16dyaqg30h00c04qv.gif">
    <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g6rk16dyaqg30h00c04qv.gif" width="80%" alt="optimization_visualization" referrerPolicy="no-referrer"/>
  </a>
</div>

对应的收敛速率如下图

<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006y8mN6gy1g6rm5cvfqzj30i50b9406.jpg">
    <img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g6rm5cvfqzj30i50b9406.jpg" width="80%" alt="optimization_converge_rate" referrerPolicy="no-referrer"/>
  </a>
</div>

从以上两幅图可以观察

1. 大部分优化算法都能够比较顺利地朝着全局最优行进
    这个主要有两个原因：
      - 这个模拟的损失函数平面其实非常理想（足够平滑），真实的损失函数平面更加层峦叠嶂、蜿蜒曲折
      - 初始点选的好，没有选在局部最优的附近。这也从侧面说明参数初始化的重要性:)

2. 虽然大家都基本都朝着正确的方向优化，不同方法优化路线和收敛速度有差别
    图中每种优化算法都迭代了 200 步，观察他们各自的优化路线和收敛速度就可以比较直观了感受他们之间的区别。
    - Adam、Momentum、NAG 三种引入了动量的方法速度最快，RMSProp（绿色）由于只有自适应学习率没有动量，因此在最优点附近出现了震荡
    - Adagrad（粉色）虽然对学习率做改进，但是调整项分母单的调递增导致的后期学习率过小，收敛速度比较慢；
    - 原生的梯度（黑色）下降没有任何处理，速度比较更慢；
    - Adadelta 由于存在统一单位项，使得初期步长非常小，因此在 200 步的迭代中几乎没有优化。

    这与我们实际应用的经验相符合，通常 Adam 和 Momentum（或 NAG）的方法效果更好，而原生梯度下降、Adadelta 等方法则较少被使用。


其他更多平面上各种优化方法的差异可以看下面这个视频，这里笔者用了随机种子生成了多个随机的光滑平面。综合来说，大部分时候 Adam 优化方法都是最快收敛的，但有时候也会收敛到局部最优。

<!--START video-->
<div class="video">
    <video width="80%" controls>
        <source src="https://raw.githubusercontent.com/borgwang/toys/master/visualize_optimization_methods/optimization_methods_video.mp4" type="video/mp4">
    </video>
</div>
<!--END video-->

<br>

上述可视化的代码可以在 [这里](https://github.com/borgwang/toys/tree/master/visualize_optimization_methods) 找到，包含了生成随机光滑平面，记录的优化方法的优化路径并且可视化出来。读者有兴趣可以参考一下，也可以自己修改地图、试验观察比较其他不同的优化方法。

<br>

### Final words

本文介绍的优化方法是梯度下降及其各种改进版本，本质上都是梯度下降，都是一阶优化算法。除了梯度下降外，还有投影次梯度下降法、近端梯度下降法、坐标下降法等一阶优化算法。另外也有利用二阶导数信息的牛顿法、拟牛顿法等等。由于本文主要关注深度学习中常用的优化方法，因此对这部分算法没有展开介绍，如果读者有兴趣可以查阅相关资料深入了解。Thanks for reading.🤘


<br><br>
