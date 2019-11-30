---
layout: post
title:  十个样本训练 MNIST 分类器
date: 2019-11-29
categories: DL
description: You only have one shot.
---

<!--START figure-->
<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006tNbRwly1g9g0ojqaifj316q0ighdt.jpg" data-lightbox="mnist-head-figure">
    <img src="https://tva1.sinaimg.cn/large/006tNbRwly1g9g0ojqaifj316q0ighdt.jpg" width="100%" alt="mnist-head-figure" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

#### 目录

-   [前言](#前言)
-   [Baseline](#Baseline)
-   [改进方向 1：数据增强](#改进方向-1数据增强)
-   [改进方向 2 ：利用 GAN 生成 fake 样本辅助训练](#改进方向-2-利用-gan-生成-fake-样本辅助训练)
-   [改进方向 3：Transfer Learning](#改进方向-3transfer-learning)
-   [一些后续可以尝试的点](#一些后续可以尝试的点)
-   [参考文献](#参考文献)

---

#### 前言

前几天看到了 Michael Nielsen 的一篇挺有趣的博客 [Reduced MNIST: how well can machines learn from small data?](Reduced MNIST: how well can machines learn from small data?)，讲的是尝试用少量的样本训练一个 MNIST 分类器。众所周知，MNIST 对于现代机器学习来说是个相对简单的 task，完整的 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/) 有 50k 条的训练数据，10k 条验证数据，10k 条测试数据。用现代 CNN 结构非常容易就达到 98 - 99% 的测试准确率。但是如果不用所有训练数据，只用少量的样本进行训练呢？用多少的数据可以达到一个 90%+ 的准确率，甚至更极端的，如果每个类别只用一个样本进行训练，最高能达到多少的准确率？

这是一个蛮有趣的 topic，不需要多少算力成本就可以 play around，周末两天尝试了一下，只使用 10 条样本，通过传统数据增强、GAN 数据增强、pretrained 模型等方法，可以在测试集上达到 68% 的预测准确率。注：笔者不是研究 zero/one/few shot learning 这个方向的，本文是探索性实验的一些结果和想法的记录，欢迎大家提意见。

本文相关代码在这里: [borgwang/toys/mnist-one-shot](https://github.com/borgwang/toys/tree/master/mnist-one-shot)

<br>

#### Baseline

首先我们先拿经典的统计机器学习模型的效果作为 Baseline。数据方面需要从完整训练集中抽取十个样本（每个类别一个样本），设置 3 个不同的随机种子，抽出来的 3 个训练集如下

<!--START figure-->
<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006y8mN6ly1g99gqz8qnfj30fy0bo752.jpg" data-lightbox="mnist-sample">
    <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g99gqz8qnfj30fy0bo752.jpg" width="70%" alt="mnist-sample" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

10k 条测试数据的结果如下（由于十个样本选取对结果的影响比较大，因此下文会给出 3 次不同种子的实验结果以及平均结果）

| 模型/准确率 | 随机森林 | GBDT   | SVM        | LogisticRegression | KNN (k=1)  |
| ----------- | -------- | ------ | ---------- | ------------------ | ---------- |
| seed=31     | 0.3978   | 0.2869 | 0.3957     | **0.4145**         | 0.3975     |
| seed=317    | 0.4446   | 0.2830 | 0.4406     | **0.4479**         | 0.4406     |
| seed=31731  | 0.3894   | 0.2138 | **0.4296** | 0.4227             | **0.4296** |
| Average     | 0.4106   | 0.2612 | 0.4220     | **<u>0.4284</u>**  | 0.4226     |

综合来说传统的机器学习模型中逻辑回归的效果最好 42.8%，SVM 和 KNN (k=1) 和随机森林效果大致处在同一水平，GBDT 模型在这里效果比较差（可能是因为没有进行参数调优）。通过这个实验我们心里有个底，只用十个样本统计机器学习模型大概是在 40% 的准确率。

我们再看看神经网络的结果，用一个修改版的 LeNet 在十条样本上训练，网络结构为

>   Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Dropout -> FC -> Dropout -> FC

训练 500 个 epoch，batch_size 为 10 ，结果为

|            | LeNet with Dropout |
| ---------- | ------------------ |
| seed=31    | 0.4213             |
| seed=317   | 0.4315             |
| seed=31731 | 0.4354             |
| Average    | <u>0.4294</u>      |

可以看到准确率和 Logistic Regression 基本持平。

<br>

#### 改进方向 1：数据增强

可以看到，无论是传统机器学习模型还是深度学习模型，在只有 10 个样本的情况下单纯通过调参已经比较难以进一步提升了。可以想到的第一个改进的方向就是数据增强，通过对 10 个样本进行拉伸、旋转、扭曲等，理论上应该可以增加数据多样性，减缓模型过拟合。

这里使用了一个数据增强库 [Augmentor](https://github.com/mdbloice/Augmentor)，Augmentor 基本方式是定义一系列的操作，每个操作都有一定的概率被执行，然后生成时他会抽取样本，然后遍历操作，根据概率确定操作是否执行。我们定义了以下的数据增强操作

```python
# 旋转
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
# 随机扭曲
p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=2)
# 透视
p.skew(probability=0.8, magnitude=0.3)
# 拉伸裁剪
p.shear(probability=0.5, max_shear_left=3, max_shear_right=3)
```

这个库支持多种数据增强的操作，这里只用了可能对 MNIST这个数据集比较有用的操作，部分增强如随机 erase、翻转、改变亮度对比度等操作可能作用不是很大，没有采用。我们用这个 pipeline 生成 1024 个数据增强样本。（至于为什么是 1024，上面所有的操作都没有被执行的概率是 0.5*0.2*0.2*0.5 = 0.01，总共有 10 个样本，大约 1000 个样本可以包含 10 个原始的训练样本；前面是瞎编的，就是比较喜欢 1024 这个数字而已）

然后用这 1024 个样本训练前面的 LeNet 网络，batch_size 改为 64，结果如下

|            | LeNet with data_augmentation (1024) |
| ---------- | ----------------------------------- |
| seed=31    | 0.5490                              |
| seed=317   | 0.5612                              |
| seed=31731 | 0.5301                              |
| Average    | <u>0.5467</u>                       |

可以看到进行数据增强后准确率夸张地提升了 12 个百分点 （!），

<!--START figure-->
<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006y8mN6ly1g99izcusuvj30hs0dct9v.jpg" data-lightbox="mnist-data-augment">
    <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g99izcusuvj30hs0dct9v.jpg" width="70%" alt="mnist-data-augment" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

上图是增强后的数据采样，其实看到这些增强后的数据就不难理解了。虽然只是基于简单的拉伸旋转变形，但是由于MNIST 手写的特点，这样简单的变换在一定程度上已经能够模拟很多手写样本，对于泛化行的提升要远大于一些复杂的任务（你很难通过对一张猫的图片进行变换，产生另外一张差异比较大的猫的图片，但是在 MNIST 上可以做到）。

我们可以再更进一步，试着如果产生 10k 条数据增强的样本进行训练，结果如下

|            | LeNet with data_augmentation (10k) |
| ---------- | ---------------------------------- |
| seed=31    | 0.5903                             |
| seed=317   | 0.5451                             |
| seed=31731 | 0.5413                             |
| Average    | <u>0.5589</u>                      |

10k 数据增强相比 1024 数据增强准确率只提升了大概 1 个百分点，说明 1024 已经比较接近数据增强的极限了，因此这里我们基于 1024 数据增强进行后续的改进实验。

小结：MNIST 通过数据增强准确率提升了约 12 个百分点，**但需注意有其特殊性**：一个是数据集相对简单，一个是我们知道真实的训练数据“应该长什么样子”，因此可以朝着正确地方向去增强数据丰富度。其他复杂 task 中数据增强有作用，但提升不会这么明显。

<br>

#### 改进方向 2 ：利用 GAN 生成 fake 样本辅助训练

我们知道 GAN（[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)） 是一种生成模型，可以生成非常逼真的图片，能否利用 GAN 来生成一些 fake 的训练样本作为辅助训练数据呢？答案是可以的。vanilla 的 GAN 生成图片是从一个分布中采样一个向量，然后基于该向量去生成图片，这里我们需要的不仅仅是图片，还需要图片对应的标签，使用 CGAN ([Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)）可以满足需求。CGAN 的网络结构如下图，基本上 CGAN 与 GAN 区别在于，无论是 generator 还是 discriminator 的输入都 concat 了类别的信息，这样使得模型在生成样本时可以指定生成某个类别的样本。更多关于CGAN 的细节有兴趣的读者可以去阅读原 paper。

<!--START figure-->
<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006tNbRwly1g9f1zrtiqjj30c10e80tb.jpg" data-lightbox="mnist-data-augment">
    <img src="https://tva1.sinaimg.cn/large/006tNbRwly1g9f1zrtiqjj30c10e80tb.jpg" width="60%" alt="mnist-data-augment" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

这里笔者只实现了一个全连接版本的 CGAN，即生成网络和对抗网络都是由全连接网络组成，超参数也没有特别地进行调优，采样一部分 CGAN 生成的样本如下图。

<!--START figure-->
<div class="figure">
  <a href="https://tva1.sinaimg.cn/large/006y8mN6ly1g99k9ikwdmj30hs0dcdgo.jpg" data-lightbox="mnist-data-augment">
    <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g99k9ikwdmj30hs0dcdgo.jpg" width="70%" alt="mnist-data-augment" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

看上去生成的质量比较一般，我们看下结果，下面的 gan_ratio 表示使用 CGAN 生成的样本数与原样本数 (1024) 的比例，gan_ratio=0.2 表示生成 204 条 fake 样本，总共 1228 条样本。结果如下表

| gan_ratio  | 0.1               | 0.2            | 0.4            | 0.6            | 1.0            |
| ---------- | ----------------- | -------------- | -------------- | -------------- | -------------- |
| seed=31    | 0.5844            | 0.5569         | 0.5755         | 0.5598         | 0.5030         |
| seed=317   | 0.5552            | 0.5901         | 0.5424         | 0.5180         | 0.5573         |
| seed=31731 | 0.5828            | 0.5452         | 0.5612         | 0.5469         | 0.5627         |
| Average    | **<u>0.5741</u>** | 0.5640         | 0.5579         | 0.5415         | 0.5410         |

可以看到当 gan_ratio 比较小时，确实相比没有使用 CGAN 时要有所提升，gan_ratio = 0.1 时相比没有使用 CGAN 提升了 3 个百分点。但是随着 gan_ratio 的提高，准确率反而逐渐下降，说明生成的样本虽然能增加数据丰富程度，但是毕竟与真实的数据有差别，一定程度上也在“误导”我们的分类器。另外需要注意的是，这里是在使用第一步数据增强产生的数据来训练 CGAN，有可能会导致生成的数据与真实数据的差异进一步增大。

小结：**使用 CGAN 本质上是一种数据增强，在 fake 样本比例合适时能给效果带来一定的提升（gan_ratio = 0.1 时提升了约 3%），但是这取决于生成模型的质量**。这一步有两个方向可以往下探索，一个是 CGAN 的调优，比如将全连接都换成卷积、更优的超参数等，尝试生成更加逼真的数据；另一个是只使用真实的十个样本去训练 CGAN，理论上生成样本与真实样本的差异性会降低。

<br>

#### 改进方向 3：Transfer Learning

前两个改进方向是在数据上面做文章，另一个方向是可以考虑迁移学习 Transfer Learning，这个也是图像task 中常用的一种方法，即使用在大型数据集上面预训练的模型作为初始模型，基于自己的 task 进行 fine-tune。由于 pretrain 的模型已经具备了抽取图像特征的能力，因此理论上应该会对我们的结果有所提升。

这里我们使用了在 ImageNet 上 pretrained ResNet-18 模型，因为ImageNet 的输入图像尺寸是 (224, 224, 3) 因此需要进行调整：首先将 MNIST 的数据从 (28, 28, 1) resize 为 (224, 224, 1)，然后在 pretrained 的模型前面加多一层 kernel 为 1 的卷积层，将 channel 从 1 转化为 3，后面接上 pretrained 的 ResNet-18。

```python
from torchvision.models import resnet18

class MnistResNet(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(1, 3, kernel_size=1)
    self.resnet = resnet18(pretrained=True)
    
  def forward(self, x):
    out = self.conv(x)
    out = self.resnet(out)
    return out
```

在前面数据增强 1024 + CGAN 增强 (gan_ratio=0.1) 的基础上，使用 ResNet-34 训练 20 个 epoch，batch_size 为 64， learning_rate 为 3e-4，结果如下

|            | data_augment (1024) + cgan_augment (gan_ratio=0.1) + pretrained ResNet-34 |
| ---------- | ------------------------------------------------------------ |
| seed=31    | 0.6823                                                       |
| seed=317   | 0.7265                                                       |
| seed=31731 | 0.6289                                                       |
| Average    | <u>0.6792</u>                                                |

可以看到使用 pretrained 的 ResNet-18 相比原先的 LeNet 准确率提升了大概 10 个百分点，三次实验的平均接近 68%。虽然 ImageNet 与 MNIST task 不同，但是由于其训练数据量大，pretrained 的模型具有非常强的特征提取能力，对 MNIST 也有较大的提升。这里由于条件限制只使用了 ResNet-18 的模型，如果使用更深、效果更好的预训练模型，最后的准确率应该还能往上一些。

<br>

#### 一些后续可以尝试的点

-   训练一个更好的生成模型 （CGAN、InfoGAN）
-   设计网络结构。one-shot learning 领域中提出了一些基于比较的网络结构，如 [Siamese Neural Networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)、[Relation Networks](https://arxiv.org/abs/1711.06025)、[Matching Networks](https://arxiv.org/abs/1606.04080) 等，大概的思路是将不同类别样本分类器输出应该不同这个信息利用起来

-   更复杂的数据增强。考虑从特征方面进行数据增强，如使用一些非监督的模型如 VAE 等生成图像特征加以利用

-   使用更大更深的 pretrained 模型？

<br>

#### 参考文献

-   [Reduced MNIST: how well can machines learn from small data?](http://cognitivemedium.com/rmnist)
-   [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
-   [One-shot Learning survey](http://ice.dlut.edu.cn/valse2018/ppt/07.one_shot_add_v2.pdf)
-   [Pytorch Documentation](https://pytorch.org/docs/stable/)

<br><br>
