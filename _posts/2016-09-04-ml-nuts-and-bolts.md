---
layout: post
title: ML Nuts and Bolts
date: 2016-09-4
categories: ML
description:
---

<br>

本文整理机器学习中比较重要的、常见的知识点，以及作者在学习过程中遇到的一些问题和笔记。主要包括过拟合欠拟合、常用损失函数、训练集验证集测试集、常见降维技术、特征工程相关小结、Learning theory 相关知识等。  

<br>

### 目录

- [Basic recipe for ML](#basic-recipe-for-ml)
- [分类和回归的区别](#分类和回归的区别)
- [Overfitting and Underfitting](#overfitting-and-underfitting)
- [分类任务中的一些问题](#分类任务中的一些问题)
    1. [多分类问题](#多分类问题)
    2. [Skewed Classes 问题](#skewed-classes-问题)
- [Loss Functions](#loss-function)
- [Train/Validation/Test Set](#trainvalidationtest-set)
- [Feature Engineering](#feature-engineering)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Useful Tips from Machine Learning Yearning](#useful-tips-from-machine-learning-yearning)
- [Active Learning](#active-learning)
- [Learning Theory](#learning-theory)
    1. [VC 维](#vc-维)
    2. [Empirical Risk and True Risk](#empirical-risk-and-true-risk)
- [Ensemble Methods](#ensemble-methods)

---

### Basic recipe for ML

对于经典的监督学习，主要可以分成几个不同的阶段：

1. 拟合训练数据。这一阶段主要的任务是尽可能拟合训练数据，确保 training error 达到合理低的数值（以人类表现或者贝叶斯误差作为参照标准）。如果与人类表现差别过大，可能是训练数据本身质量太差，也可能是神经网络结构不合适或者训练时间过短等原因导致。应首先解决欠拟合的问题再考虑下一步流程

2. 判断模型是否过拟合。判断的标准就是看 dev set 的性能。主要比较 training error 和 dev error，如果相差过大，则说明过拟合，首先考虑收集更多的数据（在可能的情况下这是第一选择），第二是可以采取 regularization 方法，比如 L1/L2、dropout 等

<!--START figure-->
<div class="figure">
  <a href="https://ws2.sinaimg.cn/large/006tKfTcgy1g1kvqs9fkxj31780ik44s.jpg" data-lightbox="ml_recipe">
    <img src="https://ws2.sinaimg.cn/large/006tKfTcgy1g1kvqs9fkxj31780ik44s.jpg" width="90%" alt="ml_recipe" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

3. 观察 dev set 和 test set 的性能差距。如果差距过大可能是因为 dev set 或者 test set 过小。

4. 观察 test set 性能和实际任务中的表现。如果差距过大（实际中的性能远比 test set 性能差，则要考虑是否是否使用了合适的评测指标进行 evluate）

<!--START figure-->
<div class="figure">
  <a href="https://ws1.sinaimg.cn/large/006tKfTcgy1g1kvsbdammj319c0f0ak7.jpg" data-lightbox="ml_recipe2">
    <img src="https://ws1.sinaimg.cn/large/006tKfTcgy1g1kvsbdammj319c0f0ak7.jpg" width="100%" alt="ml_recipe2" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

<br>

### 分类和回归的区别

可以从以下几个方面来考虑两者的区别：

1. 模型预测输出的区别

   这是最基本的区别，分类模型的预测输出是离散值，回归模型的输出是连续值

2. 模型训练（损失函数）的区别

   通常情况下回归算法和分类算法会采取不同的损失函数。如回归算法会采用均方差损失函数，分类问题会采用交叉熵或负对数损失。

   **为什么需要采取不同的损失函数？**

   均方差损失函数本质上就是假定了预测与真实值之间的误差为正态分布下的极大似然估计法。而在分类问题上，误差并不服从正态分布，因此均方差损失函数不是最佳的选择。通常采用了交叉熵损失函数。交叉熵损失函数可以从两个方面解释：1. 极大似然估计 2. 信息论，KL 散度的角度。

3. 更本质的解释

   本质上机器学习的算法都是在拟合一个 $$ x $$ 到 $$ y $$ 的映射关系，只是取决于不同问题下我们希望 $$ y $$ 是一个连续的还是离散的形式，并针对 $$ y $$ 的形式设置不同的损失函数，使其更有利于模型的训练。以最简单的线性模型为例子，如果我们希望直接输出是连续值，则直接对 $$ y $$ 进行线性拟合（线性回归）。如果我们希望输出是离散值，则可以对 $$ wx+b $$ 进行 sigmoid 函数的变化，将其值域压缩在 (0, 1) 之间，并通过一个阈值实现分类输出（逻辑回归）。也可以从另一个方面看逻辑回归，如果对逻辑回归表达式进行变化，可以发现其本质上也是一种线性回归，只是其不直接对 $$ y $$ 进行回归，而是对对数几率 $$ \mathop{\ln}(\frac{y}{( - y})$$ 进行回归。

<br>

### Overfitting and Underfitting

过拟合通常指模型对训练数据过渡拟合，导致模型泛化能力差。通常在使用表征能力强的模型（如深层神经网络等）在不足量的训练数据上过度训练时容易出现过拟合，此时模型会简单地去模仿、记住训数据，导致其在训练集上性能良好，但对于未见过的数据泛化能力差。

对于一个过拟合的模型，如果新数据与训练数据相似，那么模型表现会非常好；反之如果新数据与训练数据不相似，那么模型表现会非常差（因为泛化性能差）。即模型的表现波动非常大（取决于新数据是否与训练数据相似），因此过拟合的模型是高方差（High Variance）。

常用的解决过拟合的方法：
- 使用更多的训练数据（在有条件的情况下是第一选择）、数据增强
- 使用正则化技术（Regularization）：L1 或 L2 正则化、Dropout、BatchNormalization 等

过拟合是模型对于训练数据拟合过度了，反之欠拟合（Underfitting）则是模型无法很好地拟合训练数据的规律。通常在模型容量无法满足数据本身复杂度（比如使用一个线性模型拟合非线性数据）、训练时间过短的情况下容易出现。欠拟合模型通常在训练集和测试集上的性能都较差，与真实预测偏差都较大，因此称欠拟合模型是高偏差（High Bias）。

常见的解决欠拟合的方法：
- 尝试更加复杂、具有更强表征能力的模型（如神经网络）
- 增加数据特征数目
- 训练更长的时间

对于一个经典的监督学习任务，我们首先考虑在训练数据上拟合好，然后再考虑模型的泛化能力。即我们通常先考虑模型是否出现欠拟合（Training Error），能够较好地拟合训练数据后再考虑其泛化能力（Validation Error）。整个流程如下图。

<!--START figure-->
<div class="figure">
  <a href="/assets/figures/ml-recipe.png" data-lightbox="ml_recipe3">
    <img src="/assets/figures/ml-recipe.png" width="80%" alt="ml_recipe3" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

<br>

### 分类任务中的一些问题

#### 多分类问题

多分类有两种思路，可以直接使用适合多分类的模型（比如神经网络、LR 等），也可以使用多个简单的二分类器（SVM 等）组合成为一个多分类模型。采用哪种方案具体看训练数据：  
1. 如果样本与类别都是一对一的关系，则可以直接使用多分类的模型；
2. 如果样本与类别存在一对多的关系，则考虑采用多个二分类器的方案；

由二分类器构建多分类器有以下三种常用的方法（假设共有 4 类）：   

- 一对其余（One-vs-Rest）  
训练 4 个分类器  （[1], [2,3,4]）（[2], [1,3,4]）（[3], [1,2,4]）（[4], [1,2,3]）。  
问题：可能出现分类重叠和无类可分的现象；  
训练过程可能出现「数据集偏斜」的问题；  
- 一对一（One-vs-One）  
训练!(4-1) = 3+2+1 = 6 个一对一的分类器（1,2）（1,3）（1,4）（2,3）（2,4）（3,4）逐个进行分类。  
问题：仍然可能出现分类重叠，但不会出现无类可分；  
需要训练大量的分类器（k 类需要训练 k(k-1)/2 个分类器；  
- DAG 一对一(DAG One-vs-One)：
依然需要训练 k * (k - 1) / 2 个分类器，训练的时候还是 1 v 1 训练，分类时采用有向无环图的方法进行分类。
优点：加快分类的速度（减少分类次数），并且不会出现分类重复和无类可分的现象；  
缺点：某个节点分类错误后，接下来无论怎么分都不会分到正确的节点（正确的类别在接下来不会再出现）

<!--START figure-->
<div class="figure">
  <a href="/assets/figures/dag.png" data-lightbox="multiclass_classification_dag">
    <img src="/assets/figures/dag.png" width="80%" alt="multiclass_classification_dag" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

#### Skewed Classes 问题   

实际分类任务中训练数据的类别不均衡问题是很常见的情况，称为 Skewed classes 问题。类别不均衡一些解决思路以及注意事项：  

- 收集更多（少数类）数据，在可能的情况下这是第一选择；
- Resample: 对类别多的类下采样（剔除掉一些），对类别少的类上采样（但要避免直接重复采样，可以加入一些噪声）  
- 数据合成/增强：对于少数类不断重复采样会导致模型不能够很好地学习到特征。对于图像分类问题，可以通过对少数类图像进行旋转、翻转、加入随机噪声等方式合成样本增加少数类样本数量；
- Weight the cost：对少数类设置更高的权重（分类错误时设置更大的惩罚，或者针对少数类训练样本设置更大的学习速率）
- Precision&Recall：针对类别不均衡情况下衡量分类性能的指标，公式如下：

  <!--START formula-->
    <div class="formula">
      $$ Precision=\frac{TP}{TP+FP} $$
      $$ Recall=\frac{TP}{TP+FN} $$
    </div>
  <!--END formula-->

  >Precision 和 Recall 两者同时高时表明分类效果越好, 使用 F1 score 把两个指标统一起来。     

  <!--START formula-->
    <div class="formula">
      $$ F1=\frac{2PR}{P+R} $$
    </div>
  <!--END formula-->

  >当 P 和 R 都高时, F1 score 也会高。

- AUC：
  一种二分类评测指标，对样本是否均衡不敏感，广泛应用于不均衡样本。  

  AUC 考虑多个分类的 threshold，从左下角往右上角逐渐降低 threshold，每个 threshold 分别计算横坐标 False Positive Rate(FPR)、纵坐标 True Positive Rate(TPR)，即可画出 ROC 曲线。 False Positive Rate 计算公式为 $$ FPR=\frac{FP}{N} $$，表示所有真实负样本中被错误分类为正的比例；True Positive Rate 计算公式为 $$ TPR=\frac{TP}{P} $$，表示所有真实正样本中被正确分类为正的比例。<u>AUC 值为 ROC 曲线下半部分与座标轴围成的面积 </u>，直观理解 AUC 值表示模型正确分类出正样本的概率 $$ P1 $$ 大于错误分类为正样本的概率 $$P2$$ 的概率 $$ AUC=P(P1>P2)$$。

  <!--START figure-->
  <div class="figure">
    <a href="/assets/figures/auc.png" data-lightbox="AUC">
      <img src="/assets/figures/auc.png" width="60%" alt="AUC" referrerPolicy="no-referrer"/>
    </a>
  </div>
  <!--END figure-->

<br>

### Loss Functions

- 0-1 损失（二分类）

<!--START formula-->
  <div class="formula">
    $$ L_{0-1}=\begin{cases} 1,\quad y\neq f_\theta(x) \\ 0,\quad y=f_\theta(x) \end{cases} $$
  </div>
<!--END formula-->

- Cross Entropy cost （二分类）

  与 KL 散度的联系：KL 散度衡量两个分布 p 和 q 的相似性,这里 $$ p,q $$ 分别指真实分布 $$ y $$和模型预测分布 $$ h_\theta(x) $$ 。KL 散度计算公式为 $$ \mathbb{KL}(p\|q)=-\mathbb{H}(p)+\mathbb{H}(p,q) $$，其中 $$ \mathbb{H}(p) $$ 是分布 p 的自信息熵，$$ \mathbb{H}(p,q) $$ 是分布 $$ p,q $$ 的交叉熵。我们要使预测与真实分布尽量相似，即最小化 KL 散度；由于模型参数只与 KL 散度第二项交叉熵相关，因此最小化真实分布与模型预测分布的 KL 散度相当于最小化交叉熵损失。

<!--START formula-->
  <div class="formula">
    $$ L_{CE}(\theta)=-(y\log(h_\theta(x))+(1-y)\log(1-h_\theta(x))) $$
  </div>
<!--END formula-->

- Hinge loss （二分类）
也是一种二分类的损失，常用于 SVM 模型中，其表达式为：

<!--START formula-->
  <div class="formula">
    $$ L_{Hinge}(\theta)=\max\{0,1-h_\theta(x)\} $$
  </div>
<!--END formula-->

- 指数损失（二分类）

  常用于 Adaboost 模型中，损失函数表达式为

<!--START formula-->
  <div class="formula">
    $$ L_{EXP}(\theta)=exp[-y h_{\theta}(x)] $$
  </div>
<!--END formula-->

- Negative Log-Likelihood loss 负对数似然损失（多分类）  

  常用多分类模型，最小化负对数的似然, 本质是极大似然估计。公式如下，其中 x 为正例对应的样本， $$ h_\theta(x) $$为模型对于 ground true 类别的预测值。当$$ h_\theta(x) $$接近 1 时, $$ J(\theta) $$接近 0, 反之$$ J(\theta) $$接近无穷大。

<!--START formula-->
  <div class="formula">
    $$ J_{NLL}(\theta)=-\log(h_\theta(x)) $$
  </div>
<!--END formula-->

- Mean Square Error（回归）

<!--START formula-->
  <div class="formula">
    $$ J_{MSE}(\theta)=(y-h_\theta(x))^2 $$
  </div>
<!--END formula-->

<br>

### Train/Validation/Test Set  

- 在经典的监督学习中，通常将一个数据集划分为训练集（Train Set）、验证集（Validation Set）和验证集（Validation Set）三个集合。其中训练集用于训练模型, 验证集用于对训练得到的模型进行超参调优以及模型选择，测试集用于验证最终选择的模型的性能。

**NOTE**：  
- 测试集应该完全作为测试最终模型性能使用（不应该用于模型调优选择）。一般使用训练集进行训练，然后在验证集上进行超参调试，选定最后的模型。确定模型后再使用测试集测试模型的最终性能。(这么做的原因是测试集作用是模拟模型在未见过的数据上面的表现, 即泛化能力, 如果使用了测试集进行训练或者模型选择, 则不能模型真实泛化能力)

- 三种集合的划分比例按照数据量大小而定，general 的原则是在保证验证集和测试集的数量足以分辨出模型的性能提升精度的前提下，训练集越大越好。对于数据量不大（10000 以下）的任务，通常可以按 0.6/0.2/0.2 或 0.7/0.15/0.15 比例划分三种集合；而在具有百万数量级的数据集中，测试集和验证集的比例可以非常小（比如 1%）。

- 划分时需要注意的事项：验证集和测试集应该服从相同或尽量相似的分布；同时应选择能够反映真实问题(想解决的问题)的数据作为验证集和测试集，保证模型改进的方向是正确的。在某些情况下，训练集的分布和测试集、验证集的分布稍微不同是可以接受的。

- K-fold Cross Validation   
当数据量有限时, 可以对同一个数据集进行多次不同的划分. 得到多个 validate eror 最后取平均(k 通常取 10)  
多次划分实际上相当于使用到了整个训练集进行训练, (单次划分实际上有 1/k 的训练数据被浪费了).   

<br>

### Feature Engineering

- 特征分析  
  可视化特征的分布，连续型特征如果是长尾分布可以考虑采用对数变换；离散型如果某些 label 数量太少可以考虑合并为一类；
  对特征进行相关性分析，发现高相关性和共线性的特征。

- 特征预处理  
  连续值无量纲化（标准化、正则化、区间缩放），使不同量纲的特征可以一起计算；
  1. 离散特征 -> 连续特征
    - 直接按顺序数值化 (Label encode)；
    - 考虑使用 one-hot 编码(Onehot encode)；
  2. 连续特征 -> 离散特征
    参考：https://www.zhihu.com/collection/107663148
    1. 单个连续变量可以离散为多个离散特征，引入非线性能力，能增强鲁棒性，降低过拟合的风险
    2. 可以看作是“大量离散特征 + 简单模型” 与“少量连续特征 + 复杂模型” 的 trade-off

- 缺失值处理
  1. 如果某个特征的缺失量较大，可能直接删除；
  2. 对于缺失值较少的，可以采取以下几种方法：
    - 直接将 null 作为一个特征的值
    - 连续量采用均值、中位数、众数填充
    - 使用算法拟合填充，即使用没有缺失的数据训练一个模型(KNN、 NB、LR)拟合缺失的数据（比较麻烦但效果比较好）

- 异常值处理
  1. 基于规则处理：
    - 均值加上 k 倍方差为上下界；
    - 如果是时间序列的数据，剔除出现中断的数据；
  2. <u>基于模型处理</u>：
    可以尝试使用欠拟合的模型去训练，剔除掉残差最大的 k%样本。（用欠拟合的模型可以防止学习能力太强学到太多异常情况）
  3. 其他
    - 针对特征数目太多的情况，PCA、 LDA、t-SNE 等
    - 对于对距离比较敏感的模型（如 KNN），最好 <u>使输入特征尽量正交</u>。如果特征高度相关的话，会让距离的计算失真，会影响对距离比较敏感的模型的精度。产生正交特征的简单方法就是用 PCA。

- 特征选择  
  考虑特征是否发散（方差大小），如果方差接近 0 则这个特征对数据没有什么区分度，考虑舍去。  
  特征重要性
    1. 训练单个特征的模型，依据单个特征模型的性能对特征进行排序；
    2. 计算各个特征对目标值的皮尔逊系数，选择相关程度高的特征；
    3. 通过 L1 正则化选择特征；
    4. 借助可以对特征进行打分的预选模型（比如 RF）得到特征的 importance，进行筛选；
    5. 通过神经网络提取特征（autoencoder，或者跑某个相关任务直接提取网络隐层）；
  合成（组合）特征：
      构建交叉特征的相关矩阵，筛选相关比较低的特征合成交叉特征（在推荐系统和广告系统中比较常见）。

- 其他
  1. 模型选择：  
    特征稀疏 —> 线性模型；特征稠密 -> 树模型或神经网络；  
  2. 模型调优：  
    gridsearch，贝叶斯优化，通过 bagging，blending，stacking 提高泛化能力；  
  3. 模型评估：  
    分类：准确率、log-loss、精准率/召回率、AUC
    回归：validation / test error

<br>

### Dimensionality Reduction

- 主成分分析（PCA）

  Intuition：方差最大化的思想，即找到一组正交基使得变换座标后的数据各个维度的 variance 最大（保留最多信息）
  有特征值分解和 SVD 两种方法实现 PCA。

- 线性判别式分析（LDA）

  一种基于监督学习的降维技术，也叫 Fisher 线性判别。基本思想就是找到一种投影方式，使得类内方差尽量小，类间距离尽量大（每个类使用类均值表示）。  
  算法流程如下：

  >1. 计算每个类别样本的均值向量
  >
  >2. 通过均值向量计算类间散度矩阵 $$ S_B $$ 和类内散度矩阵 $$ S_W $$
  >
  >3. 对 $$ S_W^{-1}S_B W=\lambda W$$ 进行特征值求解，求出 $$
  z S_W^{-1}S_B$$ 的特征值和特征向量
  >
  >4. 对特征向量按照特征值的大小降序，选择前 K 个特征向量组成投影矩阵
  >
  >5. 通过(D, K)维的特征值矩阵将样本点投影到新的空间中，$$ Y=X*W$$


  LDA 和 PCA 对比：  
  两者都利用了特征值分解技术，PCA 分解的是数据的协方差矩阵，LDA 分解的是类间散度矩阵的逆矩阵和类内散度矩阵的积。
  <u>LDA 在降维的过程中考虑的类别信息，而 PCA 单纯从数据本身特点（各个维度的方差信息）出发。</u>

- Auto-encoder

  神经网络模型，隐层维度比输入小，输入和输出维度相同，通过迫使网络输出和输入相同学习到数据的隐层特征。

- t-SNE

  分别在原维度（高维）和低维空间度量数据点之间的相似度（使用高斯核度量） ，得到两个相似度矩阵，通过梯度下降法最小化两个矩阵之间的 KL 散度。

以上几种降维算法 Python 实现代码以及降维效果参考这个 [Demo](https://github.com/borgwang/toys/tree/master/dimensionalty_reduction)

<br>

### Useful Tips from [Machine Learning Yearning](http://www.mlyearning.org/)  

1. Validation/Test Set
- 验证集用于调参，测试集用于验证最后性能。应选择能够反映真实问题（想解决的问题）的数据作为验证集和测试集，保证模型改进的方向是正确的。
- 验证集和测试集应该（尽量）服从相同的分布

2. **Single-number evaluation matric**
- 确定唯一的优化目标（错误率、准确率等），多个目标不利于优化。
- 如果有多个 metric，考虑按照一定权重进行合并。如果是不同性质的指标，无法直接合并，可以考虑为某一个指标设定一个“阈值”（可以接受的范围），在不超过这个阈值的情况下优化另一个指标。（即将 metric 分为 satisficing metric 和 optimized metric，在满足 satisficing metric 的基础上优化 optimized metric）

3. Size of validation set and test set
- 验证集的大小应该足以分辨出想要的性能提升精度（比如 0.1%）
- 测试集的大小在数据较少时可以按照 37 分，数据较多时可以少于 30%

4. Fast loop
- 快速实现想法，（在好的验证集和测试集上）验证，快速改进迭代
- 验证集和测试集也需要不断改进（特别是当发现其分布与真是应用时数据的分布存在较大差异时）

5. Error Analysis
- 以分类为例，花一点时间分析分类错误的样本，确定错误的原因，再决定怎么改进。而不是直接尝试进行改进。
（若准确率为 90%，10% 错误的里面只有 5% 将猫错误分成狗，那么花大量时间优化正确识别猫狗并步划算，最多只有 0.5% 的性能提升）

<br>

### Active Learning  

Active Learning 一种特殊的半监督学习算法，不同与传统的算法，Active learning 是根据当前模型 <u>有选择地</u>获取下一步的训练数据，然后更新模型，如此迭代。

Active Learning 思路就是先拿已有的一小部分数据训练一个不那么精确的模型，然后使用这个模型来选择比较 promising 的数据进行后续训练，然后不断迭代。优点是更少的步数内达到更好的性能，适用于训练数据较少和训练数据噪声较多的情况。

对于已经少量 labeled 数据和大量 unlabeled 数据的情况，传统的做法就是对这些 unlabeled 数据都进行标注(which is very expensive)，并且很可能这些 unlabeled 数据中有一大部分对最后训练的模型没有太大贡献。因此可以使用 Active Learning 的思路，训练一个粗糙的模型选择比较  promising 的样本进行标注。

以 SVM 作为例子，由于 SVM 最后的模型由支持向量决定，我们可以从 unlabeled 数据中挑出比较有可能成为支持向量的数据进行标注，具体来说就是找到离超平面比较近的数据点进行标注。这样可以降低标注成本，同时在更少的步数内达到更好的性能。

<br>

### Learning Theory

#### VC 维  

对于一个二分类问题，给定一个数据集 S 包含 m 个数据点，则总共有 2^m 种可能的分类方案。令 H 表示假设集合（hypothesis class），对于数据集 S 的任意分类方案（2^m），假设空间 H 中至少有一个假设 h 能够正确地对 S 中每个点进行分类，称 H 能够 shatter 数据集 S。

一个假设空间 H 能够 shatter 至多包含 d 个点的数据集 $$ S_d $$ ，则称假设空间 H 的 VC 维为 d。即 VC 维为 d 的假设空间，能对最多包含 d 个数据点的数据集划分出所有可能。

**总结** ：
- VC 维衡量了 <u>假设空间的 capacity（容量），或者 expressive power（表达能力），或者 complexity（复杂程度）</u>；
- 训练好一个 VC 维为 d 的模型所需的最少的训练数据量 m 与模型的数量呈线性关系；

参考 [这篇 blog](http://www.flickering.cn/machine_learning/2015/04/vc%E7%BB%B4%E7%9A%84%E6%9D%A5%E9%BE%99%E5%8E%BB%E8%84%89/)

<br>

#### Empirical Risk and True Risk

Empirical Risk 可以理解为在训练集上的 Error

<!--START formula-->
  <div class="formula">
    $$ R_{emp}(\alpha)=\sum_{x_i,y_i \in D_{train}}L(f(x_i,\alpha), y_i) \to Training\ Error$$
  </div>
<!--END formula-->

True Risk 是在测试集上的 Error

<!--START formula-->
  <div class="formula">
    $$ R(\alpha)=\sum_{x_i,y_i \in D_{test}}L(f(x_i,\alpha), y_i) \to Test\ Error$$
  </div>
<!--END formula-->

True risk 的上界可以由 Empirical risk 加上置信风险项确定

<!--START formula-->
  <div class="formula">
    $$ R(\alpha)\leq R_{emp}(\alpha)+\sqrt{\frac{h(\log(\frac{2m}{h}+1))-\log(\frac{\eta}{4})}{m}} $$
  </div>
<!--END formula-->

训练样本数量 $$ m $$ 越多，置信风险越小；VC 维 $$ h $$ 越大，置信风险越大。  
如果模型 complexity 很高，虽然可以做到 Training error 很小，但是 test error 可能会很大（overfitting）  
如果模型 complexity 很小，模型太简单，那么 training error 不可能达到很小 （underfitting）  

**总结**：  
经验风险就是训练误差，真实风险就是测试误差。经验风险和真实风险是存在一定关系的，由联合界定力和霍夫丁不等式可以推出真实风险和经验风险之间的差值存在一个上界（置信风险），这个上界随着训练数据的增加而降低，随着模型复杂度增大而增加。  

因此可以通过最小化经验风险来最小化真实风险。这也是机器学习算法的基本思路， <u>通过最小化训练集上的误差（经验风险），并加入控制模型复杂度的正则项（置信风险），达到最小化测试集误差（真实风险）的效果</u>。

在经验风险中加入了惩罚（正则）项称为结构化风险，其目的是在保证经验风险尽量小的同事尽量减少过拟合，最小化真实风险。

<br>

### Ensemble Methods

集成方法（Ensemble Methods）是实践中非常受欢迎的一类机器学习算法，主要包括了以随机森林为代表的 Bagging 算法和以 AdaBoost、GBDT 为代表的 Boosting 算法。有些资料也将模型平均方法、 Stacking 归为集成方法。

集成方法从理论上可以推导出下列式子：

<!--START formula-->
  <div class="formula">
    $$ E_{ensemble} = \bar{E}_{base} - \bar{A}_{base}$$
  </div>
<!--END formula-->

> 其中 $$ E_{ensemble} $$ 为集成模型的泛化误差， $$ \bar{E}_{base} $$ 为基学习器的加权泛化误差，$$ \bar{A}_{base} $$ 为基学习器之间的加权差异性

上式的直观理解是：**每个 base learner 的泛化误差越低，base learner 之间的差异性越高，集成方法的泛化性能越好。**

使用集成方法的时候应该对 base learner 的泛化误差和 base learner 之间的多样性两个维度进行分析。分析方法就是将两者在二维平面画出来，平面图中每个数据点 X 轴表示表示两个 base learner 的平均准确率，Y 轴表示两个 base learner 之间的多样性。可以通过观察所有 base learner 整体所处在的位置，处于平均准确率高、多样性大的表示集成方法性能较佳

<!--START figure-->
<div class="figure">
  <a href="https://ws4.sinaimg.cn/large/006tKfTcgy1g1l14y7wxrj31120gsahv.jpg" data-lightbox="ensemble_diversity">
    <img src="https://ws4.sinaimg.cn/large/006tKfTcgy1g1l14y7wxrj31120gsahv.jpg" width="80%" alt="ensemble_diversity" referrerPolicy="no-referrer"/>
  </a>
</div>
<!--END figure-->

多样性度量的指标有很多种：不合度量、相关系数、Q-统计量、κ-统计量等（详见西瓜书 P187），上图横轴即为 κ-度量。

<br>
