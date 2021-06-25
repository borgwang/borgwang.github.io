---
layout: post
title: Similarity and Distance Measurements
date: 2016-05-13
categories: ML
---

<br>

## Abstract

总结一下机器学习中常用的距离度量、相似度度量，以及两者之间的联系。

<br>

## Euclidean Space Distance

#### Euclidean distance

<!--START formula-->
  <div class="formula">
    $$ d(u,v)=\|u-v\|^2=\sqrt{\sum_{i=1}^{n}(u_i-v_i)^2} $$
  </div>
<!--END formula-->

#### Manhattan distance

<!--START formula-->
  <div class="formula">
    $$ d(u,v)=\|u-v\|^1=\sum_{i=1}^{n}|u_i-v_i| $$
  </div>
<!--END formula-->

#### Minkowski distance

<!--START formula-->
  <div class="formula">
    $$ d(u,v)=\|u-v\|^p= \sqrt[p]{\sum_{i=1}^{n}(u_i-v_i)^p}$$
  </div>
<!--END formula-->

#### Chebyshev distance

<!--START formula-->
  <div class="formula">
    $$ d(u,v)=\|u-v\|^\infty=\lim_{p \to \infty}\sqrt[p]{\sum_{i=1}^{n}(u_i-v_i)^p}=\max_i |u_i-v_i|$$
  </div>
<!--END formula-->

#### Hamming distance
The Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different.

#### Mahalanobis Distance
马氏距离基本思路就是对原数据进行坐标旋转，使得旋转后各个维度尽量线性无关；再进行缩放，使得各个维度经过缩放后方差都为 1；最后计算经过变换后的数据的欧式距离即为马氏距离。

<br>

## Similarity

#### Cosine similarity

<!--START formula-->
  <div class="formula">
    $$ S(u,v)=\frac{\sum_{i=1}^{n}u_i v_i}{\sqrt{\sum_{i-1}^{n}u_i^2}\sqrt{\sum_{i-1}^{n}v_i^2}} $$
  </div>
<!--END formula-->

#### Pearson Correlation Coefficient

<!--START formula-->
  <div class="formula">
    $$ S(u,v)=\frac{\sum_{i=1}^{n}(u_i-\overline{u})(v_i-\overline{v})}{\sqrt{\sum_{i=1}^{n}(u_i-\overline{u})^2}\sqrt{\sum_{i=1}^{n}(u_i-\overline{u})^2}} $$
  </div>
<!--END formula-->

#### Jaccard similarity coefficient

<!--START formula-->
  <div class="formula">
    $$ S(U,V)=\frac{|U\cap V|}{|U\cup V|} $$
  </div>
<!--END formula-->

#### KL-divergence
KL divergence is a measure of how one probability distribution diverges from a second expected probability distribution.

<!--START formula-->
  <div class="formula">
    $$ D_{KL}(p,q)=-\sum_{x}p(x)\log q(x)+\sum_{x}p(x)\log p(x) $$
  </div>
<!--END formula-->

Note that KL-divergence is not symmetry, i.e

<!--START formula-->
  <div class="formula">
    $$ D_{KL}(p,q)\neq D_{KL}(p,q) $$
  </div>
<!--END formula-->

We can define

<!--START formula-->
  <div class="formula">
    $$ \bar{D_{KL}(p,q)}=\frac{1}{2}D_{KL}(p,q)+\frac{1}{2}D_{KL}(q,p) $$
  </div>
<!--END formula-->

so that it satisfies symmetry.

<br>

## Note

<u>Distance is lack of similarity and similarity is resemblance</u>. Some authors prefer to use the term ‘dissimilarity’ instead of distance.

Distance satisfy three conditions: reflexivity, symmetry, and
triangular inequality.
(Consider three points a, b, and c describing a triangle in a 2D-space)
- reflexivity

<!--START formula-->
  <div class="formula">
    $$ D(a,a)=D(b,b)=D(c,c)=0 $$
  </div>
<!--END formula-->

- symmetry

<!--START formula-->
  <div class="formula">
    $$ D(a,b)=D(b,a) $$
  </div>
<!--END formula-->

- triangular inequality

<!--START formula-->
  <div class="formula">
    $$ D(a,b)+D(b,c)\leq D(a,c) $$
  </div>
<!--END formula-->

Similarity is a measure of the resemblance between data sets.
Similarity only satisfies symmetry condition. The similarity of a vector to itself is 1, $$ S(a, a)=1 $$. Similarity can be negative while Distance only adopts non-negative values.
We can arithmetically average, add, or subtract
distances to compute new distances, but we cannot do the same with similarities.

Similarity can be transform to distance metric using following tricks:

<!--START formula-->
  <div class="formula">
    $$ D(u,v)=1-S(u,v) $$
    $$ D(u,v)=\sqrt{k(1-S(u,v))} $$
    $$ D(u,v)=-\ln{S(u,v)} $$
  </div>
<!--END formula-->

References: [this tutorial](http://www.minerazzi.com/tutorials/distance-similarity-tutorial.pdf)

<br><br>
