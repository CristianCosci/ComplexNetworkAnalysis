# **Link Prediction** :mag:
***Definition:*** :key:<br>
Link prediction finds missing links (in static networks) or predicts the likelihood of future links (in dynamic networks).

There exists a wide range of link prediction techniques like similarity-based indices, probabilistic methods, dimensionality
reduction approaches, etc.

## **Index**
1. [**Introduction**](#1-introduction-and-background)
2. [**Methods description**](#2-existing-methods-📑)


# 1. **Introduction and Background**
A **social network** (a more general term is a complex network) is a standard approach to model communication in a group or community of persons. Such networks can be represented as a graphical model in which a node maps to a person or social entity, and a link corresponds to an association or collaboration between corresponding persons or social entities. When links can be deleted or added, during time, the network is called **dynamic**. Lots of issues arise when we study a social network,
some of which are changing association patterns over time, factors that drive those associations, and the effects of those associations to other nodes. Here, we address a specific problem termed as link prediction.

#### **Problem Characterization**
Consider a simple undirected network $G(V, E)$ (Refer to the Figure 1), where $V$ characterizes a vertex-set and $E$, the link-set.

|   |   |
| - | - |
|<img src="imgs/img1.png" width="70%" height="70%"> | Network representation as a graph. |



We use (`vertex ≡ node`), (`link ≡ edge`) and (`graph ≡ network`) interchangeably. In the graph, a universal set $U$ contains a total of $\frac{n(n−1)}{2}$ links (total node-pairs), where $n = |V|$ represents the number of total vertices of the
graph. <br>
(|U| − |E|) number of links are termed as the *non-existing links*, and some of these links may appear in the near future when we talk about dynamic network. ***Finding such missing links (i.e., AC, BD, and AD) is the aim of link prediction***.

The link prediction problem can be defined as follow:<br>
*Suppose a graph* $\mathit{G\_{t\_0 − t\_1} (V, E)}$ *represents a snapshot of a network during time interval* $\mathit{[t_0 ,t_1]}$ *and* $\mathit{E_{t_0 − t_1}}$ *, a set of links  present in that snapshot. The task of link prediction is to find set of links* $\mathit{E_{t_0' − t_1'}}$ *during the time interval* $\mathit{[t_0' ,t_1']}$ *where* $\mathit{[t_0 ,t_1] \leq [t_0' ,t_1']}$ *.*

The link prediction idea is useful in several domains of application. Examples include automatic hyperlink creation, website hyper-link prediction in the Internet
and web science domain, and friend recommendation on Facebook.

# 2. **Existing methods** :bookmark_tabs:
Recently, numerous methodologies of link prediction have been implemented. These methods can be grouped into several categories, like **similarity-based, probabilistic models, learning-based models**, etc.

### ***Sub-Index*** :open_file_folder:
#### 2.1 [**Similarity-based methods**](#21-similarity-based-methods) <br>
&nbsp;&nbsp;2.1.1 [**Local similarity indices**](#211-local-similarity-indices)<br>
&nbsp;&nbsp;2.1.2 [**Global similarity indices**](#212-global-similarity-indices)<br>
&nbsp;&nbsp;2.1.3 [**Quasi-local Indices**](#213-quasi-local-indices)<br>
#### 2.2 [**Probabilistic and maximum likelihood models**](#22-probabilistic-and-maximum-likelihood-models) <br>
&nbsp;&nbsp;2.2.1 [**Local probabilistic model for link prediction**](#221-local-probabilistic-model-for-link-prediction)<br>
&nbsp;&nbsp;2.2.2 [**Probabilistic relational model for link prediction (PRM)**](#222-probabilistic-relational-model-for-link-prediction-prm)<br>
&nbsp;&nbsp;2.2.3 [**Hierarchical structure model (HSM)**](#223-hierarchical-structure-model-hsm)<br>
&nbsp;&nbsp;2.2.4 [**Stochastic block model (SBM)**](#224-stochastic-block-model-sbm)<br>
&nbsp;&nbsp;2.2.5 [**Exponential random graph model (ERGM) or P-star model**](#225-exponential-random-graph-model-ergm-or-p-star-model)<br>
#### 2.3 [**Link prediction using dimensionality reduction**](#23-link-prediction-using-dimensionality-reduction) <br>
&nbsp;&nbsp;2.3.1 [**Embedding-based link prediction**](#231-embedding-based-link-prediction)<br>
&nbsp;&nbsp;2.3.2 [**Matrix factorization/decomposition-based link prediction**](#232-matrix-factorizationdecomposition-based-link-prediction)<br>
#### 2.4 [**Other approaches**](#24-other-approaches) <br>
&nbsp;&nbsp;2.4.1 [**Learning-based frameworks for link prediction**](#241-learning-based-frameworks-for-link-prediction)<br>
&nbsp;&nbsp;2.4.2 [**Information theory-based link prediction**](#242-information-theory-based-link-prediction)<br>
&nbsp;&nbsp;2.4.3 [**Clustering-based Link Prediction**](#243-clustering-based-link-prediction)<br>
&nbsp;&nbsp;2.4.4 [**Structural Perturbation Method**](#244-structural-perturbation-method)<br>

<img src="imgs/img2.jpg" width="60%" height="60%"> 

## 2.1 **Similarity-based methods**
### 2.1.1 **Local similarity indices**
### 2.1.2 **Global similarity indices**
### 2.1.3 **Quasi-local Indices**

## 2.2 **Probabilistic and maximum likelihood models**
### 2.2.1 **Local probabilistic model for link prediction**
### 2.2.2 **Probabilistic relational model for link prediction (PRM)**
### 2.2.3 **Hierarchical structure model (HSM)**
### 2.2.4 **Stochastic block model (SBM)**
### 2.2.5 **Exponential random graph model (ERGM) or P-star model**

## 2.3 **Link prediction using dimensionality reduction**
### 2.3.1 **Embedding-based link prediction**
### 2.3.2 **Matrix factorization/decomposition-based link prediction**

## 2.4 **Other approaches**
### 2.4.1 **Learning-based frameworks for link prediction**
### 2.4.2 **Information theory-based link prediction**
### 2.4.3 **Clustering-based Link Prediction**
### 2.4.4 **Structural Perturbation Method**

