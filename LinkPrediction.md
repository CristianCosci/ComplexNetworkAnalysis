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
|<img src="imgs/img1.png" width="100%" height="100%"> | Network representation as a graph. |



We use (`vertex ≡ node`), (`link ≡ edge`) and (`graph ≡ network`) interchangeably. In the graph, a universal set $U$ contains a total of $\frac{n(n−1)}{2}$ links (total node-pairs), where $n = |V|$ represents the number of total vertices of the
graph. <br>
(|U| − |E|) number of links are termed as the *non-existing links*, and some of these links may appear in the near future when we talk about dynamic network. ***Finding such missing links (i.e., AC, BD, and AD) is the aim of link prediction***.

The link prediction problem can be defined as follow: <br>
*Suppose a graph $G_{t_0 − t_1} (V, E)$ represents a snapshot of a network during time interval $[t_0 ,t_1]$ and $E_{t_0 − t_1}$ , a set of links  present in that snapshot. The task of link prediction is to find set of links $E_{t_0' − t_1'}$ during the time interval $[t_0' ,t_1']$ where $[t_0 ,t_1] \leq [t_0' ,t_1']$.*

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
Similarity-based metrics are the simplest one in link prediction, in which for each pair $x$ and $y$, a similarity score $S(x, y)$ is calculated. The score $S(x, y)$ is based on the structural or node’s properties of the considered pair. The non-observed links (i.e., $U − E^T$ ) are assigned scores according to their similarities. **The pair of nodes having a higher score represents the predicted link between them**. The similarity measures between every pair *can be calculated using several properties of the network*, one of which is structural property. Scores based on this property can be grouped in several categories like **local and global**, and so on.

### 2.1.1 **Local similarity indices**
Local indices are generally calculated using information about common neighbors and node degree. These indices **consider immediate neighbors of a node**. The following are some examples of local similarity indices with a description and method to calculate them:
- `Common Neighbors (CN)`: In a given network or graph, the size of common neighbors for a given pair of nodes $x$ and $y$ is calculated as the size of the intersection of the two nodes neighborhoods ($\Gamma$).
    $$S(x, y) = |\Gamma(x) \cap \Gamma(y)|$$
    The likelihood of the existence of a link between x and y increases with the number of common neighbors between them.
- `Jaccard Coefficient`: This metric is similar to the Common Neighbors. Additionally, it normalizes the above score, as given below:
    $$S(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{|\Gamma(x) \cup \Gamma(y)|}$$
    The Jaccard coefficient is defined as the probability of selection of common neighbors of pairwise vertices from all the neighbors of either vertex. The pairwise Jaccard score increases with the number of common neighbors between the two vertices considered. Some researcher (***Liben-Nowell et al.***) demonstrated that this similarity metric **performs worse** as compared to Common Neighbors.
- `Adamic/Adar Index`: Adamic and Adar presented a metric to calculate a similarity score between two web pages based on shared features, which are further used in link prediction after some modification
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{1}{log k_z}$$
    where $k_z$ is the degree of the node $z$. It is clear from the equation that more weights are assigned to the common neighbors having smaller degrees. This is also intuitive in the real-world scenario, for example, a person with more number of friends spend less time/resource with an individual friend as compared to the less number of friends.
- `Preferential Attachment (PA)`: The idea of preferential attachment is applied to generate a growing scale-free network. The term **growing** represents the incremental nature of nodes over time in the network. The likelihood incrementing new connection associated with a node $x$ is proportional to $k_x$ , the degree of the node. Preferential attachment score between two nodes x and y can be computed as:
    $$S(x, y) = k_x k_y$$
    This index shows the worst performance on most networks. The **simplicity**
(as it requires the least information for the score calculation) and the **computational time** of this metric are the main advantages. PA shows better results if larger degree nodes are densely connected, and lower degree nodes are rarely connected. In the above equation, summation can also be used instead of multiplication as an aggregate function.
- `Resource Allocation Index (RA)`
- `Cosine similarity or Salton Index (SI)`

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

