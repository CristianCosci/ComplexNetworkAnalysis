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
Similarity-based metrics are the simplest one in link prediction, in which for each pair $x$ and $y$, a similarity score $S(x, y)$ is calculated. The score $S(x, y)$ is based on the structural or node’s properties of the considered pair. The non-observed links (i.e., $U − E^T$ ) are assigned scores according to their similarities. **The pair of nodes having a higher score represents the predicted link between them**. The similarity measures between every pair *can be calculated using several properties of the network*, one of which is structural property. Scores based on this property can be grouped in several categories like **local and global**, and so on.

### 2.1.1 **Local similarity indices**
Local indices are generally calculated using information about common neighbors and node degree. These indices **consider immediate neighbors of a node**. The following are some examples of local similarity indices with a description and method to calculate them:
- `Common Neighbors (CN)`: In a given network or graph, the size of common neighbors for a given pair of nodes $x$ and $y$ is calculated as the size of the intersection of the two nodes neighborhoods ($\Gamma$).
    $$S(x, y) = |\Gamma(x) \cap \Gamma(y)|$$
    The likelihood of the existence of a link between x and y increases with the number of common neighbors between them.
- `Jaccard Coefficient`: This metric is similar to the Common Neighbors. Additionally, it normalizes the above score, as given below:
    $$S(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{|\Gamma(x) \cup \Gamma(y)|}$$
    The Jaccard coefficient is defined as the probability of selection of common neighbors of pairwise vertices from all the neighbors of either vertex. The pairwise Jaccard score increases with the number of common neighbors between the two vertices considered. Some researcher (***Liben-Nowell et al.***) demonstrated that this similarity metric **performs worse** as compared to Common Neighbors.
- `Adamic/Adar Index (AA)`: Adamic and Adar presented a metric to calculate a similarity score between two web pages based on shared features, which are further used in link prediction after some modification
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{1}{log k_z}$$
    where $k_z$ is the degree of the node $z$. It is clear from the equation that more weights are assigned to the common neighbors having smaller degrees. This is also intuitive in the real-world scenario, for example, a person with more number of friends spend less time/resource with an individual friend as compared to the less number of friends.
- `Preferential Attachment (PA)`: The idea of preferential attachment is applied to generate a growing scale-free network. The term **growing** represents the incremental nature of nodes over time in the network. The likelihood incrementing new connection associated with a node $x$ is proportional to $k_x$ , the degree of the node. Preferential attachment score between two nodes x and y can be computed as:
    $$S(x, y) = k_x k_y$$
    This index shows the worst performance on most networks. The **simplicity**
(as it requires the least information for the score calculation) and the **computational time** of this metric are the main advantages. PA shows better results if larger degree nodes are densely connected, and lower degree nodes are rarely connected. In the above equation, summation can also be used instead of multiplication as an aggregate function.
- `Resource Allocation Index (RA)`: Consider two non-adjacent vertices $x$ and $y$. Suppose node $x$ sends some resources to $y$ through the common nodes of both $x$ and $y$ then the similarity between the two vertices is computed in terms of **resources sent** from $x$ to $y$. This is expressed mathematically as:
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{1}{k_z}$$
    The difference between **RA** and **AA** is that the RA index heavily penalizes to higher degree nodes compared to the AA index. Prediction results of these indices become almost the same for smaller average degree networks. This index
shows good performance on heterogeneous networks with a high clustering coefficient, especially on transportation networks.
- `Cosine similarity or Salton Index (SI)`: This similarity index between two records (documents) is measured by calculating the Cosine of the angle between them. The metric is all about the orientation and not magnitude. The Cosine similarity can be computed as
    $$S(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{\sqrt(k_x k_y)}$$
- `Sorensen Index`: It is very similar to the Jaccard index. **McCune et al.** show that it **is more robust than Jaccard against the outliers**.
    $$S(x, y) = \frac{2|\Gamma(x) \cap \Gamma(y)|}{k_X + k_y}$$
- `CAR-based Common Neighbor Index (CAR)`: CAR-based indices are presented based on the assumption that the link existence between two nodes is more likely if their common neighbors are members of a local community (local-community-paradigm (LCP) theory). In other words, the likelihood existence increases with the number of links among the common neighbors (local community links (LCLs)) of the seed node pair as described in the following figure.
    $$S(x, y) = CN(x, y) \text{ x } LCL(x, y) = CN(x, y) \text{ x } \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{|\gamma(z)|}{2} $$
    where $CN(x, y) = |\Gamma(x) ∩ \Gamma(y)|$ is number of common neighbors. $LCL(x, y)$ refers to the number of local community links which are defined as the links among the common neighbors of seed nodes x and y. $\gamma(z)$ is the subset of neighbors of node $z$ that are also common neighbors of $x$ and $y$. <br>
    <img src="imgs/img3.png" width="70%" height="70%"> 
- `CAR-based Adamic/Adar Index (CAA)`: If $LCL$ is considered as an accuracy enhancer, then the $CAA$ index is obtained by incorporating the $LCL$ theory to the well known AA index and mathematically expressed by the equation given below.
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{|\gamma(z)|}{\log_2(k_z)} $$
- `CAR-based Resource Allocation Index (CRA)`: Is a general application of the LCL theory to other indices and generate the CRA index by incorporating this concept into the existing RA index of the literature. Mathematically, the CRA can be expressed as
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{|\gamma(z)|}{k_z}$$
- `CAR-based Preferential Attachment Index (CPA)`: This is the preferential attachment index based on the CAR index. CPA is obtained by incorporating the LCL theory to the original PA method and expressed mathematically by
    $$S(x, y) = e_x e_y + e_x CAR(x, y) + e_y CAR(x, y) + CAR(x, y)^2$$
    where $e_x$ is the number of neighbors of $x$ not shared by $y$ and $CAR(x, y)$ is the similarity score of the node pair $x$ and $y$ using CAR index. <br>
    CAR-based methods listed above show the best performance on LCP networks. The LCP networks are related to dynamic and heterogeneous systems and facilitate network evolution of social and biological networks.
- `Hub Promoted Index (HPI)`: This similarity index promotes the formation of links between the sparsely connected nodes and hubs. It also tries to prevent links formation between the hub nodes. This similarity metric can be expressed mathematically as
    $$S(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{min(k_x, k_y)}$$
- `Hub Depressed Index (HDI)`: This index is the same as the previous one but with the opposite goal as it avoids the formation of links between hubs and low degree nodes in the networks. The Hub depressed index promotes the links evolution between the hubs as well as the low degree nodes. The mathematical expression for this index is given below.
    $$S(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{max(k_x, k_y)}$$
- `Local Naive Bayes-based Common Neighbors (LNBCN)`: The above similarity indices are somehow based on common neighbors of the node pair where each of the which are equally weighted. This method is based on the Naive Bayes theory and arguments that different common neighbors play different role in the network and hence contributes differently to the score function computed for non-observed node pairs
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} [log(\frac{C(z)}{1 - C(z)}) + log(\frac{1 - \rho}{\rho})]$$
    where $C(z)$ is node clustering coefficient and $\rho$ is the network density expressed as
    $$\rho = \frac{|E|}{n(n-1)/2}$$
- `Leicht-Holme-Newman Local Index (LHNL)`: The logic below this index is that two vertices are similar to each other if their corresponding neighbors are self-similar to themselves. This score is defined by the ratio of the path of length two that exits between two vertices and the expected path of the same length between them.
    $$S(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{k_x k_y}$$
- `Node Clustering Coefficient (CCLP)`: This index is also based on the clustering coefficient property of the network in which the clustering coefficients of all the common neighbors of a seed node pair are computed and summed to find the final similarity score of the pair. Mathematically
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} C(z)$$
    where
    $$C(z) = \frac{t(z)}{k_z(k_z - 1)}$$
    is clustering coefficient of the node $z$ and $t(z)$ is the total triangles passing through the node $z$.
- `Node and Link Clustering coefficient (NLC)`: This similarity index is based on the basic topological feature of a network called ”*Clustering Coefficient*”. The
clustering coefficients of both nodes and links are
incorporated to compute the similarity score.
    $$S(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{|\Gamma(x) \cap \Gamma(z)|}{k_z -1} \text{ x }C(z) + \frac{|\Gamma(y) \cap \Gamma(z)|}{k_z -1} \text{ x }C(z)$$
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

