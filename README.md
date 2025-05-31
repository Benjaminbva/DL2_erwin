# GATrErwin: A Geometric Algebra-Inspired Equivariant Hierarchical Transformer

## Overview
- **Goal**: Integrate Geometric Algebra Transformer (GATr) into the Erwin hierarchical transformer to enforce full E(3)-equivariance on 3D point clouds.
- **Key idea**: Replace Erwin’s non-equivariant ball-tree attention and pooling/unpooling architecture with GATr’s equivariant layers.
- **Applications**: Physical simulations, cosmological velocity prediction, and any domain where respecting rotations/translations/reflections improves robustness.

---

## Introduction & Related Work
1. **Erwin summary**  
   - Erwin (Zhdanov et al., 2025), is a hierarchical transformer inspired by the field of computational many-body physics. It combines the efficiency of tree-based algorithms with the expressivity of attention mechanisms. Erwin organizes irregular 3D point clouds into binary ball trees, where each ball (node) covers a subset of points, and subsequently computes multi-head self-attention independently within each ball. Key aspects within Erwin's architecture include:
   1) Hierarchical coarsening and refinement: after each attention block, Erwin coarsens the tree by pooling features up the ball-tree hierarchy, and then refines features back down through the encoder, enabling the capture of both fine-grained local details and global features.
   2) Cross-ball interaction: attention is calculated within disjoint balls. To allow for interaction between points in different balls, Erwin alternates between the orginal tree and one built on the rotated copy of the 3D point cloud.
   3) Linear-time attention: by restricting attention to local fixed-size neighbourhoods, the models compexity is reduced from quadratic to linear in the number of points.
   In the evaluation on three large-scale physical domain Erwin was able to consistently outperform other baselines, achieving state-of-the-art performance in both computational efficiency and prediction accuracy. 
2. **Related work**  
   - Achieving sub-quadratic attention on large inputs has been studied extensively. On regular grid data, the SwinTransformer (BRON) reduces complexity by restricting self-attention to non-overlapping local windows and connecting them via shifted windows. Other methods, like (Liu et al., 2023; Sun et al., 2022 ;PointTransformer v3(bron)) focus on inducing structures that allow for patching. However transforming point clouds into sequences, while scalable, can introduce artificial discontinuities that harm local relationships (Zhdanov et al., 2025).  In 1D, the hierarchical attention methods H-Transformer (Zhu & Soricut, 2021) and Fast Multipole Attention (Kang et al., 2023) avoid quadratic cost by computing full-scale atention localy and learning distant relationships through coarsening. For 3D point clouds, OctFormer (Wang, 2023) builds an octree so that spatially adjacent points are consecutive in memory, but relies on costly octree convolutions. Other methods, like (Janny et al., 2023; Alkin et al., 2024a), avoid the use of hierarchical attention, by proposing to instead use cluster attention. By applying attention to clustered points, and scattering features back, they trade off efficiency for potential sacrifce of more fine-grained details, posing scalability problems. 

---

## Strengths, Weaknesses of Erwin & Motivation for GATrErwin
- **Strengths of Erwin**  
  - Erwin is able to perform linear-time attention on irregular 3D point clouds via ball tree. This makes it a highly effective model across diverse physical domains. It is able to handle large input data, while keeping runtime low, making it an a suitable candidate for any tasks that require modeling large particle systems. 
- **Weaknesses**  
  - Models that respect physical laws are of increasing interest in scientific data analysis [Suk et al., 2025]. Equivariant neural networks implement this principle by embedding known symmetries directly into their architecture, freeing the model from having to learn geometric information and thereby improving generalization, robustness, and sample efficiency. While Erwin is able to model large particle systems at state-of-the-art level, Erwin is neither rotation nor refletion quivariant. As a result, performance drops under E(3) transformations of the data.
- **Why GATrErwin?**  
  - Erwin is already performing at a high standard level. However, enforcing equivariance can lead to better generalization, robustness, sample efficiency. Motivated by ViNE-GATr [Suk et al., 2025], which combines Perceiver-style virtual nodes with a Geometric Algebra Transformer (GATr) [Brehmer et al., 2023] to achieve linear complexity while preserving full E(3) equivariance, we propose extending Erwin with the same geometric attention mechanism. Leveraging geometric algebra to make the model fully E(3) symmetric can take the already highly effective model to even higher levels.   
  
---

## Where does Erwin Break equivariance?
Erwin breaks equivariance at several points in it's architecture. (1) the tree’s split axes, its padding scheme, and the bijection $\pi$ in the ball tree construction all depend on the point cloud's absolute orientation and on fixed conventions, so rotations or arbitrary permutations produce different tree topologies and leaf orderings. (2) While the local encoding in the MPNN is permutation‐equivariant, it does not guarantee invariance under continuous Euclidean transformations. (3) The learnable matrices $\mathbf W_q,\mathbf W_k,\mathbf W_v, \mathbf W_c,\mathbf W_r$ in the ball attention and the coarsening/refinement, together with SwiGLU MLP, have no built-in symmetry constraints and the attention bias $\mathcal B$ and the positional encoding rely on absolute positions. As a result, Erwin does not preserve SE(3) or full E(3) equivariance.

---

## Our Main Contribution
GATrErwin is not the first model to add the equivariance requirement. The SE(3)-Transformer [Fuchs et al., 2020] introduced attention equivariance to rigid motions in3D, and LieConv [Finzi et al., 2020] provided a continuous-group convolutional alternative Finziet al. [2020]. More recent models like Equiformer [Liao and Smidt, 2022] and E2PN [Zhu et al.,2023] demonstrated that full SE(3) symmetry can match non-equivariant baselines on molecular and point-cloud benchmarks. This performance was retrained in SE(3)-Hyena [Moskalev et al.,2024] while leveraging sub-quadratic long-range convolutions. Qu and Krishnapriyan [2024] show that a semi-equivariant model yields minor accuracy loss, a strategy AlphaFold 3 [Abramson et al.,2024] adopts to accelerate model predictions. However, Perin and Deny [2024] report that learned symmetries degrade under sparse orbit coverage, emphasizing the importance of explicit equivariant architectures for data involving physical laws. Tackling these challenges, we introduce GATrErwin, which integrates GATr in the hierarchical Erwin transformer to enforce E(3) equivariance, thereby capturing intrinsic symmetries and yielding more robust, accurate, and generalizable models forlarge-scale physical systems. Specifically, GATrErwin makes the following changes to the original Erwin architecture, tackeling 3 of the 4 equivariance breaking parts of the original model:
- **Equivariant Ball-Tree Attention**  
  - GATrErwin swaps Erwin’s standard attention to the GATr E(3)-equivariant multivector attention.
- **Equivariant Pooling & Unpooling**  
  - The model substitutes Erwin’s BallPooling and BallUnPooling linear layers and batch norms with GATr’s equivariant linearlayer and equivariant layer normalization.
- **Enhanced InvMPNN**  
  - We were able to use GATr to implement an invariant MPNN, which we further enhance by introducing trainable Besselstyleradial basis functions (RBFs) with smooth cosine cutoffs to decompose 

---

## Experiments & Ablation

The experiments in GATrErwin were run on the Cosmology dataset. To run/replicate experiments, you will need to download:  
- [Cosmology Dataset (7 GB)](https://zenodo.org/records/11479419)

| Experiment                                              | Paper Table | Code                                                           | Command                                                                                             |
|---------------------------------------------------------|:-----------:|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Cosmology Velocity Prediction (GATrErwin)**           |    Table 4  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --epochs 100<br>    ``` |
| **Ablation: Relative distances as translations**        |    Table 2  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --use-relative-distances<br>    ``` |
| **Ablation: InvMPNN message-passing**                   |    Table 2  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --invmpnn<br>    ``` |
| **Ablation: InvMPNN as auxiliary scalar**               |    Table 2  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --aux-scalar<br>    ``` |
| **Ablation: Scaled GATrErwin**                          |    Table 2  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --scaled<br>    ``` |
| **Ablation: RADMPNN (RBF-enhanced MPNN)**               |    Table 2  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --radmpnn<br>    ``` |
| **Ablation: Rotating-Tree cross-ball connections**       |    Table 2  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --rotating-tree<br>    ``` |
| **RBF-dimensionality study**                            |    Table 3  | [`experiments/train_cosmology.py`](./experiments/train_cosmology.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/train_cosmology.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --rbf-dim 128<br>    ``` |
| **Equivariance evaluation under rotations & translations** |    Table 4  | [`experiments/evaluate_equivariance.py`](./experiments/evaluate_equivariance.py) | &nbsp;&nbsp;&nbsp;&nbsp;```bash<br>    python experiments/evaluate_equivariance.py \\<br>      --model GATrErwin \\<br>      --dataset cosmology \\<br>      --rotate-angles 15 45 145 \\<br>      --translate 50 150 300<br>    ``` |

---

## Results & Notebooks

---

## Conclusion
Summarize your findings:
- Enforcing E(3) equivariance yields lower MSE on standard and perturbed test sets.
- Trade-offs: compute & params vs. robustness.
- Insights on geometric priors and potential for future work (e.g., adaptive equivariance breaking, larger scale).


## Strengths, Weaknesses of GATrErwin
- **Strengths**  
  - GATrErwin outperforms other state-of-the-art equivariant models like SEGNN and NequiLP in MSE on the Cosmology dataset. 
- **Weaknesses**  
  - Due to the increased dimensionality of adding Cliford Algebra the parameter count increases significantly, leading to large increases in run-time. This was expected, but seeing as a key benefit of Erwin is speed, it is not ideal. 
  - The inclusion of rigid equivariant constraints of the GATr architecture the model is less flexible in adapting to scenario's where full equivariance is not guaranteed. for example, when inductive bias is violated. 
  - 
---

## Contributions
- **Benjamin Van Altena**: 
- **Fleur Dolmans**:
- **Emanuele Arcelli**: 
- **Matthijs Vork**: 

---

## References
1. Zhdanov et al., *Erwin: A tree-based hierarchical transformer for large-scale physical systems*, 2025.  
2. Brehmer et al., *Geometric Algebra Transformer*, NeurIPS 2023.  
3. Suk et al., *ViNE-GATr: Scaling geometric algebra transformers with virtual-node embeddings*, ICLR 2025.  
4. Fuchs et al., *SE(3)-Transformers*, NeurIPS 2020.  
5. … _(add others as needed)_

---
