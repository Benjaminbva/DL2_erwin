# GATrErwin: A Geometric Algebra-Inspired Equivariant Hierarchical Transformer
- **Authors**  
    - Benjamin Van Altena, Fleur Dolmans, Emanuele Arcelli, Matthijs Vork 
- **Paper**  
[GATrErwin: A Geometric Algebra-Inspired Equivariant Hierarchical Transformer](...) 

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
   - Achieving sub-quadratic attention on large inputs has been studied extensively. Erwin (Zhdanov et al., 2025) discusses such related work: on regular grid data, the SwinTransformer (Liu et al., 2021) reduces complexity by restricting self-attention to non-overlapping local windows and connecting them via shifted windows. Other methods, like (Liu et al., 2023; Sun et al., 2022 ; Wuet al., 2024) focus on inducing structures that allow for patching. However transforming point clouds into sequences, while scalable, can introduce artificial discontinuities that harm local relationships (Zhdanov et al., 2025).  In 1D, the hierarchical attention methods H-Transformer (Zhu & Soricut, 2021) and Fast Multipole Attention (Kang et al., 2023) avoid quadratic cost by computing full-scale atention localy and learning distant relationships through coarsening. For 3D point clouds, OctFormer (Wang, 2023) builds an octree so that spatially adjacent points are consecutive in memory, but relies on costly octree convolutions. Other methods, like (Janny et al., 2023; Alkin et al., 2024), avoid the use of hierarchical attention, by proposing to instead use cluster attention. By applying attention to clustered points, and scattering features back, they trade off efficiency for potential sacrifce of more fine-grained details, posing scalability problems. 

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
  - We were able to use GATr to implement an invariant MPNN, which we further enhance by introducing trainable Besselstyleradial basis functions (RBFs) with smooth cosine cutoffs to decompose. 

---

## Experiments 

The experiments in GATrErwin were run on the Cosmology dataset. To run/replicate experiments, you will need to download:  
- [Cosmology Dataset (7 GB)](https://zenodo.org/records/11479419)

## Results & Notebooks
Below are our main quantative results. These, and other results shown in the appendix of our paper, can be reproduced following the steps and explanation found in [Reproduction Notebook](https://github.com/Benjaminbva/DL2_erwin/blob/main/The%20notebook.ipynb).
### 1 Invariance Experiment (Untrained Model)

| Component        | Avg IE   |
|------------------|----------|
| Erwin            | 0.7266   |
| + Fixed tree     | 0.002    |
| + Inv MPNN       | 0.0024   |
| + Inv Eq. 1      | 0.0025   |
| + Inv Coarsening 2 | 0.0025  |
| + Inv Refinement 2 | 1.4246e-6 |
| + Wrap           | 0.1739   |

> *Table 1a: Average invariance Error (IE) over several rotation angles on a newly initialized (untrained) model as we incrementally fix invariant breaking components by incorporating distances between points instead of relative positions.*

### 2 Invariance Experiment (Trained Model)

| Model          | MSE   | IE @ 15°  | IE @ 45°  | IE @ 90°  | IE @ 160°  |
|----------------|-------|-----------|----------|----------|-----------|
| Erwin          | 0.609  | 0.502    | 1.03    | 1.02    | 1.50      |
| + Inv MPNN     | 0.625  | 0.606    | 1.17    | 1.02    | 1.55      |
| + Inv Eq. 1    | 0.639  | 0.668    | 1.16    | 0.961  | 1.44      |
| + Inv Coarsening 2 | 0.627  | 0.640    | 1.33    | 1.02    | 1.59      |
| + Inv Refinement 2 | 0.621  | 0.610    | 1.19    | 1.02    | 1.58      |

> *Table 1b: Mean Squared Error (MSE) and Invariance Error (IE) at 15°, 45°, 90°, 160° on trained models as we incrementally fix invariant breaking components by incorporating distances between points instead of relative positions.*

### 3 Equivariance Experiment

| Model           | MSE  | 15°   | 45°   | 90°   | 160°  |
|-----------------|------|------|------|------|------|
| Erwin+          | 0.609 | 0.668 | 1.16  | 0.961 | 1.44  |
| GATrErwin       | 0.672 | 0.640 | 1.33  | 1.02  | 1.59  |
| GATrErwin+      | 0.699 | 0.610 | 1.19  | 1.02  | 1.58  |

> *Table 2: MSE and IE at various rotation angles for Erwin vs. GATrErwin, with (“+”) and without rotated‐tree augmentation. The “+” versions use a rotated ball‐tree. Results are reported on the original dataset D (MSE) and under rotated variations of the dataset D<sub>θ</sub>.*

---

## Conclusion
- **Strengths**
    - GATrErwin successfully embeds E(3)-equivariant Geometric Algebra Transformer (GATr) modules into the hierarchical Erwin architecture, replacing ball‐tree attention, coarsening and refinement projections, and MPNN steps with multivector operations in projective geometric algebra. On the Cosmology velocity‐prediction benchmark, GATrErwin achieves the lowest MSE of all evaluated equivariant models
    - Raw inter-particle distances can possibly span many orders of magnitude, possibly causing unstable gradients when they are fed directly into a neural network, leading to unstable training in Erwin. GATrErwin adresses this problem by introducing radial basis functions. Through the projection of each distance onto a set of orthogonal radial basis functions, thereby transforming any distance into a fixed-length vector whose components stay within a controlled range. With this addition the training of GATrErwin is more stabe and converges reliably. 
- **Weaknesses**
    - By enforcing full E(3) equivariance with the introduction of GATr, GATrErwin has about ten times more parameters and runs roughly ten times slower per training epoch compared to Erwin. Specifically, in cases where inference speed is the priority, the simpler Erwin model more convenient. However, the extra cost can be justified when datasets are small or when robustness to arbitrary orientations is critical.
    - On the rotated test set, Erwin achieves a lower MSE than GATrErwin despite both models using the same non-equivariant ball tree construction, indicating that GATrErwin's rigid equivariant layers increase its vulnerability to rotational perturbations.
- **Overall**
    - In summary, GATrErwin represents a trade-off: it achieves state‐of‐the‐art equivariant performance on large‐scale 3D point clouds but at the cost of runtime and model size. Future work should focus on replacing the axis-aligned tree construction with one which doesn't break equivariance.   

## Contributions
- **Benjamin Van Altena**: 
- **Fleur Dolmans**:
- **Emanuele Arcelli**: 
- **Matthijs Vork**: 

---

## References

1. Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al.  
   **Accurate structure prediction of biomolecular interactions with AlphaFold 3.**  
   *Nature, 630(8016):493–500, 2024.*

2. Alkin, B., Fürst, A., Schmid, S., Gruber, L., Holzleitner, M., and Brandstetter, J.  
   **Universal physics transformers: A framework for efficiently scaling neural operators.**  
   In *Proceedings of NeurIPS, 2024.*

3. Brehmer, J., De Haan, P., Behrends, S., and Cohen, T. S.  
   **Geometric algebra transformer.**  
   In *Advances in Neural Information Processing Systems*, vol. 36, pages 35472–35496, 2023.

4. Finzi, M., Stanton, S., Izmailov, P., and Wilson, A. G.  
   **Generalizing convolutional neural networks for equivariance to Lie groups on arbitrary continuous data.**  
   In *Proceedings of ICML*, pages 3165–3176. PMLR, 2020.

5. Fuchs, F., Worrall, D., Fischer, V., and Welling, M.  
   **SE(3)-Transformers: 3D roto-translation equivariant attention networks.**  
   In *Advances in Neural Information Processing Systems*, vol. 33, pages 1970–1981, 2020.

6. Janny, S., Béneteau, A., Nadri, M., Digne, J., Thome, N., and Wolf, C.  
   **EAGLE: Large-scale learning of turbulent fluid dynamics with mesh transformers.**  
   In *ICLR Workshops, 2023.*

7. Kang, Y., Tran, G., and Sterck, H. D.  
   **Fast multipole attention: A divide-and-conquer attention mechanism for long sequences.**  
   *arXiv:2310.11960, 2023.*

8. Liao, Y.-L. and Smidt, T.  
   **Equiformer: Equivariant graph attention transformer for 3D atomistic graphs.**  
   *arXiv:2206.11990, 2022.*

9. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B.  
   **Swin Transformer: Hierarchical vision transformer using shifted windows.**  
   In *Proceedings of ICCV, 2021.*

10. Liu, Z., Yang, X., Tang, H., Yang, S., and Han, S.  
    **Flatformer: Flattened window attention for efficient point cloud transformer.**  
    In *Proceedings of CVPR, 2023.*

11. Moskalev, A., Prakash, M., Liao, R., and Mansi, T.  
    **SE(3)-Hyena operator for scalable equivariant learning.**  
    *arXiv:2407.01049, 2024.*

12. Perin, A. and Deny, S.  
    **On the ability of deep networks to learn symmetries from data: A neural kernel theory.**  
    *arXiv:2412.11521, 2024.*

13. Qu, E. and Krishnapriyan, A.  
    **The importance of being scalable: Improving the speed and accuracy of neural network interatomic potentials across chemical domains.**  
    In *Advances in Neural Information Processing Systems*, vol. 37, pages 139030–139053, 2024.

14. Suk, J., Hehn, T., Behboodi, A., and Cesa, G.  
    **ViNE-GATr: Scaling geometric algebra transformers with virtual‐node embeddings.**  
    In *ICLR 2025 Workshop on Machine Learning Multiscale Processes, 2025.*

15. Sun, P., Tan, M., Wang, W., Liu, C., Xia, F., Leng, Z., and Anguelov, D.  
    **Swformer: Sparse window transformer for 3D object detection in point clouds.**  
    In *Proceedings of ECCV, 2022.*

16. Wang, P.-S.  
    **Octformer: Octree-based transformers for 3D point clouds.**  
    *ACM Transactions on Graphics (SIGGRAPH), 42(4), 2023.*

17. Wu, X., Jiang, L., Wang, P., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., and Zhao, H.  
    **Point Transformer V3: Simpler, faster, stronger.**  
    In *Proceedings of CVPR, 2024.*

18. Zhdanov, M., Welling, M., and van de Meent, J.-W.  
    **Erwin: A tree-based hierarchical transformer for large-scale physical systems.**  
    *arXiv:2502.17019, 2025.*

19. Zhu, Z. and Soricut, R.  
    **H-transformer-1d: Fast one-dimensional hierarchical attention for sequences.**  
    In *Proceedings of NeurIPS*, pages 3801–3815. ACL, 2021.

20. Zhu, M., Ghaffari, M., Clark, W. A., and Peng, H.  
    **E2PN: Efficient SE(3)-equivariant point network.**  
    In *Proceedings of CVPR, pages 1223–1232, 2023.*

---


