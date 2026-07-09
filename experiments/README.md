# Experiments

Each notebook is standalone. Cell 1 installs and patches dependencies
(`imagecorruptions`, `scipy`, NumPy/skimage compatibility shims); Cell 2 runs
the experiment and prints a results table.

| Notebook | What it runs |
|---|---|
| `imagenet_c_benchmark.ipynb` | BMIA vs TENT/EATA on ImageNet-C (ResNet-50), 15 corruptions, severity 5, two learning rates |
| `imagenet_c_fog_glass_diagnostic.ipynb` | Diagnostic runs for the fog and glass-blur corruptions |
| `imagenet_c_glass_blur.ipynb` | Glass-blur corruption at the aggressive learning rate |
| `cifar100c_main_comparison.ipynb` | Main CIFAR-100-C comparison (ResNet-18): collapse rates and accuracy |
| `cifar100c_all_corruptions.ipynb` | All 15 CIFAR-100-C corruption types |
| `cifar100c_accuracy_ceiling.ipynb` | Accuracy-ceiling study for BMIA on CIFAR-100-C |
| `eata_collapse_analysis.ipynb` | Entropy-filter and Fisher-anchor analysis of EATA collapse |
| `tinyimagenet_c.ipynb` | Tiny-ImageNet-C (200 classes) |
| `medmnist_benchmark.ipynb` | PathMNIST / DermaMNIST / BloodMNIST |
| `vit_small_generalization.ipynb` | ViT-Small (LayerNorm) generalization |
| `batch_size_sensitivity.ipynb` | Collapse vs batch size |
| `multiseed_stability.ipynb` | Multi-seed stability of the zero-collapse property |
| `threshold_ablation.ipynb` | Sensitivity to the ORCA `τ` threshold |
| `bmia_extreme_lr_mode.ipynb` | Optional configuration for the extreme-lr regime |
| `bmia_accuracy_mode.ipynb` | Optional configuration for accuracy at normal lr |
| `csd_theorem_validation.ipynb` | Empirical checks of the CSD theorem |
| `come_cotta_baselines.ipynb` | CoME and CoTTA baselines |
| `robustness_defence.ipynb` | Additional robustness experiments |
| `bmia_peak_analysis.ipynb` | Peak-accuracy analysis |
| `lambda_trajectory.ipynb` | ORCA `λ` trajectory over adaptation |

## Common settings

- Optimizer: SGD, momentum 0.9; BatchNorm affine parameters only.
- Collapse criterion: a run is collapsed if the dominant class exceeds `2/K` of
  predictions and fewer than `K/2` classes are predicted, or fewer than `K/5`
  classes are predicted (`K` = number of classes).
- Corruptions: severity 5 unless otherwise noted, generated with
  `imagecorruptions`.
