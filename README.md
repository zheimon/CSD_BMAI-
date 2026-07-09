# CSD-BMIA: Collapse–Stability Duality and Mutual-Information Test-Time Adaptation
> New here? Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a 1-minute visual overview.
Code for the paper *"Collapse Stability Duality: Test-Time Adaptation via Mutual
Information Maximization."*

Test-time adaptation (TTA) methods based on entropy minimization can collapse —
predicting a single class for every input — especially at aggressive learning
rates and small batch sizes. This repository provides:

- **BMIA-Adaptive**, a TTA method that maximizes mutual information `I(X;Y)` with
  an anchor whose strength is controlled online by **ORCA** (Online Real-time
  Collapse Avoidance), and
- the full set of experiments used to study collapse across datasets,
  architectures, learning rates, and batch sizes.

## Method

For a model with BatchNorm affine parameters `θ = (γ, β)` and source values `θ₀`,
BMIA-Adaptive minimizes

```
L(θ) = H(Y|X) − H(Y) + λ · ‖θ − θ₀‖²
```

The first two terms maximize mutual information (confident but diverse
predictions); the anchor keeps `θ` near the source. ORCA monitors the dominant
prediction fraction (`dom%`) each batch and adjusts `λ` — raising it when
concentration is detected, relaxing it when the model is stable.

Only BatchNorm affine parameters are updated; the backbone is frozen.

## Repository layout

```
CSD-BMIA/
├── experiments/     # one notebook per experiment (see experiments/README.md)
├── theory/          # theorem statements and proofs
├── requirements.txt
└── LICENSE
```

## Datasets

- **CIFAR-10-C / CIFAR-100-C**, **Tiny-ImageNet-C**, **ImageNet-C**
  (Hendrycks & Dietterich, 2019)
- **MedMNIST** (PathMNIST, DermaMNIST, BloodMNIST)

ImageNet-C experiments expect the ImageNet validation set (the
*ImageNet Object Localization Challenge* layout). Corruptions are generated
on the fly with the `imagecorruptions` package.

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook experiments/
```

Each notebook is self-contained: the first cell installs/patches dependencies,
the second runs the experiment and prints a results table. A GPU is recommended
for the ImageNet-C notebooks.

## Reproducing the main results

| Result | Notebook |
|---|---|
| ImageNet-C, 15 corruptions × 2 lr | `experiments/imagenet_c_benchmark.ipynb` (+ `imagenet_c_fog_glass_diagnostic.ipynb`, `imagenet_c_glass_blur.ipynb`) |
| CIFAR-100-C main comparison | `experiments/cifar100c_main_comparison.ipynb` |
| CIFAR-100-C, all 15 corruptions | `experiments/cifar100c_all_corruptions.ipynb` |
| EATA collapse analysis | `experiments/eata_collapse_analysis.ipynb` |
| Tiny-ImageNet-C | `experiments/tinyimagenet_c.ipynb` |
| MedMNIST | `experiments/medmnist_benchmark.ipynb` |
| ViT-Small generalization | `experiments/vit_small_generalization.ipynb` |
| Batch-size sensitivity | `experiments/batch_size_sensitivity.ipynb` |
| Multi-seed stability | `experiments/multiseed_stability.ipynb` |

## Author

**Anish Kumar Thakur** — Alumni 2025, Department of Computer Science and
Engineering, ABV-IIITM. 2026.

## License

Released under the MIT License. See `LICENSE`.
