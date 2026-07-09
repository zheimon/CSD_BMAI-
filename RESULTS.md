# Results — ImageNet-C (ResNet-50, severity 5, seed 42, episodic)

BMIA-Adaptive on the full 50,000-image ImageNet-C validation set, all 15
corruptions, two learning rates. Zero collapses across all runs.

| Corruption | Acc lr=0.005 | Acc lr=0.05 |
|---|---|---|
| gaussian_noise | 32.80 | 12.26 |
| shot_noise | 34.99 | 17.56 |
| impulse_noise | 34.44 | 16.65 |
| defocus_blur | 36.46 | 10.17 |
| glass_blur | 31.13 | 13.28 |
| motion_blur | 52.20 | 40.24 |
| zoom_blur | 56.75 | 28.21 |
| snow | 52.38 | 42.55 |
| frost | 45.36 | 32.11 |
| fog | 64.17 | 60.44 |
| brightness | 68.29 | 65.11 |
| contrast | 62.88 | 52.25 |
| elastic_transform | 52.68 | 44.32 |
| pixelate | 60.05 | 54.30 |
| jpeg_compression | 53.87 | 46.30 |
| **Average** | **49.23** | **35.72** |
