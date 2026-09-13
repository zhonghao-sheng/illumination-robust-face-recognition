# Illumination-Robust Face Recognition

How does model complexity affect face identification when the lighting direction changes? This project compares a linear baseline, a compact convolutional network, and EfficientNet-B0 using lighting groups that are held out from training.

The aim is to demonstrate a careful machine-learning workflow: define a meaningful test shift, fit preprocessing and models within training folds, choose hyperparameters on inner folds, and report limitations. The goal is **not** to maximize one headline accuracy number.

## Study design

Images are grouped by their recorded light-source azimuth and elevation. The angles are mapped to three-dimensional unit vectors; Euclidean k-means and a best-effort balancing step produce ten lighting bins. Each outer fold holds out one bin, so its lighting directions are absent from model training. Three grouped inner folds select hyperparameters using **macro F1**. Final models are trained on the full outer-training partition for a fixed epoch budget, then evaluated on the untouched outer test bin.

| Model | Training-side selection | Preprocessing | Final training |
| --- | --- | --- | --- |
| LDA + linear SVM | C ∈ {0.001, 0.01, 0.1} | CLAHE, per-image standardization, LDA fitted within each fold | Fit on all outer-training images |
| Custom CNN | 3×3, 5×5, or 7×7 kernels | CLAHE and per-image standardization; no stochastic augmentation | 18 epochs; 32→64→128-channel blocks |
| EfficientNet-B0 | Learning rate ∈ {0.001, 0.0005, 0.0003}; weight decay ∈ {0.00001, 0.0001, 0.0005} | Grayscale converted to RGB, 224×224, ImageNet normalization; mild training-only augmentation | ImageNet-pretrained, 15 epochs |

The inner search uses 10 epochs per CNN candidate and 3 per EfficientNet candidate. These fixed budgets keep the nested evaluation feasible; they do not guarantee that every model has converged. The public runner implements the described protocol, with configurable image sizes and budgets.

## Reported results

The final study reported the following **mean ± standard deviation across ten outer lighting folds**. These figures are transcribed from the final analysis. This repository does not include the original face images, per-fold predictions, or exact data snapshot, so the table is a record of that study rather than a claim that the current checkout independently reproduced the same numbers.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| LDA + SVM | 0.763 ± 0.160 | 0.773 ± 0.150 | 0.763 ± 0.160 | 0.762 ± 0.160 |
| Custom CNN | 0.851 ± 0.143 | 0.876 ± 0.111 | 0.851 ± 0.143 | 0.852 ± 0.142 |
| EfficientNet-B0 | 0.917 ± 0.113 | 0.925 ± 0.084 | 0.914 ± 0.045 | 0.916 ± 0.041 |

The most frequently selected settings were SVM C = 0.01 (9 of 10 folds), CNN kernel size 5×5 (7 folds), EfficientNet learning rate 0.0005 (6 folds), and EfficientNet weight decay 0.00001 (6 folds). EfficientNet had the highest mean scores in this study. The fold-to-fold variation matters: a model can perform well on average and still struggle with particular unseen lighting directions. The summary alone does not establish statistical significance between models.

## Reproduce the workflow

Use Python 3.10 or newer. Download the cropped Extended Yale B image set separately and place subject folders under `data/YaleFace/cropped/`; see [data/README.md](data/README.md) for the expected filenames. Face images are not included in this repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_splits.py --data-dir data/YaleFace/cropped --output-dir splits
python run_experiments.py --data-root data/YaleFace/cropped --splits-dir splits --folds all --output results/latest.json
```

The full nested run trains many neural models and can take substantial time. For a quick pipeline check, run `pytest` or restrict the runner to one fold and the linear baseline:

```bash
pytest
python run_experiments.py --folds 0 --models svm
```

The runner writes per-fold metrics, selected settings, and mean/standard-deviation summaries to the optional JSON output. Generated split CSVs and run outputs remain local by default. The checked-in `splits/fold_summary.json` is an example generated from a local non-ambient image set; rerun split generation for your own data copy.

## Data and interpretation

The [Extended Yale B dataset](https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b) contains controlled lighting and poses for 28 subjects. The study used a [cropped Kaggle derivative](https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full). The dataset page gives inconsistent image totals, so this repository reports the usable count found by its own filename parser when generating splits. Ambient frames and files without angle metadata are skipped.

This is closed-set identification in a controlled environment. It does not test unseen identities or lighting in everyday scenes. Bin sizes may differ because each recorded lighting direction is kept intact; the split generator reports actual counts instead of promising exact balance.

## References

- Georghiades, Belhumeur, and Kriegman. “From Few to Many: Illumination Cone Models for Face Recognition under Variable Lighting and Pose.” *IEEE TPAMI*, 2001.
- Belhumeur, Hespanha, and Kriegman. “Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.” *IEEE TPAMI*, 1997.
- Tan and Le. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” *ICML*, 2019.

The code is available under the [MIT License](LICENSE). Dataset images have their own provenance and are not redistributed here.
