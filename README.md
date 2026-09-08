# Analysis and Prediction of Medical Service Utilization Among Adults Aged 50 and Older: A Machine Learning Approach

**Pedro Kauan Silveira Silva**, Mikeyas Brito dos Santos, Wesley Barbosa Silva, Bruno Riccelli dos Santos Silva, Wellington Franco, Paulo Cesar Cortez

Federal University of Ceara (UFC), Brazil

---

## Associated Publication

This repository contains the experimental code and data supporting the IEEE conference paper:

> Silva, P. K. S. et al. "Analysis and Prediction of Medical Service Utilization Among Adults Aged 50 and Older: A Machine Learning Approach." IEEE Conference Proceedings.

The LaTeX source files are located in `.docs/article_updated.tex` (initial draft) and `.docs/article_updated_v2.tex` (revised version).

---

## Abstract

The progressive aging of the global population poses major challenges to healthcare networks, demanding a thorough understanding of healthcare utilization patterns among older adults. This study applies machine learning methods to examine and model predictive variables associated with healthcare-seeking behavior among individuals aged 50 and older. Using data from the National Poll on Healthy Aging (NPHA), ten supervised classifiers -- spanning linear, probabilistic, instance-based, tree-based, neural network, and graph-based paradigms -- are rigorously evaluated under a 10-fold stratified cross-validation protocol with cost-sensitive learning. Statistical significance is assessed via the Wilcoxon signed-rank test, and feature importance is analyzed through Mean Decrease Impurity (MDI). Results demonstrate that instance-based methods (KNN) achieve the highest discriminative performance under preserved epidemiological prevalence, offering actionable insights for clinical triage and healthcare resource planning.

---

## Project Structure

```
.
├── src/
│   ├── AUSMI.ipynb                            # Main analysis notebook
│   ├── AUSMI_executed.ipynb                   # Fully executed version
│   ├── AUSMI_backup_pre_cost_sensitive.ipynb  # Pre-cost-sensitive snapshot
│   └── NPHA-doctor-visits.csv                 # Dataset (714 samples, 14 features)
├── imagens/                                   # Figures for the LaTeX article
├── .docs/
│   ├── article_updated.tex                    # LaTeX article (initial draft)
│   ├── article_updated_v2.tex                 # LaTeX article (revised, IEEE format)
│   ├── metodologia_artigo.md                  # Extended methodology (Portuguese)
│   ├── cost_sensitive_learning.md             # Cost-sensitive implementation notes
│   ├── analise_hiperparametros.md             # Hyperparameter analysis and critique
│   └── resultados_finais_cost_sensitive.md    # Final cost-sensitive results
├── requirements.txt                           # Python dependencies
├── README.md
└── .gitignore
```

---

## Dataset

| Property   | Description |
|------------|-------------|
| **Source** | National Poll on Healthy Aging (NPHA) |
| **Samples**| 714 |
| **Features**| 14 demographic and health-related variables (all categorical) |
| **Target** | Annual doctor-visit frequency, binarized: **0--1 visits** vs. **2+ visits** |

All features are categorical (ordinal or nominal) and include: age bracket, race, gender, employment status, physical/mental/dental health self-assessments, sleep disturbance indicators (stress, medication, pain, bathroom needs, unknown causes), trouble sleeping, and prescription sleep medication usage.

The target variable was initially a 3-class distribution. Classes 2 and 3 were merged to form the positive class (2+ visits), creating a binary problem with approximately 18.4% minority class samples (0--1 visits: 131; 2+ visits: 583).

---

## Methodology

### Preprocessing

1. **Exploratory data analysis**: descriptive statistics, distribution plots per feature, demographic distributions, boxplot of target variable, and a correlation heatmap.
2. **Target binarization**: merging classes 2 and 3 into a single "2+ visits" category to produce a more balanced binary problem.
3. **One-Hot Encoding**: applied to all categorical features (no first-column drop), producing a full binary representation.
4. **Feature selection**: 24 low-importance features removed via MDI (Mean Decrease Impurity) consensus across multiple iterations. Removed categories include under-represented race codes, sparse health-assessment extremes, and near-constant features.
5. **Normalization**: MinMaxScaler (scale [0, 1]) applied inside the cross-validation loop (`fit_transform` on training splits, `transform` on test splits) to prevent data leakage.

### Dimensionality Reduction (Visualization Only)

Principal Component Analysis (PCA) with 2 and 3 components was applied using StandardScaler for exploratory visualization. PCA was not used as input to the classifiers.

### Validation Strategy

- **10-fold stratified cross-validation** (`KFold`, `shuffle=True`, `random_state=42`).
- Within each fold, the training set is further split into internal training (80%) and validation (20%) sets.
- Hyperparameters are tuned via **manual grid search** using `sklearn.model_selection.ParameterGrid`. The selection criterion is **macro F1-score** on the validation set.
- Best hyperparameters are retrained on the full training fold and evaluated on the held-out test fold.

### Cost-Sensitive Learning

To address the class imbalance (18.4% minority), the following techniques were applied:

- `class_weight='balanced'` for Decision Tree, Random Forest, SVM, and Logistic Regression.
- `scale_pos_weight` for XGBoost (ratio computed per fold on the internal training split).
- `sample_weight=compute_sample_weight('balanced', y)` for MLP (scikit-learn's MLPClassifier does not accept `class_weight` directly).
- `priors=[0.5, 0.5]` (uniform prior) for Gaussian Naive Bayes.
- KNN uses `weights='distance'` for distance-based neighbor weighting.
- Nearest Centroid and Supervised OPF do not support native class weighting and are evaluated as-is.

### Evaluation Metrics

For each fold and each model, five metrics are computed on the test set:

| Metric     | Averaging |
|------------|-----------|
| Accuracy   | --        |
| Precision  | weighted  |
| Recall     | weighted  |
| F1-score   | weighted  |
| AUROC      | binary (using predicted probabilities via `predict_proba`) |

Results are reported as mean and standard deviation across 10 folds, with 95% confidence intervals via Student's t-distribution. Confusion matrices are generated for each model.

### Statistical Validation

The **Wilcoxon signed-rank test** is applied to evaluate whether performance differences between model pairs are statistically significant. All pairwise comparisons are performed (10 models choose 2 = 45 comparisons) for each of the five metrics, at significance level alpha = 0.05. The null hypothesis (equal performance) is rejected when p < 0.05.

---

## Models Evaluated

| # | Model | Library | Hyperparameter Search Space |
|---|-------|---------|----------------------------|
| 1 | K-Nearest Neighbors | scikit-learn | `n_neighbors`: [10, 20, 30, 40]; `metric`: [euclidean, manhattan, minkowski, cosine] |
| 2 | Decision Tree | scikit-learn | `max_depth`: [3, 5, 7]; `min_samples_split`: [10, 20, 30]; `min_samples_leaf`: [5, 10] |
| 3 | Random Forest | scikit-learn | `n_estimators`: [100, 200, 300]; `max_depth`: [3, 5, 7]; `min_samples_leaf`: [2, 5, 10] |
| 4 | SVM (RBF) | scikit-learn | `C`: [0.1, 1, 10]; `kernel`: [rbf]; `gamma`: [scale, 0.1] |
| 5 | MLP | scikit-learn | `hidden_layer_sizes`: [(32,16), (64,)]; `alpha`: [0.01, 0.1, 1.0]; `activation`: [relu, tanh]; `solver`: [adam]; `learning_rate_init`: [0.001] |
| 6 | Logistic Regression | scikit-learn | `C`: [0.01, 0.1, 1, 10]; `solver`: [saga]; `penalty`: [l1, l2] |
| 7 | XGBoost | xgboost | `n_estimators`: [100, 200]; `max_depth`: [3, 5]; `learning_rate`: [0.01, 0.05, 0.1]; `subsample`: [0.8]; `colsample_bytree`: [0.8]; `reg_alpha`: [0.1, 1.0] |
| 8 | Gaussian Naive Bayes | scikit-learn | `var_smoothing`: [1e-11, 1e-10, 1e-9, 1e-8, 1e-7] |
| 9 | Nearest Centroid | scikit-learn | `metric`: [euclidean, manhattan]; `shrink_threshold`: [None, 0.1, 0.5, 1.0] |
| 10 | Supervised OPF | opfython | `distance`: [euclidean, squared_euclidean, log_squared_euclidean, manhattan, canberra, chebyshev] |

All grid searches use macro F1-score on the internal validation split as the selection criterion. A detailed hyperparameter analysis and critique is available in `.docs/analise_hiperparametros.md`.

---

## Results

Mean metrics across 10-fold cross-validation after cost-sensitive learning. Full results with standard deviations, 95% confidence intervals, and per-fold tracking are available in `src/AUSMI.ipynb` and `.docs/resultados_finais_cost_sensitive.md`.

| Model | Accuracy | F1-score (weighted) |
|-------|----------|---------------------|
| K-Nearest Neighbors | 0.788 | 0.736 |
| Supervised OPF | 0.622 | 0.651 |
| SVM | 0.548 | 0.589 |
| XGBoost | 0.537 | 0.586 |
| Random Forest | 0.515 | 0.569 |
| MLP | 0.504 | 0.556 |
| Nearest Centroid | 0.497 | 0.553 |
| Logistic Regression | 0.537 | 0.546 |
| Decision Tree | 0.416 | 0.451 |
| Gaussian Naive Bayes | 0.373 | 0.397 |

### Pre-Cost-Sensitive Baseline Comparison

For reference, the table below shows the results *prior* to cost-sensitive adjustment (mean +/- standard deviation). These values reflect models that did not penalize misclassification of the minority class and therefore achieved higher raw accuracy at the expense of recall on the under-represented class.

| Model | Accuracy | F1-score (weighted) |
|-------|----------|---------------------|
| XGBoost | 0.817 +/- 0.045 | 0.735 +/- 0.062 |
| Random Forest | 0.817 +/- 0.045 | 0.735 +/- 0.062 |
| MLP | 0.811 +/- 0.044 | 0.732 +/- 0.062 |
| Logistic Regression | 0.811 +/- 0.043 | 0.732 +/- 0.062 |
| KNN | 0.810 +/- 0.051 | 0.740 +/- 0.060 |
| Decision Tree | 0.807 +/- 0.035 | 0.732 +/- 0.053 |
| SVM | 0.780 +/- 0.039 | 0.729 +/- 0.051 |

---

## Key Findings

- Binarizing the target variable (merging 2--3 and 4+ visit counts into "2+ visits") substantially improved class separability compared to the original 3-class formulation.
- Removing 24 low-importance features via MDI consensus reduced noise and clarified decision boundaries without significant information loss.
- Cost-sensitive learning (class weighting, scale_pos_weight, sample_weight) was essential for meaningful minority-class predictions. Without it, several classifiers collapsed to a majority-class-only strategy despite superficially high accuracy.
- KNN with distance weighting (weights='distance') and Supervised OPF demonstrated the best balance between accuracy and F1-score after cost-sensitive adjustment.
- The Wilcoxon signed-rank test confirmed statistically significant performance differences (p < 0.05) among multiple model pairs, providing a principled basis for model ranking.
- Tree-based ensembles (Random Forest, XGBoost) and linear models (Logistic Regression, SVM) showed moderate performance degradation under cost-sensitive constraints, highlighting their sensitivity to class priors.
- The project serves as a reproducible benchmark for applying cost-sensitive learning to imbalanced healthcare survey data of older adults.

---

## Setup and Reproduction

### Prerequisites

Python 3.8 or later is recommended. A virtual environment is included (`.venv/`, not tracked by git). Dependencies are listed in `requirements.txt`.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# or: .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation and numerical operations |
| scikit-learn | Models, metrics, preprocessing, PCA, model selection |
| xgboost | XGBoost classifier |
| opfython | Supervised Optimum-Path Forest |
| imbalanced-learn | Resampling utilities (imported, not used in final pipeline) |
| matplotlib, seaborn | Visualizations |
| scipy | Wilcoxon test, confidence intervals |
| jupyterlab | Interactive notebook environment |

### Running the Analysis

1. Activate the virtual environment.
2. Start JupyterLab: `jupyter lab`
3. Open `src/AUSMI.ipynb`.
4. Run all cells in order. The notebook is fully self-contained and will:
   - Load and explore the dataset.
   - Perform preprocessing, encoding, and feature selection.
   - Execute the 10-fold cross-validation pipeline for all 10 models.
   - Generate all tables, charts, confusion matrices, and statistical test outputs.
   - The executed output is pre-saved in `src/AUSMI_executed.ipynb`.

Reproducibility is ensured by fixed `random_state=42` across all stochastic components (KFold splits, model initializations, train/test splits).

---

## References

- United Nations, Department of Economic and Social Affairs, Population Division. *World Population Prospects 2019*.
- National Poll on Healthy Aging (NPHA). University of Michigan Institute for Healthcare Policy and Innovation.
- Dua, D. and Graff, C. *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science.
- Demsar, J. "Statistical Comparisons of Classifiers over Multiple Data Sets." *Journal of Machine Learning Research*, 7:1--30, 2006.
- Wilcoxon, F. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, 1(6):80--83, 1945.
- Breiman, L. "Random Forests." *Machine Learning*, 45(1):5--32, 2001.
- Chen, T. and Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*, 2016.
- Papa, J. P., Falcao, A. X., and Suzuki, C. T. N. "Supervised pattern classification based on optimum-path forest." *International Journal of Imaging Systems and Technology*, 19(2):120--131, 2009.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{silva2025medical,
  title     = {Analysis and Prediction of Medical Service Utilization Among Adults Aged 50 and Older: A Machine Learning Approach},
  author    = {Silva, Pedro Kauan Silveira and Santos, Mikeyas Brito dos and Silva, Wesley Barbosa and Silva, Bruno Riccelli dos Santos and Franco, Wellington and Cortez, Paulo Cesar},
  booktitle = {IEEE Conference Proceedings},
  year      = {2025}
}
```
