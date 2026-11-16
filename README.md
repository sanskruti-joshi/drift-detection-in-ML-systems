# Drift Detection in Machine Learning Systems

A framework for monitoring, detecting, and responding to data drift in machine learning systems using statistical detection techniques and adaptive model retraining.

**Research Project | MSc Data Science | Published Research**

## Overview

Machine learning models are trained on historical data, but real-world data distributions can change over time. These changes, known as *data drift*, can reduce model reliability and negatively affect predictive performance.

This project explores a framework for identifying distributional changes, evaluating their impact on model performance, and initiating model retraining when significant drift is detected.

The framework is evaluated through a customer churn prediction experiment using controlled feature and label drift simulations.

## Key Features

- **Baseline Model:** Random Forest classifier for customer churn prediction.
- **Drift Simulation:** Controlled modifications to numerical features and target class distribution.
- **Statistical Drift Detection:** Kolmogorov–Smirnov (KS) test and Population Stability Index (PSI).
- **Performance Monitoring:** Evaluation using accuracy, precision, recall, and F1-score.
- **Adaptive Retraining:** PSI-based drift severity classification and retraining trigger.
- **Visualization:** Interactive Plotly visualizations for feature-level drift and model performance comparisons.

## Methodology

The experimental workflow consists of the following stages:

1. **Data Preparation** – Cleaning, handling missing values, encoding categorical variables, and preparing training and test datasets.
2. **Baseline Training** – Training a Random Forest classifier and evaluating its initial performance.
3. **Drift Simulation** – Introducing controlled changes to `tenure`, `MonthlyCharges`, and `TotalCharges`, along with modifications to the churn class distribution.
4. **Drift Detection** – Comparing baseline and drifted feature distributions using KS statistics and PSI.
5. **Performance Evaluation** – Measuring model performance under baseline and drifted conditions.
6. **Adaptive Retraining** – Triggering retraining when a monitored feature reaches the high-drift threshold.
7. **Visualization** – Presenting drift severity, feature distribution changes, and model performance comparisons.

## Drift Detection Techniques

Two statistical methods are used:

| Technique | Purpose |
|---|---|
| Kolmogorov–Smirnov (KS) Test | Identifies differences between feature distributions using the KS statistic and p-value. |
| Population Stability Index (PSI) | Quantifies the magnitude of distributional change and categorizes drift severity. |

### PSI-based Drift Classification

| PSI Value | Drift Severity |
|---|---|
| PSI < 0.1 | Low |
| 0.1 ≤ PSI < 0.25 | Moderate |
| PSI ≥ 0.25 | High |

High drift in an important feature triggers the adaptive retraining mechanism.

## Experimental Results

### Baseline Model Performance

The published experiment reported the following baseline results:

| Metric | Score |
|---|---:|
| Accuracy | 0.7882 |
| Precision | 0.6587 |
| Recall | 0.4759 |
| F1-score | 0.5526 |

### Drift Detection

The original dissertation reported the following KS and PSI results:

| Feature | KS Statistic | PSI |
|---|---:|---:|
| `tenure` | 0.181 | 0.093 |
| `MonthlyCharges` | 0.293 | 0.300 |
| `TotalCharges` | 0.081 | 0.027 |

**Post-publication implementation update:** The repository's PSI calculation was subsequently corrected to account for observations outside the reference histogram range. The updated notebook produces recalculated PSI values:

| Feature | Updated PSI | Drift Severity |
|---|---:|---|
| `tenure` | 0.232984 | Moderate |
| `MonthlyCharges` | 1.661896 | High |
| `TotalCharges` | 0.263045 | High |

These updated values belong to the maintained repository implementation; they are not the values reported in the original publication.

### Model Performance Under Drift

The experiment evaluates model performance at three stages:

- Baseline model
- Model under drifted conditions
- Model after adaptive retraining

The published results showed that drift affected predictive performance, particularly recall. Retraining improved recall and F1-score relative to the drifted model, although baseline accuracy was not fully recovered.

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Statistical Analysis:** SciPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Development Environment:** Jupyter Notebook

## Getting Started

### Prerequisites

- Python
- Jupyter Notebook or JupyterLab

### Installation

Clone the repository:

```bash
git clone https://github.com/sanskruti-joshi/drift-detection-in-ML-systems.git
cd drift-detection-in-ML-systems
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `notebooks/Drift_detection_with_retrain_trigger.ipynb` and execute the notebook cells sequentially.

The dataset is included in the repository under `data/`.

## Research Publication

**Title:** A Framework for Monitoring, Detecting, and Adaptive Response to Data Drift in Machine Learning Systems

**Publication:** International Journal of Science and Innovative Research (IJSCI)

**DOI:** [10.70849/ijsci03052674252](https://ijsci.com/paper-details.php/14074252)

The dissertation and associated publication details are available in the `docs/` directory.

### Implementation Notes

The repository has undergone post-publication updates to improve code organization, preprocessing transparency, and calculation correctness.

The maintained notebook may therefore produce numerical results that differ from those in the original publication. Published findings and subsequently recalculated results are identified separately where applicable.

## Future Scope

Potential extensions include:

- Real-time drift monitoring using streaming data.
- Monitoring additional feature and concept drift scenarios.
- Integration with automated ML deployment and retraining pipelines.
- Evaluation on continuously evolving real-world datasets.

## License

The original project code is distributed under the MIT License. See [LICENSE](LICENSE) for details.

Third-party datasets and research documents may be subject to separate copyright and licensing terms.
