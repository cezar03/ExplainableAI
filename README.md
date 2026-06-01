# Pandora's Box - Explainable AI for Credit Risk

This repository contains the final Explainable AI project notebooks for modeling
the German Credit dataset and comparing explanation methods for credit-risk
predictions. The work includes exploratory data analysis, model-assumption
checks, decision tree and random forest classifiers, LIME/SHAP explanations, and
a user-study design.

## Project Structure

```text
.
|-- data/
|   |-- raw/                  # Original German Credit ARFF dataset
|   `-- processed/            # Encoded tabular data
|-- docs/                     # Study-design and project documentation
|-- notebooks/                # Analysis, modeling, and XAI notebooks
|-- outputs/
|   `-- figures/              # Exported figures used in the project
|-- requirements.txt          # Main Python environment
`-- README.md
```

## Setup

Create a virtual environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then start Jupyter with `notebooks/` as the working directory:

```powershell
Set-Location notebooks
jupyter lab
```

The notebooks use paths relative to `notebooks/`, for example
`../data/raw/dataset_31_credit-g.arff`.

## Suggested Notebook Order

1. `notebooks/eda.ipynb`
2. `notebooks/assumption_lda.ipynb`
3. `notebooks/assumption_linearity.ipynb`
4. `notebooks/assumption_multicollinearity_correlation_vif.ipynb`
5. `notebooks/random_forest.ipynb`
6. `notebooks/decision_tree_exai.ipynb`
7. `notebooks/rf_exai.ipynb`
8. `notebooks/random_forest_lime.ipynb`

## Data

- `data/raw/dataset_31_credit-g.arff` is the original German Credit dataset.
- `data/processed/credit_g_encoded_onehot.csv` is the encoded tabular version.

## Outputs

Selected generated figures are stored in `outputs/figures/` so they can be used
directly in reports, presentations, and the user-study materials.

## Documentation

The user-study design is documented in `docs/user_study.md`.

## Git Notes

Local IDE settings and virtual environments are ignored by `.gitignore`.
Generated figures in `outputs/figures/` are intentionally kept in the repository.
