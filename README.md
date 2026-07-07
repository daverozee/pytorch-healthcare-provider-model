# PyTorch Healthcare Provider Market Intelligence Starter

This starter trains a PyTorch regression model on the CMS **Medicare Physician & Other Practitioners - by Provider** CSV.

The first model predicts `Tot_Mdcr_Pymt_Amt` from provider specialty, location, entity/participation fields, and utilization/charge columns. This is a baseline for market intelligence questions like:

- Which provider segments have unusually high payment volume?
- Which specialties or regions are high-value?
- Which providers look unusual relative to peers?

## 1. Install Dependencies

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, use the official selector at https://pytorch.org/get-started/locally/ and install the CPU build.

## 2. Train The Model

Replace the CSV path with the location where you downloaded the CMS file:

```powershell
python train_provider_payment_model.py --csv "C:\Users\admin\Downloads\MUP_PHY_R26_P05_V10_D24_Prov.csv"
```

The script creates an `artifacts` folder containing:

- `provider_payment_model.pt` - trained PyTorch model weights
- `preprocessor.joblib` - scikit-learn preprocessing pipeline
- `metrics.json` - validation/test metrics
- `feature_columns.json` - selected input and target columns

## 3. What The Metrics Mean

- `mae` is mean absolute error in dollars after converting predictions back from log scale.
- `rmse` is root mean squared error in dollars.
- `r2_log` is R-squared on the log payment target. For this skewed healthcare payment data, the log metric is usually more meaningful than raw-dollar R-squared.

## 4. Next Project Step

After this single-year model works, download 2021-2024 provider CSVs. Then we can train a forecasting model: use 2021-2023 provider features to predict 2024 payment growth.

