import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET = "Tot_Mdcr_Pymt_Amt"
DROP_ALWAYS = {
    "Rndrng_NPI",
    "Rndrng_Prvdr_Last_Org_Name",
    "Rndrng_Prvdr_First_Name",
    "Rndrng_Prvdr_MI",
    "Rndrng_Prvdr_Crdntls",
    "Rndrng_Prvdr_St1",
    "Rndrng_Prvdr_St2",
    "Rndrng_Prvdr_City",
    "Rndrng_Prvdr_Zip5",
    "Rndrng_Prvdr_RUCA_Desc",
}


class PaymentRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.10),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch model on CMS provider-level Medicare data.")
    parser.add_argument("--csv", required=True, help="Path to the CMS provider CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model artifacts.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Optional row limit for a fast smoke test.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(csv_path: str, sample_rows: int) -> pd.DataFrame:
    nrows = sample_rows if sample_rows and sample_rows > 0 else None
    df = pd.read_csv(csv_path, low_memory=False, nrows=nrows)
    df.columns = [column.strip() for column in df.columns]
    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column {TARGET!r}. Found columns: {list(df.columns)[:30]}..."
        )
    return df


def money_to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
        .astype(float)
    )


def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = df.copy()

    # Convert likely numeric-looking object columns before selecting features.
    for column in df.columns:
        if column == TARGET or any(token in column for token in ["Amt", "Chrg", "Srvcs", "Benes", "HCPCS", "RUCA"]):
            try:
                df[column] = money_to_float(df[column])
            except ValueError:
                pass

    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    y = np.log1p(df[TARGET].astype(float))

    leakage_columns = {
        TARGET,
        "Tot_Mdcr_Alowd_Amt",
        "Tot_Mdcr_Stdzd_Amt",
        "Drug_Mdcr_Pymt_Amt",
        "Drug_Mdcr_Alowd_Amt",
        "Drug_Mdcr_Stdzd_Amt",
        "Med_Mdcr_Pymt_Amt",
        "Med_Mdcr_Alowd_Amt",
        "Med_Mdcr_Stdzd_Amt",
    }
    candidate_columns = [
        column
        for column in df.columns
        if column not in DROP_ALWAYS and column not in leakage_columns
    ]
    X = df[candidate_columns]

    numeric_columns = [
        column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]
    return X, y, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )


def to_tensor_dataset(X: np.ndarray, y: pd.Series) -> TensorDataset:
    features = torch.tensor(X.astype(np.float32), dtype=torch.float32)
    target = torch.tensor(y.to_numpy(dtype=np.float32), dtype=torch.float32)
    return TensorDataset(features, target)


def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_val = math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(X.astype(np.float32), dtype=torch.float32), batch_size=4096)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.csv, args.sample_rows)
    X, y, numeric_columns, categorical_columns = prepare_frame(df)

    print(f"Rows after cleaning: {len(X):,}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed
    )

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    train_loader = DataLoader(
        to_tensor_dataset(X_train_np, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        to_tensor_dataset(X_val_np, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {X_train_np.shape[1]:,} encoded features...")

    model = PaymentRegressor(input_dim=X_train_np.shape[1]).to(device)
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    pred_log = predict(model, X_test_np, device)
    actual_log = y_test.to_numpy()
    pred_dollars = np.expm1(pred_log)
    actual_dollars = np.expm1(actual_log)

    metrics = {
        "rows": int(len(X)),
        "encoded_features": int(X_train_np.shape[1]),
        "mae": float(mean_absolute_error(actual_dollars, pred_dollars)),
        "rmse": float(mean_squared_error(actual_dollars, pred_dollars, squared=False)),
        "r2_log": float(r2_score(actual_log, pred_log)),
    }

    print(json.dumps(metrics, indent=2))

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_dim": X_train_np.shape[1],
            "target": TARGET,
        },
        output_dir / "provider_payment_model.pt",
    )
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "target": TARGET,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
