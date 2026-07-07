import argparse
from contextlib import contextmanager
import json
import logging
import math
from pathlib import Path
import sys
import time

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
LOGGER = logging.getLogger("provider_payment_model")
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
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write detailed local trace logs with timings for each major pipeline phase.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console and trace log verbosity. Defaults to DEBUG when --trace is used, otherwise INFO.",
    )
    parser.add_argument(
        "--library-verbose",
        action="store_true",
        help="Pass verbose=True into supported library calls, especially sklearn pipelines.",
    )
    parser.add_argument(
        "--torch-detect-anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection for debugging training issues.",
    )
    parser.add_argument(
        "--trace-every-n-batches",
        type=int,
        default=0,
        help="When tracing, log training batch progress every N batches. Use 0 to disable batch progress.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path, log_level: str) -> Path:
    log_path = output_dir / "training_trace.log"
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level))

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    return log_path


@contextmanager
def trace_step(name: str, enabled: bool):
    start = time.perf_counter()
    if enabled:
        LOGGER.info("TRACE START %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if enabled:
            LOGGER.info("TRACE DONE %s in %.2fs", name, elapsed)


def load_data(csv_path: str, sample_rows: int) -> pd.DataFrame:
    nrows = sample_rows if sample_rows and sample_rows > 0 else None
    LOGGER.info("Reading CSV from %s", csv_path)
    if nrows:
        LOGGER.info("Limiting read to %,d rows for smoke test", nrows)
    df = pd.read_csv(csv_path, low_memory=False, nrows=nrows)
    df.columns = [column.strip() for column in df.columns]
    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column {TARGET!r}. Found columns: {list(df.columns)[:30]}..."
        )
    LOGGER.info("Loaded %,d rows and %,d columns", len(df), len(df.columns))
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
    starting_rows = len(df)

    # Convert likely numeric-looking object columns before selecting features.
    converted_columns = []
    for column in df.columns:
        if column == TARGET or any(token in column for token in ["Amt", "Chrg", "Srvcs", "Benes", "HCPCS", "RUCA"]):
            try:
                df[column] = money_to_float(df[column])
                converted_columns.append(column)
            except ValueError:
                LOGGER.debug("Column %s looked numeric but could not be converted", column)
                pass

    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    y = np.log1p(df[TARGET].astype(float))
    LOGGER.info("Removed %,d rows with missing or nonpositive target", starting_rows - len(df))
    LOGGER.debug("Converted %,d numeric-like columns: %s", len(converted_columns), converted_columns)

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
    LOGGER.debug("Numeric columns: %s", numeric_columns)
    LOGGER.debug("Categorical columns: %s", categorical_columns)
    return X, y, numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    library_verbose: bool,
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
        verbose=library_verbose,
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)),
        ],
        verbose=library_verbose,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
        verbose=library_verbose,
    )


def to_tensor_dataset(X: np.ndarray, y: pd.Series) -> TensorDataset:
    features = torch.tensor(X.astype(np.float32), dtype=torch.float32)
    target = torch.tensor(y.to_numpy(dtype=np.float32), dtype=torch.float32)
    return TensorDataset(features, target)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    device,
    trace_every_n_batches,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_val = math.inf
    best_state = None
    LOGGER.info("Optimizer: AdamW learning_rate=%s weight_decay=1e-4", learning_rate)
    LOGGER.info("Loss: SmoothL1Loss")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        for batch_index, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if trace_every_n_batches and batch_index % trace_every_n_batches == 0:
                LOGGER.info(
                    "TRACE epoch=%02d batch=%04d/%04d batch_loss=%.4f",
                    epoch,
                    batch_index,
                    len(train_loader),
                    loss.item(),
                )

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
        epoch_elapsed = time.perf_counter() - epoch_start
        LOGGER.info(
            "epoch=%02d train_loss=%.4f val_loss=%.4f elapsed=%.2fs",
            epoch,
            train_loss,
            val_loss,
            epoch_elapsed,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            LOGGER.debug("New best validation loss %.4f at epoch %02d", best_val, epoch)

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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_level = args.log_level or ("DEBUG" if args.trace else "INFO")
    log_path = configure_logging(output_dir, log_level)

    if args.trace and args.trace_every_n_batches == 0:
        args.trace_every_n_batches = 25

    LOGGER.info("=" * 72)
    LOGGER.info("CMS provider payment model run started")
    if args.trace:
        LOGGER.info("TRACE ENABLED: detailed local tracing is active")
    else:
        LOGGER.info("TRACE DISABLED: add --trace for stage timings and batch progress")
    LOGGER.info("Trace log: %s", log_path.resolve())
    LOGGER.info("Arguments: %s", vars(args))
    LOGGER.info("Effective log level: %s", log_level)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        LOGGER.warning("PyTorch autograd anomaly detection is enabled; training will be slower.")

    with trace_step("load_data", args.trace):
        df = load_data(args.csv, args.sample_rows)
    with trace_step("prepare_frame", args.trace):
        X, y, numeric_columns, categorical_columns = prepare_frame(df)

    LOGGER.info("Rows after cleaning: %,d", len(X))
    LOGGER.info("Numeric features: %,d", len(numeric_columns))
    LOGGER.info("Categorical features: %,d", len(categorical_columns))

    with trace_step("train_validation_test_split", args.trace):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=args.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=args.seed
        )
        LOGGER.info(
            "Split rows: train=%,d validation=%,d test=%,d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

    preprocessor = build_preprocessor(numeric_columns, categorical_columns, args.library_verbose)
    with trace_step("fit_transform_preprocessor", args.trace):
        X_train_np = preprocessor.fit_transform(X_train)
        LOGGER.info("Encoded training matrix shape: %s", X_train_np.shape)
    with trace_step("transform_validation_and_test", args.trace):
        X_val_np = preprocessor.transform(X_val)
        X_test_np = preprocessor.transform(X_test)
        LOGGER.info("Encoded validation matrix shape: %s", X_val_np.shape)
        LOGGER.info("Encoded test matrix shape: %s", X_test_np.shape)

    with trace_step("build_dataloaders", args.trace):
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
        LOGGER.info("Train batches: %,d", len(train_loader))
        LOGGER.info("Validation batches: %,d", len(val_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training on %s with %,d encoded features", device, X_train_np.shape[1])

    model = PaymentRegressor(input_dim=X_train_np.shape[1]).to(device)
    LOGGER.debug("Model architecture: %s", model)
    with trace_step("train_model", args.trace):
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            trace_every_n_batches=args.trace_every_n_batches,
        )

    with trace_step("predict_test_set", args.trace):
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

    LOGGER.info("Metrics:\n%s", json.dumps(metrics, indent=2))

    with trace_step("save_artifacts", args.trace):
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
    LOGGER.info("Saved artifacts to: %s", output_dir.resolve())
    LOGGER.info("Saved trace log to: %s", log_path.resolve())


if __name__ == "__main__":
    main()
