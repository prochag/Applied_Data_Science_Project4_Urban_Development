
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "data" / "final"
OUTPUT_DIR = ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(num: str, title: str):
    print(f"\n{'═' * 60}")
    print(f"  MODELING SECTION {num} — {title}")
    print(f"{'═' * 60}")


def save_fig(filename: str):
    path = OUTPUT_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → outputs/models/{filename}")


# ═════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    section("LOAD", "Reading Final Dataset")

    path = ROOT / "nyc_urban_features.csv"
    df = pd.read_csv(path)

    print(f"  Loaded: {path}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target distribution:\n{df['high_development'].value_counts().to_string()}")

    return df


# ═════════════════════════════════════════════════════════════════════
# PREPARE FEATURES
# ═════════════════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame):
    section("5.1", "Preparing Features")

    df = df.copy()

    target = "high_development"

    # Drop ID / label columns that should not be directly used as numeric predictors
    drop_cols = [
        "high_development",
        "borough",
        "community_board",
    ]

    # Potential leakage columns:
    # The target is defined using permit activity, so direct permit count variables
    # can make the model artificially strong.
    leakage_cols = [
        "total_permits",
        "new_buildings",
        "major_alterations",
        "permits_3yr_avg",
    ]

    # One-hot encode borough
    if "borough" in df.columns:
        df = pd.get_dummies(df, columns=["borough"], drop_first=False)

    # Replace inf and fill missing
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # All-feature version
    all_features = [
        col for col in df.columns
        if col not in drop_cols and col != target
    ]

    # Strict no-leakage version
    no_leakage_features = [
        col for col in all_features
        if col not in leakage_cols
    ]

    X_all = df[all_features]
    X_strict = df[no_leakage_features]
    y = df[target]

    print(f"  All-feature model features      : {len(all_features)}")
    print(f"  No-leakage model features       : {len(no_leakage_features)}")
    print("  Leakage-risk features removed in strict setting:")
    for col in leakage_cols:
        if col in all_features:
            print(f"    • {col}")

    return X_all, X_strict, y, all_features, no_leakage_features


# ═════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════
def get_models():
    section("5.2", "Defining Models")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
        print("  XGBoost available: included in model comparison")
    else:
        print("  XGBoost not installed: skipping XGBoost")
        print("  To install: pip install xgboost")

    print("  Models:")
    for name in models:
        print(f"    • {name}")

    return models


# ═════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_set_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "feature_set": feature_set_name,
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics, y_pred


def run_experiment(X, y, models, feature_set_name):
    section("5.3", f"Training Models — {feature_set_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    print(f"  Train size: {X_train.shape[0]}")
    print(f"  Test size : {X_test.shape[0]}")

    results = []

    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")

        metrics, y_pred = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            model_name,
            feature_set_name
        )

        results.append(metrics)

        print(
            f"    Accuracy={metrics['accuracy']:.3f} | "
            f"Precision={metrics['precision']:.3f} | "
            f"Recall={metrics['recall']:.3f} | "
            f"F1={metrics['f1']:.3f} | "
            f"ROC-AUC={metrics['roc_auc']:.3f}"
        )

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Low Development", "High Development"]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{model_name} Confusion Matrix\n({feature_set_name})")
        save_fig(
            f"confusion_matrix_{feature_set_name.lower().replace(' ', '_')}_"
            f"{model_name.lower().replace(' ', '_')}.png"
        )

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════
# CROSS VALIDATION
# ═════════════════════════════════════════════════════════════════════
def cross_validate_models(X, y, models, feature_set_name):
    section("5.4", f"Cross-Validation — {feature_set_name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for model_name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="roc_auc"
        )

        cv_results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "cv_roc_auc_mean": scores.mean(),
            "cv_roc_auc_std": scores.std(),
        })

        print(
            f"  {model_name}: "
            f"ROC-AUC = {scores.mean():.3f} ± {scores.std():.3f}"
        )

    return pd.DataFrame(cv_results)


# ═════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════
def save_feature_importance(X, y, features):
    section("5.5", "Feature Importance")

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X, y)

    importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print("  ✓ Saved → outputs/models/feature_importance.csv")

    top10 = importance.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10["feature"][::-1], top10["importance"][::-1])
    plt.title("Top 10 Feature Importances — Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    save_fig("feature_importance_top10.png")

    print("\n  Top 10 important features:")
    print(top10.to_string(index=False))

    return importance


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  PROJECT STEP 5 — MODEL TRAINING & COMPARISON")
    print("  NYC Urban Development Prediction")
    print("═" * 60)

    df = load_data()
    X_all, X_strict, y, all_features, strict_features = prepare_features(df)
    models = get_models()

    # Experiment 1: all features
    results_all = run_experiment(
        X_all,
        y,
        models,
        feature_set_name="All Features"
    )

    cv_all = cross_validate_models(
        X_all,
        y,
        models,
        feature_set_name="All Features"
    )

    # Experiment 2: no-leakage features
    results_strict = run_experiment(
        X_strict,
        y,
        models,
        feature_set_name="No Leakage Features"
    )

    cv_strict = cross_validate_models(
        X_strict,
        y,
        models,
        feature_set_name="No Leakage Features"
    )

    # Combine results
    results = pd.concat([results_all, results_strict], ignore_index=True)
    cv_results = pd.concat([cv_all, cv_strict], ignore_index=True)

    final_results = results.merge(
        cv_results,
        on=["feature_set", "model"],
        how="left"
    )

    final_results = final_results.sort_values(
        by=["feature_set", "roc_auc"],
        ascending=[True, False]
    )

    final_results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n  ✓ Saved → outputs/models/model_comparison.csv")

    print("\n" + "═" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("═" * 60)
    print(final_results.round(3).to_string(index=False))

    # Feature importance using strict features
    save_feature_importance(X_strict, y, strict_features)

    print("\n" + "═" * 60)
    print("  STEP 5 COMPLETE")
    print("═" * 60)
    print("  Outputs saved to: outputs/models/")
    print("  Main result file: outputs/models/model_comparison.csv")
    print("═" * 60)


if __name__ == "__main__":
    main()