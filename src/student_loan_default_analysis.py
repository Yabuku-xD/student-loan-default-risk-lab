"""
End-to-end workflow for analyzing U.S. college-level student loan default risk.

The script:
1. Loads the latest College Scorecard institution file and state-level SAIPE data.
2. Cleans and engineers features covering academics, affordability, outcomes, and demographics.
3. Generates exploratory visualizations saved to the `figures/` directory.
4. Trains a classification model to flag high-default-risk institutions and ranks feature importance.
5. Writes curated datasets, summaries, and metrics to `data/processed/` and `reports/`.

Run from the repository root:
    python src/student_loan_default_analysis.py
"""

# Personal note: I keep this script intentionally chatty so future me remembers
# why each step exists when running the pipeline half-asleep.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid", palette="deep")

# --------------------------------------------------------------------------------------
# Configuration (aka the knobs I fiddle with most)
# TODO: externalize these paths if this ever becomes a proper package.
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

SCORECARD_PATH = DATA_RAW_DIR / "scorecard" / "public" / "Most-Recent-Cohorts-Institution.csv"
SAIPE_PATH = BASE_DIR / "data" / "external" / "saipe_est22all.xls"

# Core Scorecard columns to load (kept lean for memory). Still curating manually,
# so TODO: pull this list from metadata if DOE ever gives us a schema endpoint.
PCIP_COLUMNS = [
    "PCIP01", "PCIP03", "PCIP04", "PCIP05", "PCIP09", "PCIP10", "PCIP11", "PCIP12",
    "PCIP13", "PCIP14", "PCIP15", "PCIP16", "PCIP19", "PCIP22", "PCIP23", "PCIP24",
    "PCIP25", "PCIP26", "PCIP27", "PCIP29", "PCIP30", "PCIP31", "PCIP38", "PCIP39",
    "PCIP40", "PCIP41", "PCIP42", "PCIP43", "PCIP44", "PCIP45", "PCIP46", "PCIP47",
    "PCIP48", "PCIP49", "PCIP50", "PCIP51", "PCIP52", "PCIP54",
]

SCORECARD_COLUMNS = [
    "UNITID", "INSTNM", "CITY", "STABBR", "ZIP", "CONTROL", "PREDDEG", "HIGHDEG",
    "REGION", "LOCALE", "LOCALE2", "CURROPER", "MAIN", "ST_FIPS", "CCSIZSET",
    "CCUGPROF", "TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITIONFEE_PROG",
    "NPT4_PUB", "NPT4_PRIV", "NPT4_PROG", "COSTT4_A", "TUITFTE", "INEXPFTE",
    "AVGFACSAL", "UGDS", "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
    "UGDS_AIAN", "UGDS_NHPI", "UGDS_2MOR", "UGDS_NRA", "UGDS_UNKN",
    "PCTPELL", "PCTFLOAN", "UG25ABV", "RET_FT4", "RET_FTL4", "RET_PT4", "RET_PTL4",
    "C150_4", "C150_L4", "ADM_RATE", "SAT_AVG", "ACTCMMID",
    "MD_EARN_WNE_P6", "MD_EARN_WNE_P8", "MD_EARN_WNE_P10",
    "CDR2", "CDR3",
] + PCIP_COLUMNS

NUMERIC_COLUMNS = [
    "CONTROL", "PREDDEG", "HIGHDEG", "REGION", "LOCALE", "LOCALE2", "ST_FIPS",
    "CCSIZSET", "CCUGPROF", "TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITIONFEE_PROG",
    "NPT4_PUB", "NPT4_PRIV", "NPT4_PROG", "COSTT4_A", "TUITFTE", "INEXPFTE",
    "AVGFACSAL", "UGDS", "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
    "UGDS_AIAN", "UGDS_NHPI", "UGDS_2MOR", "UGDS_NRA", "UGDS_UNKN", "PCTPELL",
    "PCTFLOAN", "UG25ABV", "RET_FT4", "RET_FTL4", "RET_PT4", "RET_PTL4", "C150_4",
    "C150_L4", "ADM_RATE", "SAT_AVG", "ACTCMMID", "MD_EARN_WNE_P6", "MD_EARN_WNE_P8",
    "MD_EARN_WNE_P10", "CDR2", "CDR3",
] + PCIP_COLUMNS

REGION_MAP = {
    0: "U.S. Service Schools",
    1: "New England",
    2: "Mid East",
    3: "Great Lakes",
    4: "Plains",
    5: "Southeast",
    6: "Southwest",
    7: "Rocky Mountains",
    8: "Far West",
    9: "Outlying Areas",
}

CONTROL_MAP = {
    1: "Public",
    2: "Private nonprofit",
    3: "Private for-profit",
}

PREDDEG_MAP = {
    0: "Not classified",
    1: "Certificates",
    2: "Associate's",
    3: "Bachelor's",
    4: "Graduate",
}

LOCALE_SIMPLE_MAP = {
    11: "City",
    12: "City",
    13: "City",
    21: "Suburb",
    22: "Suburb",
    23: "Suburb",
    31: "Town",
    32: "Town",
    33: "Town",
    41: "Rural",
    42: "Rural",
    43: "Rural",
}

MAJOR_GROUPS: Dict[str, Sequence[str]] = {
    "STEM": ["PCIP01", "PCIP03", "PCIP11", "PCIP14", "PCIP15", "PCIP26", "PCIP27", "PCIP40", "PCIP41", "PCIP45"],
    "Health": ["PCIP51"],
    "Business": ["PCIP52"],
    "Education": ["PCIP13"],
    "Humanities": ["PCIP05", "PCIP16", "PCIP23", "PCIP24", "PCIP25", "PCIP50", "PCIP54"],
    "Social Science & Public Service": ["PCIP19", "PCIP22", "PCIP30", "PCIP42", "PCIP44", "PCIP39"],
    "Trades & Applied": ["PCIP12", "PCIP47", "PCIP48", "PCIP46", "PCIP10", "PCIP09"],
}

# Classification threshold for "high default risk" (>=2% default rate)
# If we ever change how we message risk externally, revisit this magic number.
HIGH_DEFAULT_THRESHOLD = 0.02
SCATTER_DEFAULT_CAP = 20  # cap for visualization clarity

WATCHLIST_COLUMNS = [
    "UNITID",
    "INSTNM",
    "control_label",
    "predominant_degree_label",
    "region_label",
    "locale_label",
    "dominant_program",
    "default_rate_pct",
    "net_price",
    "completion_rate",
    "PCTPELL",
]


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        # Friendly reminder to myself: brand-new environments never have these.
        path.mkdir(parents=True, exist_ok=True)


def load_scorecard_data(path: Path, columns: List[str]) -> pd.DataFrame:
    print(f"Loading College Scorecard data from {path} ...")
    # I still peek at the raw CSV every few months—this keeps memory usage sane.
    df = pd.read_csv(path, usecols=columns, low_memory=False)
    df.replace({"PrivacySuppressed": np.nan, "NULL": np.nan, "NaN": np.nan}, inplace=True)
    return df


def load_saipe_data(path: Path) -> pd.DataFrame:
    print(f"Loading SAIPE state-level data from {path} ...")
    # TODO: cache a CSV export so the Excel parsing doesn't slow down CI someday.
    saipe = pd.read_excel(path, skiprows=3)
    saipe.rename(
        columns={
            "State FIPS Code": "state_fips",
            "County FIPS Code": "county_fips",
            "Poverty Percent, All Ages": "saipe_poverty_pct",
            "Median Household Income": "saipe_mhi",
        },
        inplace=True,
    )
    saipe = saipe[saipe["county_fips"] == 0].copy()
    saipe["state_fips"] = saipe["state_fips"].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else np.nan)
    saipe["saipe_poverty_pct"] = pd.to_numeric(saipe["saipe_poverty_pct"], errors="coerce")
    saipe["saipe_mhi"] = pd.to_numeric(saipe["saipe_mhi"], errors="coerce")
    return saipe[["state_fips", "saipe_poverty_pct", "saipe_mhi"]]


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    # Little helper because pandas silently keeps things as objects otherwise.
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _compute_major_clusters(df: pd.DataFrame) -> pd.Series:
    group_shares = {}
    # Yeah, this is intentionally forgiving—datasets show up missing columns a lot.
    for group, cols in MAJOR_GROUPS.items():
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            continue
        group_shares[group] = df[existing_cols].sum(axis=1, skipna=True)
    if not group_shares:
        return pd.Series("Unknown", index=df.index)
    group_df = pd.DataFrame(group_shares)
    return group_df.idxmax(axis=1).fillna("Unknown")


def engineer_features(scorecard: pd.DataFrame, saipe: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning and engineering features ...")
    # TODO: break this beast into smaller helpers the next time we add a feature block.
    # Filter to currently operating institutions with valid default data
    scorecard = scorecard[scorecard["CURROPER"] == 1].copy()
    scorecard = _coerce_numeric(scorecard, NUMERIC_COLUMNS)

    scorecard["state_fips"] = scorecard["ST_FIPS"].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else np.nan)
    scorecard = scorecard.merge(saipe, on="state_fips", how="left")

    scorecard["default_rate"] = scorecard["CDR3"]
    scorecard = scorecard[scorecard["default_rate"].notna()].copy()
    scorecard["default_rate_pct"] = scorecard["default_rate"] * 100
    scorecard["high_default_flag"] = (scorecard["default_rate"] >= HIGH_DEFAULT_THRESHOLD).astype(int)

    tuition_stack = scorecard[["TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITIONFEE_PROG"]].copy()
    scorecard["tuition_avg"] = tuition_stack.mean(axis=1, skipna=True)

    net_price = np.select(
        [
            scorecard["CONTROL"] == 1,
            scorecard["CONTROL"] == 2,
            scorecard["CONTROL"] == 3,
        ],
        [
            scorecard["NPT4_PUB"],
            scorecard["NPT4_PRIV"],
            scorecard["NPT4_PROG"],
        ],
        default=np.nan,
    )
    scorecard["net_price"] = net_price
    scorecard["net_price"] = scorecard["net_price"].fillna(scorecard["COSTT4_A"])

    scorecard["completion_rate"] = np.where(
        scorecard["PREDDEG"].isin([3, 4]),
        scorecard["C150_4"],
        scorecard["C150_L4"],
    )

    race_cols = ["UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN", "UGDS_AIAN", "UGDS_NHPI", "UGDS_2MOR"]
    race_array = np.square(scorecard[race_cols].fillna(0))
    scorecard["diversity_index"] = 1 - race_array.sum(axis=1)

    scorecard["dominant_program"] = _compute_major_clusters(scorecard)

    scorecard["region_label"] = scorecard["REGION"].map(REGION_MAP).fillna("Other/Unknown")
    scorecard["control_label"] = scorecard["CONTROL"].map(CONTROL_MAP).fillna("Other/Unknown")
    scorecard["predominant_degree_label"] = scorecard["PREDDEG"].map(PREDDEG_MAP).fillna("Other/Unknown")
    scorecard["locale_label"] = scorecard["LOCALE"].map(LOCALE_SIMPLE_MAP).fillna("Other/Unknown")

    scorecard["saipe_mhi"] = pd.to_numeric(scorecard["saipe_mhi"], errors="coerce")
    scorecard["saipe_poverty_pct"] = pd.to_numeric(scorecard["saipe_poverty_pct"], errors="coerce")
    scorecard["cost_to_income_ratio"] = scorecard["net_price"] / scorecard["saipe_mhi"]

    scorecard["default_rate_bucket"] = pd.cut(
        scorecard["default_rate_pct"],
        bins=[-np.inf, 5, 10, 100],
        labels=["<5%", "5-10%", ">=10%"],
    )
    return scorecard


# --------------------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------------------

def _save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    # Keeping the DPI high—even if it makes git diffs noisy—has saved a few decks.
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    print("Running exploratory data analysis ...")
    outputs: List[Path] = []
    palette = sns.color_palette("colorblind")
    # Future me: feel free to swap palettes; nothing downstream depends on colors.

    def _scatter_sample(frame: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
        if len(frame) > sample_size:
            return frame.sample(sample_size, random_state=42)
        return frame.copy()

    def _cap_defaults(frame: pd.DataFrame, cap: float = SCATTER_DEFAULT_CAP) -> pd.DataFrame:
        subset = frame[frame["default_rate_pct"] <= cap].copy()
        if subset.empty:
            return frame.copy()
        return subset

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["default_rate_pct"].clip(upper=10),
        bins=30,
        ax=ax,
        color=palette[0],
        edgecolor="white",
        alpha=0.9,
    )
    ax.set_title("Distribution of 3-Year Cohort Default Rates")
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("Number of institutions")
    ax.set_xlim(left=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_distribution.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    control_palette = {"Public": "#4C72B0", "Private nonprofit": "#55A868", "Private for-profit": "#C44E52"}

    box_df = _cap_defaults(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=box_df,
        x="control_label",
        y="default_rate_pct",
        hue="control_label",
        dodge=False,
        palette=control_palette,
        legend=False,
        width=0.55,
        ax=ax,
    )
    ax.set_title("Default Rates by Control Type")
    ax.set_xlabel("Control")
    ax.set_ylabel("Default rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_by_control.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    degree_order = (
        box_df["predominant_degree_label"]
        .value_counts()
        .sort_index()
        .index
    )
    degree_palette = dict(zip(degree_order, sns.color_palette("pastel", len(degree_order))))
    sns.boxplot(
        data=box_df,
        x="predominant_degree_label",
        y="default_rate_pct",
        order=degree_order,
        hue="predominant_degree_label",
        dodge=False,
        palette=degree_palette,
        legend=False,
        width=0.55,
        ax=ax,
    )
    ax.set_title("Default Rates by Predominant Degree")
    ax.set_xlabel("Predominant degree")
    ax.set_ylabel("Default rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_by_degree.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    region_order = (
        df.groupby("region_label")["default_rate_pct"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.barplot(
        data=box_df,
        x="region_label",
        y="default_rate_pct",
        hue="region_label",
        order=region_order,
        dodge=False,
        estimator=np.mean,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Average Default Rates by Region")
    ax.set_ylabel("Average default rate (%)")
    ax.set_xlabel("Region")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_by_region.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    capped_scatter = _cap_defaults(df)
    scatter_df = _scatter_sample(capped_scatter, 2500)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(
        data=scatter_df,
        x="net_price",
        y="default_rate_pct",
        hue="control_label",
        palette=["#4C72B0", "#55A868", "#C44E52"],
        alpha=0.45,
        s=40,
        ax=ax,
    )
    ax.set_title("Net Price vs Default Rate")
    ax.set_xlabel("Average net price (USD)")
    ax.set_ylabel("Default rate (%)")
    ax.set_xlim(left=0)
    ax.legend(title="Control", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_vs_net_price.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.regplot(
        data=_scatter_sample(_cap_defaults(df.dropna(subset=["saipe_poverty_pct"])), 2500),
        x="saipe_poverty_pct",
        y="default_rate_pct",
        scatter_kws={"alpha": 0.35, "s": 35, "color": palette[1]},
        line_kws={"color": palette[0]},
        ax=ax,
    )
    ax.set_title("State Poverty vs Default Rate")
    ax.set_xlabel("State poverty rate (%)")
    ax.set_ylabel("Default rate (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_vs_poverty.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    comp_df = _cap_defaults(df)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.regplot(
        data=_scatter_sample(comp_df.dropna(subset=["completion_rate"]), 2500),
        x="completion_rate",
        y="default_rate_pct",
        scatter_kws={"alpha": 0.35, "s": 35, "color": palette[2]},
        line_kws={"color": palette[0]},
        ax=ax,
    )
    ax.set_title("Completion Rate vs Default Rate")
    ax.set_xlabel("Completion rate")
    ax.set_ylabel("Default rate (%)")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    out_path = out_dir / "default_rate_vs_completion.png"
    _save_plot(fig, out_path)
    outputs.append(out_path)

    return outputs


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["control_label", "predominant_degree_label", "dominant_program", "region_label", "locale_label"]
    # This is basically a pivot table for stakeholders who live in spreadsheets.
    summary = (
        df.groupby(group_cols)
        .agg(
            institutions=("UNITID", "count"),
            avg_default=("default_rate_pct", "mean"),
            avg_pell=("PCTPELL", "mean"),
            avg_completion=("completion_rate", "mean"),
            avg_net_price=("net_price", "mean"),
        )
        .reset_index()
    )
    summary.sort_values(by="avg_default", ascending=False, inplace=True)
    return summary


# --------------------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------------------

def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

def _build_rf_pipeline(numeric_features: List[str], categorical_features: List[str]) -> SkPipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    return SkPipeline([("preprocessor", preprocessor), ("model", clf)])


def _build_balanced_rf_pipeline(numeric_features: List[str], categorical_features: List[str]) -> SkPipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    clf = BalancedRandomForestClassifier(
        n_estimators=600,
        max_depth=12,
        random_state=42,
        sampling_strategy="auto",
    )
    return SkPipeline([("preprocessor", preprocessor), ("model", clf)])


def _build_smote_logistic_pipeline(numeric_features: List[str], categorical_features: List[str]) -> ImbPipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    sampler = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    return ImbPipeline([("preprocessor", preprocessor), ("sampler", sampler), ("model", clf)])


def _analyze_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str,
    reports_dir: Path,
) -> Dict[str, Any]:
    thresholds = np.linspace(0, 1, 101)
    rows = []
    best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append(
            {"threshold": t, "precision": precision, "recall": recall, "f1": f1}
        )
        if f1 > best["f1"]:
            best = {"threshold": float(t), "precision": precision, "recall": recall, "f1": f1}

    thresh_df = pd.DataFrame(rows)
    thresholds_path = reports_dir / f"thresholds_{model_name}.csv"
    thresh_df.to_csv(thresholds_path, index=False)

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_df = pd.DataFrame(
        {"precision": pr_precision, "recall": pr_recall, "threshold": np.append(pr_thresholds, np.nan)}
    )
    pr_path = reports_dir / f"pr_curve_{model_name}.csv"
    pr_df.to_csv(pr_path, index=False)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    roc_path = reports_dir / f"roc_curve_{model_name}.csv"
    roc_df.to_csv(roc_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pr_recall, pr_precision, color="#1f77b4")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve: {model_name}")
    pr_fig_path = FIGURES_DIR / f"pr_curve_{model_name}.png"
    fig.tight_layout()
    fig.savefig(pr_fig_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#d62728", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.legend()
    roc_fig_path = FIGURES_DIR / f"roc_curve_{model_name}.png"
    fig.tight_layout()
    fig.savefig(roc_fig_path, dpi=300)
    plt.close(fig)

    return {
        "thresholds_csv": str(thresholds_path.relative_to(BASE_DIR)),
        "pr_curve_csv": str(pr_path.relative_to(BASE_DIR)),
        "roc_curve_csv": str(roc_path.relative_to(BASE_DIR)),
        "pr_curve_png": str(pr_fig_path.relative_to(BASE_DIR)),
        "roc_curve_png": str(roc_fig_path.relative_to(BASE_DIR)),
        "recommended_threshold": best["threshold"],
        "recommended_precision": best["precision"],
        "recommended_recall": best["recall"],
        "recommended_f1": best["f1"],
    }


def _generate_watchlists(
    model_builders: Dict[str, Any],
    preprocess_args: Tuple[List[str], List[str]],
    model_df: pd.DataFrame,
    y: pd.Series,
    thresholds_info: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    watchlists: Dict[str, Dict[str, str]] = {}
    numeric_features, categorical_features = preprocess_args
    X_full = model_df[numeric_features + categorical_features]
    # Friendly heads-up: this refits models on the full dataset, so budget time for it.

    for name, builder in model_builders.items():
        pipeline = builder(*preprocess_args)
        pipeline.fit(X_full, y)

        if not hasattr(pipeline, "predict_proba"):
            continue

        scores = pipeline.predict_proba(X_full)[:, 1]
        scores_df = model_df[WATCHLIST_COLUMNS].copy()
        scores_df["risk_score"] = scores
        threshold = thresholds_info.get(name, {}).get("recommended_threshold", 0.5)
        scores_df["risk_flag"] = (scores_df["risk_score"] >= threshold).astype(int)

        scores_path = DATA_PROCESSED_DIR / f"{name}_risk_scores.parquet"
        scores_df.to_parquet(scores_path, index=False)

        watchlist_df = scores_df[scores_df["risk_flag"] == 1].sort_values("risk_score", ascending=False)
        watchlist_path = REPORTS_DIR / f"watchlist_{name}.csv"
        watchlist_df.to_csv(watchlist_path, index=False)

        watchlists[name] = {
            "scores": str(scores_path.relative_to(BASE_DIR)),
            "watchlist": str(watchlist_path.relative_to(BASE_DIR)),
        }

    return watchlists


def train_models(df: pd.DataFrame, reports_dir: Path) -> Dict[str, Any]:
    feature_numeric = [
        "tuition_avg",
        "net_price",
        "completion_rate",
        "AVGFACSAL",
        "UGDS",
        "PCTPELL",
        "PCTFLOAN",
        "UG25ABV",
        "RET_FT4",
        "RET_FTL4",
        "MD_EARN_WNE_P10",
        "saipe_mhi",
        "saipe_poverty_pct",
        "cost_to_income_ratio",
        "diversity_index",
    ]

    feature_categorical = [
        "control_label",
        "predominant_degree_label",
        "region_label",
        "locale_label",
        "dominant_program",
    ]

    available_numeric = [f for f in feature_numeric if f in df.columns]
    available_categorical = [f for f in feature_categorical if f in df.columns]

    feature_cols = available_numeric + available_categorical
    model_df = df.dropna(subset=["high_default_flag"]).copy()
    X = model_df[feature_cols]
    y = model_df["high_default_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocess_args = (available_numeric, available_categorical)
    model_builders = {
        "random_forest": _build_rf_pipeline,
        "balanced_random_forest": _build_balanced_rf_pipeline,
        "smote_logistic": _build_smote_logistic_pipeline,
    }
    # TODO: plug in a time-split validation option once we track true cohorts.

    metrics: Dict[str, Dict[str, float]] = {}
    model_artifacts: Dict[str, Dict[str, str]] = {}
    threshold_artifacts: Dict[str, Any] = {}

    for name, builder in model_builders.items():
        print(f"Training {name} ...")
        pipeline = builder(*preprocess_args)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan"),
        }

        report_path = reports_dir / f"classification_report_{name}.txt"
        report_text = classification_report(y_test, y_pred, digits=3, zero_division=0)
        report_path.write_text(report_text, encoding="utf-8")

        feature_path = None
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        estimator = pipeline.named_steps["model"]

        if hasattr(estimator, "feature_importances_"):
            df_imp = pd.DataFrame(
                {"feature": feature_names, "importance": estimator.feature_importances_}
            ).sort_values(by="importance", ascending=False)
            feature_path = reports_dir / f"feature_importances_{name}.csv"
            df_imp.to_csv(feature_path, index=False)
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_.ravel()
            df_coef = pd.DataFrame(
                {"feature": feature_names, "coefficient": coef}
            ).sort_values(by="coefficient", ascending=False)
            feature_path = reports_dir / f"feature_importances_{name}.csv"
            df_coef.to_csv(feature_path, index=False)

        model_artifacts[name] = {
            "report": str(report_path.relative_to(BASE_DIR)),
            "features": str(feature_path.relative_to(BASE_DIR)) if feature_path else "",
        }

        if y_proba is not None:
            threshold_artifacts[name] = _analyze_thresholds(y_test, y_proba, name, reports_dir)

    watchlist_artifacts = _generate_watchlists(model_builders, preprocess_args, model_df, y, threshold_artifacts)

    metrics_payload: Dict[str, Any] = {
        "overall": metrics,
        "thresholds": threshold_artifacts,
        "watchlists": watchlist_artifacts,
        "artifacts": model_artifacts,
    }
    metrics_path = reports_dir / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    return metrics_payload


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def main() -> None:
    ensure_directories([DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR])

    if not SCORECARD_PATH.exists():
        raise FileNotFoundError(
            f"Scorecard data not found at {SCORECARD_PATH}. "
            "Download the College Scorecard raw archive and extract the CSVs first."
        )
    if not SAIPE_PATH.exists():
        raise FileNotFoundError(
            f"SAIPE file not found at {SAIPE_PATH}. "
            "Download the county/state estimates Excel file to data/external/."
        )
    # Personal sanity check: both datasets are big, so be patient on first run.

    scorecard = load_scorecard_data(SCORECARD_PATH, SCORECARD_COLUMNS)
    saipe = load_saipe_data(SAIPE_PATH)

    enriched = engineer_features(scorecard, saipe)
    processed_path = DATA_PROCESSED_DIR / "scorecard_enriched.parquet"
    enriched.to_parquet(processed_path, index=False)

    eda_paths = run_eda(enriched, FIGURES_DIR)
    cluster_summary = summarize_clusters(enriched)
    cluster_summary.to_csv(REPORTS_DIR / "high_risk_clusters.csv", index=False)

    model_results = train_models(enriched, REPORTS_DIR)

    artifact_manifest = {
        "processed_dataset": str(processed_path.relative_to(BASE_DIR)),
        "figures": [str(path.relative_to(BASE_DIR)) for path in eda_paths],
        "cluster_summary": str((REPORTS_DIR / "high_risk_clusters.csv").relative_to(BASE_DIR)),
        "model_results": model_results,
        "metrics_path": str((REPORTS_DIR / "model_metrics.json").relative_to(BASE_DIR)),
    }
    (REPORTS_DIR / "pipeline_artifacts.json").write_text(
        json.dumps(artifact_manifest, indent=2),
        encoding="utf-8",
    )

    print("Pipeline complete.")
    print(json.dumps(model_results, indent=2))


if __name__ == "__main__":
    main()

