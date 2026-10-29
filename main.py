
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Project setup
# -------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS PROJECT")
print("=" * 60)

# -------------------------------
# 2. Load dataset
# -------------------------------
try:
    df = pd.read_csv(DATA_URL)
except (URLError, OSError, ValueError) as exc:
    raise SystemExit(
        "Could not download the dataset. Check your internet "
        "connection and try again."
    ) from exc

print("\nDataset loaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Standardize column names
df.columns = [
    str(col).strip().lower().replace(" ", "_")
    for col in df.columns
]

# -------------------------------
# 3. Inspect data quality
# -------------------------------
print("\nFirst five rows:")
print(df.head())

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0])

print("\nDuplicate rows:", df.duplicated().sum())

# Save data-quality summary
quality = pd.DataFrame({
    "data_type": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_percent": (
        df.isna().mean() * 100
    ).round(2),
    "unique_values": df.nunique()
})
quality.to_csv(OUTPUT_DIR / "data_quality.csv")

# -------------------------------
# 4. Statistical summaries
# -------------------------------
numeric = df.select_dtypes(include=np.number)

summary = numeric.describe().round(2)
summary.to_csv(OUTPUT_DIR / "statistical_summary.csv")

print("\nStatistical summary:")
print(summary)

# -------------------------------
# 5. Survival analysis
# -------------------------------
if "survived" not in df.columns:
    raise SystemExit(
        "Expected 'Survived' column was not found in the dataset."
    )

survival_rate = df["survived"].mean() * 100

print(f"\nOverall survival rate: {survival_rate:.2f}%")

if "sex" in df.columns:
    by_sex = (
        df.groupby("sex", observed=True)["survived"]
        .mean().mul(100).round(2)
    )
    print("\nSurvival rate by sex (%):")
    print(by_sex)
    by_sex.rename("survival_rate_percent").to_csv(
        OUTPUT_DIR / "survival_by_sex.csv"
    )

if "pclass" in df.columns:
    by_class = (
        df.groupby("pclass", observed=True)["survived"]
        .mean().mul(100).round(2)
    )
    print("\nSurvival rate by passenger class (%):")
    print(by_class)
    by_class.rename("survival_rate_percent").to_csv(
        OUTPUT_DIR / "survival_by_class.csv"
    )

# -------------------------------
# 6. Visualization: age distribution
# -------------------------------
if "age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df, x="age", bins=25, kde=True,
        color="steelblue"
    )
    plt.title("Distribution of Passenger Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "age_distribution.png", dpi=200
    )
    plt.close()

# -------------------------------
# 7. Visualization: survival by sex
# -------------------------------
if "sex" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, x="sex", y="survived",
        errorbar=None
    )
    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate (0 to 1)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "survival_by_sex.png", dpi=200
    )
    plt.close()

# -------------------------------
# 8. Visualization: survival by class
# -------------------------------
if "pclass" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, x="pclass", y="survived",
        errorbar=None
    )
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (0 to 1)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "survival_by_class.png", dpi=200
    )
    plt.close()

# -------------------------------
# 9. Visualization: fare distribution
# -------------------------------
if "fare" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="fare")
    plt.title("Distribution of Passenger Fares")
    plt.xlabel("Fare")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fare_boxplot.png", dpi=200
    )
    plt.close()

# -------------------------------
# 10. Correlation analysis
# -------------------------------
if len(numeric.columns) >= 2:
    correlation = numeric.corr()
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png", dpi=200
    )
    plt.close()

# -------------------------------
# 11. Generate insights report
# -------------------------------
report_lines = [
    "EXPLORATORY DATA ANALYSIS REPORT",
    "=" * 40,
    "",
    f"Dataset rows: {df.shape[0]}",
    f"Dataset columns: {df.shape[1]}",
    f"Duplicate rows: {df.duplicated().sum()}",
    f"Total missing cells: {int(df.isna().sum().sum())}",
    f"Overall survival rate: {survival_rate:.2f}%",
    "",
    "KEY FINDINGS",
    "-" * 40,
]

if "sex" in df.columns:
    rates = df.groupby("sex")["survived"].mean() * 100
    if len(rates) >= 2:
        highest = rates.idxmax()
        lowest = rates.idxmin()
        report_lines.append(
            f"Survival rate was highest for {highest} "
            f"({rates[highest]:.2f}%) and lowest for "
            f"{lowest} ({rates[lowest]:.2f}%)."
        )

if "pclass" in df.columns:
    rates = df.groupby("pclass")["survived"].mean() * 100
    if not rates.empty:
        best_class = rates.idxmax()
        report_lines.append(
            f"Passenger class with the highest observed "
            f"survival rate: {best_class} "
            f"({rates[best_class]:.2f}%)."
        )

report_lines += [
    "",
    "INTERPRETATION NOTES",
    "-" * 40,
    "Correlations indicate association, not causation.",
    "Missing values may affect the interpretation of results.",
    "Findings describe this dataset and are not proof of "
    "general causal relationships.",
    "",
    "OUTPUTS",
    "-" * 40,
    "See the CSV summary tables and PNG charts in outputs/."
]

(OUTPUT_DIR / "eda_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nAnalysis completed!")
print(f"Reports and charts saved in: {OUTPUT_DIR.resolve()}")
print("Review outputs/eda_report.txt before submitting.")
