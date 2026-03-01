from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


st.set_page_config(page_title="Student Performance - 4 Model Dashboard", layout="wide")


REQUIRED_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
    "Exam_Score",
]

CAT_NORMALIZE_COLS = [
    "Parental_Education_Level",
    "Distance_from_Home",
    "Family_Income",
    "Teacher_Quality",
    "Peer_Influence",
    "Motivation_Level",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
]

ORDINAL_MAPPINGS = {
    "Parental_Involvement": {"Low": 1, "Medium": 2, "High": 3},
    "Access_to_Resources": {"Low": 1, "Medium": 2, "High": 3},
    "Motivation_Level": {"Low": 1, "Medium": 2, "High": 3},
    "Family_Income": {"Low": 1, "Medium": 2, "High": 3},
    "Teacher_Quality": {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0},
    "Parental_Education_Level": {
        "High School": 1,
        "College": 2,
        "Postgraduate": 3,
        "Unknown": 0,
    },
    "Peer_Influence": {"Negative": 1, "Neutral": 2, "Positive": 3, "Unknown": 0},
    "Distance_from_Home": {"Far": 1, "Moderate": 2, "Near": 3, "Unknown": 0},
}

BINARY_MAPPINGS = {
    "Extracurricular_Activities": {"No": 0, "Yes": 1},
    "Internet_Access": {"No": 0, "Yes": 1},
    "Learning_Disabilities": {"No": 0, "Yes": 1},
}

NOMINAL_COLS = ["School_Type", "Gender"]
OUTLIER_COLS = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]
CLUSTER_FEATURES = ["Exam_Score", "Hours_Studied", "Attendance", "Motivation_Level"]


@dataclass
class PipelineArtifacts:
    data_processed: pd.DataFrame
    X: pd.DataFrame
    X_scaled: pd.DataFrame
    y_reg: pd.Series
    y_cls: pd.Series
    scaler: StandardScaler
    linear_model: LinearRegression
    poly_transformer: PolynomialFeatures
    poly_model: LinearRegression
    logistic_model: LogisticRegression
    kmeans_pipeline: Pipeline
    metrics_regression: pd.DataFrame
    metrics_logistic: pd.DataFrame
    cluster_summary: pd.DataFrame
    predictions: pd.DataFrame
    outlier_table: pd.DataFrame



def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}



def detect_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr



def validate_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]



def preprocess_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_value)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    outlier_rows: list[dict[str, float | str]] = []
    for col in OUTLIER_COLS + ["Exam_Score"]:
        lower, upper = detect_bounds(df[col])
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_rows.append(
            {
                "feature": col,
                "outlier_count": int(outlier_count),
                "outlier_pct": float(outlier_count / len(df) * 100),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
        )

    for col in OUTLIER_COLS:
        lower, upper = detect_bounds(df[col])
        df[col] = df[col].clip(lower, upper)

    df["Exam_Score"] = df["Exam_Score"].clip(0, 100)

    for col in CAT_NORMALIZE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df.loc[df[col].str.lower() == "nan", col] = "Unknown"

    for col, mapping in ORDINAL_MAPPINGS.items():
        encoded = df[col].map(mapping)
        if encoded.isna().any():
            unseen = sorted(df.loc[encoded.isna(), col].unique().tolist())
            raise ValueError(f"{col} has unseen categories: {unseen}")
        df[col] = encoded.astype("int64")

    for col, mapping in BINARY_MAPPINGS.items():
        encoded = df[col].map(mapping)
        if encoded.isna().any():
            unseen = sorted(df.loc[encoded.isna(), col].unique().tolist())
            raise ValueError(f"{col} has unseen categories: {unseen}")
        df[col] = encoded.astype("int64")

    df = pd.get_dummies(df, columns=[c for c in NOMINAL_COLS if c in df.columns], drop_first=True)

    df["Study_Effort"] = df["Hours_Studied"]
    df["Attendance_Score"] = df["Attendance"] * df["Previous_Scores"] / 100
    df["Support_System"] = (
        df["Parental_Involvement"] + df["Access_to_Resources"] + df["Teacher_Quality"]
    ) / 3
    df["Optimal_Sleep"] = ((df["Sleep_Hours"] >= 6) & (df["Sleep_Hours"] <= 8)).astype(int)
    df["High_Intensity_Study"] = (df["Hours_Studied"] > 25).astype(int)
    df["Risk_Factor"] = ((df["Attendance"] < 70) & (df["Motivation_Level"] == 1)).astype(int)

    outlier_table = pd.DataFrame(outlier_rows)
    return df, outlier_table



def train_four_models(df_processed: pd.DataFrame, pass_threshold: int, test_size: float) -> PipelineArtifacts:
    X = df_processed.drop(columns=["Exam_Score"])
    y_reg = df_processed["Exam_Score"]
    y_cls = (y_reg >= pass_threshold).astype(int)

    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns, index=X.index)

    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X_scaled,
        y_reg,
        y_cls,
        test_size=test_size,
        random_state=42,
        stratify=None,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train_reg)
    y_pred_linear = linear_model.predict(X_test)

    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_transformer.fit_transform(X_train)
    X_test_poly = poly_transformer.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train_reg)
    y_pred_poly = poly_model.predict(X_test_poly)

    logistic_model = LogisticRegression(max_iter=1500)
    logistic_model.fit(X_train, y_train_cls)
    y_pred_log = logistic_model.predict(X_test)

    kmeans_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=3, random_state=42, n_init=10)),
        ]
    )
    kmeans_pipeline.fit(df_processed[CLUSTER_FEATURES])
    cluster_labels = kmeans_pipeline.predict(df_processed[CLUSTER_FEATURES])

    reg_linear = evaluate_regression(y_test_reg, y_pred_linear)
    reg_poly = evaluate_regression(y_test_reg, y_pred_poly)
    metrics_regression = pd.DataFrame([reg_linear, reg_poly], index=["Linear", "Polynomial_deg2"])

    metrics_logistic = pd.DataFrame(
        [
            {
                "Accuracy": accuracy_score(y_test_cls, y_pred_log),
                "Precision": precision_score(y_test_cls, y_pred_log, zero_division=0),
                "Recall": recall_score(y_test_cls, y_pred_log, zero_division=0),
                "F1": f1_score(y_test_cls, y_pred_log, zero_division=0),
                "Pass Threshold": pass_threshold,
            }
        ],
        index=["Logistic"],
    )

    cluster_summary = (
        df_processed.assign(Cluster=cluster_labels)
        .groupby("Cluster")[CLUSTER_FEATURES]
        .mean()
        .reset_index()
    )

    full_linear = linear_model.predict(X_scaled)
    full_poly = poly_model.predict(poly_transformer.transform(X_scaled))
    full_log_proba = logistic_model.predict_proba(X_scaled)[:, 1]
    full_log_pred = logistic_model.predict(X_scaled)

    predictions = df_processed.copy()
    predictions["Pred_Exam_Linear"] = full_linear
    predictions["Pred_Exam_Polynomial"] = full_poly
    predictions[f"Pred_Pass_Prob_(>={pass_threshold})"] = full_log_proba
    predictions[f"Pred_Pass_Label_(>={pass_threshold})"] = full_log_pred
    predictions["Cluster"] = cluster_labels

    return PipelineArtifacts(
        data_processed=df_processed,
        X=X,
        X_scaled=X_scaled,
        y_reg=y_reg,
        y_cls=y_cls,
        scaler=scaler,
        linear_model=linear_model,
        poly_transformer=poly_transformer,
        poly_model=poly_model,
        logistic_model=logistic_model,
        kmeans_pipeline=kmeans_pipeline,
        metrics_regression=metrics_regression,
        metrics_logistic=metrics_logistic,
        cluster_summary=cluster_summary,
        predictions=predictions,
        outlier_table=pd.DataFrame(),
    )



def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")



def render_scatter(actual: pd.Series, pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(actual, pred, alpha=0.55, edgecolor="k")
    lo = min(float(actual.min()), float(np.min(pred)))
    hi = max(float(actual.max()), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual Exam_Score")
    ax.set_ylabel("Predicted Exam_Score")
    ax.set_title(title)
    st.pyplot(fig)



def main() -> None:
    st.title("Student Performance: Raw CSV -> 4 Models Dashboard")
    st.caption(
        "Notebook-faithful pipeline: preprocessing + feature engineering + "
        "Linear, Polynomial, Logistic, and KMeans."
    )

    with st.sidebar:
        st.header("Run Config")
        st.markdown("**Recommended strategy:** Recreate and train pipeline from uploaded raw CSV.")
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        pass_threshold = st.slider("Pass Threshold for Logistic Model", min_value=40, max_value=90, value=60, step=1)

    uploaded_file = st.file_uploader("Upload raw CSV (StudentPerformanceFactors format)", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to run the full pipeline.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as err:
        st.error(f"Failed to read CSV: {err}")
        return

    missing_cols = validate_columns(raw_df)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.write(f"Rows: **{raw_df.shape[0]}**, Columns: **{raw_df.shape[1]}**")

    run = st.button("Run Full Pipeline", type="primary")
    if not run:
        return

    with st.spinner("Processing data and training 4 models..."):
        try:
            processed_df, outlier_table = preprocess_raw(raw_df)
            artifacts = train_four_models(processed_df, pass_threshold=pass_threshold, test_size=test_size)
            artifacts.outlier_table = outlier_table
        except Exception as err:
            st.error(f"Pipeline failed: {err}")
            return

    st.success("Pipeline complete.")

    t1, t2, t3, t4 = st.tabs(["Preprocessing", "Model Metrics", "Clusters", "Predictions"])

    with t1:
        st.markdown("**Outlier Detection Summary (IQR, before capping)**")
        st.dataframe(artifacts.outlier_table, use_container_width=True)

        before_missing = raw_df.isna().sum().sum()
        after_missing = processed_df.isna().sum().sum()
        c1, c2 = st.columns(2)
        c1.metric("Missing Values Before", int(before_missing))
        c2.metric("Missing Values After", int(after_missing))

        st.markdown("**Processed Data Preview**")
        st.dataframe(processed_df.head(10), use_container_width=True)

    with t2:
        st.markdown("**Regression Models (Exam_Score Prediction)**")
        st.dataframe(artifacts.metrics_regression.round(4), use_container_width=True)

        st.markdown("**Logistic Model (Pass/Fail Classification)**")
        st.dataframe(artifacts.metrics_logistic.round(4), use_container_width=True)

        reg_cols = st.columns(2)
        with reg_cols[0]:
            y_true = artifacts.y_reg
            y_pred_full_linear = artifacts.linear_model.predict(artifacts.X_scaled)
            render_scatter(y_true, y_pred_full_linear, "Linear: Actual vs Predicted")

        with reg_cols[1]:
            y_pred_full_poly = artifacts.poly_model.predict(
                artifacts.poly_transformer.transform(artifacts.X_scaled)
            )
            render_scatter(y_true, y_pred_full_poly, "Polynomial: Actual vs Predicted")

    with t3:
        st.markdown("**KMeans Cluster Summary**")
        st.dataframe(artifacts.cluster_summary.round(3), use_container_width=True)

        plot_df = artifacts.predictions[["Hours_Studied", "Attendance", "Cluster"]].copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=plot_df,
            x="Hours_Studied",
            y="Attendance",
            hue="Cluster",
            palette="Set2",
            ax=ax,
            alpha=0.7,
        )
        ax.set_title("Student Clusters: Hours_Studied vs Attendance")
        st.pyplot(fig)

    with t4:
        st.markdown("**Predictions Table**")
        st.dataframe(artifacts.predictions.head(30), use_container_width=True)
        st.download_button(
            label="Download Full Predictions CSV",
            data=to_csv_bytes(artifacts.predictions),
            file_name="student_predictions_4_models.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
