from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from constellation.legacy.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_cols: List[str],
    title: str = "Time Series",
    ylabel: str = "Value",
    figsize: tuple = (14, 6),
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=figsize)

    for col in value_cols:
        ax.plot(df[time_col], df[col], label=col, alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame, top_n: int = 20, title: str = "Feature Importance"
) -> go.Figure:

    top_features = importance_df.head(top_n).sort_values("importance")

    fig = go.Figure(
        go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h",
            marker=dict(color=top_features["importance"], colorscale="Viridis"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, top_n * 20),
        showlegend=False,
    )

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:

    if features:
        df = df[features]

    corr = df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 8},
        )
    )

    fig.update_layout(title=title, width=800, height=800)

    return fig


def plot_distribution(
    df: pd.DataFrame, column: str, title: Optional[str] = None, bins: int = 50
) -> go.Figure:

    if title is None:
        title = f"Distribution of {column}"

    fig = px.histogram(df, x=column, nbins=bins, title=title, labels={column: column})

    fig.update_layout(showlegend=False, xaxis_title=column, yaxis_title="Count")

    return fig


def plot_anomaly_scores(
    df: pd.DataFrame,
    time_col: str,
    score_col: str,
    threshold: float,
    title: str = "Anomaly Scores",
) -> go.Figure:

    fig = go.Figure()

    # Plot scores
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[score_col],
            mode="lines",
            name="Anomaly Score",
            line=dict(color="blue", width=1),
        )
    )

    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[df[time_col].min(), df[time_col].max()],
            y=[threshold, threshold],
            mode="lines",
            name="Threshold",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    # Highlight anomalies
    anomalies = df[df[score_col] > threshold]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies[time_col],
                y=anomalies[score_col],
                mode="markers",
                name="Anomalies",
                marker=dict(color="red", size=8, symbol="x"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        hovermode="x unified",
    )

    return fig
