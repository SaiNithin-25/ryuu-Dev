from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from metrics_logger import (
    build_run_snapshot,
    discover_run_dirs,
    infer_checkpoint_dir,
    load_checkpoint_index,
    load_scalar_frame,
    pivot_scalar_frame,
    summarize_available_metrics,
)

PALETTE = {
    "ink": "#17324D",
    "ink_soft": "#4F667A",
    "sand": "#F7F1E3",
    "paper": "#FFF9F1",
    "teal": "#0F766E",
    "sky": "#3FA7D6",
    "coral": "#D1603D",
    "gold": "#E9A33B",
    "mist": "#DCE8EC",
    "grid": "rgba(23, 50, 77, 0.10)",
}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=False, ttl=20)
def load_dashboard_data(
    log_dir: str,
    checkpoint_dir: str | None,
    max_steps: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    frame = load_scalar_frame(log_dir)
    wide = pivot_scalar_frame(frame)
    snapshot = build_run_snapshot(frame, checkpoint_dir=checkpoint_dir, max_steps=max_steps)
    checkpoints = load_checkpoint_index(checkpoint_dir)
    return frame, wide, snapshot, checkpoints


def inject_styles() -> None:
    st.set_page_config(
        page_title="Ryuu Training Observatory",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: {PALETTE["ink"]};
            --ink-soft: {PALETTE["ink_soft"]};
            --sand: {PALETTE["sand"]};
            --paper: {PALETTE["paper"]};
            --teal: {PALETTE["teal"]};
            --sky: {PALETTE["sky"]};
            --coral: {PALETTE["coral"]};
            --gold: {PALETTE["gold"]};
            --mist: {PALETTE["mist"]};
        }}
        .stApp {{
            background:
                radial-gradient(circle at 12% 18%, rgba(63, 167, 214, 0.18), transparent 26%),
                radial-gradient(circle at 86% 14%, rgba(233, 163, 59, 0.18), transparent 24%),
                radial-gradient(circle at 78% 78%, rgba(209, 96, 61, 0.12), transparent 24%),
                linear-gradient(180deg, #fffaf3 0%, #f7f1e3 52%, #f2eadb 100%);
            color: var(--ink);
            font-family: "Bahnschrift", "Trebuchet MS", "Segoe UI", sans-serif;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(255, 249, 241, 0.96), rgba(247, 241, 227, 0.96));
            border-right: 1px solid rgba(23, 50, 77, 0.08);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--ink);
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        .hero-panel {{
            position: relative;
            overflow: hidden;
            padding: 1.8rem 2rem 1.5rem 2rem;
            border: 1px solid rgba(23, 50, 77, 0.08);
            border-radius: 28px;
            background:
                radial-gradient(circle at 8% 18%, rgba(63, 167, 214, 0.18), transparent 24%),
                radial-gradient(circle at 82% 14%, rgba(233, 163, 59, 0.16), transparent 20%),
                linear-gradient(140deg, rgba(255, 255, 255, 0.94), rgba(250, 244, 232, 0.92));
            box-shadow: 0 28px 60px rgba(23, 50, 77, 0.10);
            margin-bottom: 1rem;
        }}
        .hero-kicker {{
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--teal);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .hero-title {{
            font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
            font-size: 2.4rem;
            line-height: 1.05;
            color: var(--ink);
            margin: 0;
        }}
        .hero-subtitle {{
            font-size: 1rem;
            color: var(--ink-soft);
            max-width: 48rem;
            margin: 0.75rem 0 0 0;
        }}
        .hero-footer {{
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.42rem 0.9rem;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid rgba(23, 50, 77, 0.08);
        }}
        .status-live {{
            background: rgba(15, 118, 110, 0.12);
            color: var(--teal);
        }}
        .status-recent {{
            background: rgba(233, 163, 59, 0.12);
            color: #946100;
        }}
        .status-stale, .status-missing {{
            background: rgba(209, 96, 61, 0.10);
            color: var(--coral);
        }}
        .metric-card {{
            height: 100%;
            padding: 1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(23, 50, 77, 0.08);
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 16px 30px rgba(23, 50, 77, 0.07);
            backdrop-filter: blur(10px);
        }}
        .metric-label {{
            color: var(--ink-soft);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.55rem;
        }}
        .metric-value {{
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }}
        .metric-hint {{
            color: var(--ink-soft);
            font-size: 0.88rem;
        }}
        .metric-accent-teal {{
            box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.09), 0 16px 30px rgba(15, 118, 110, 0.08);
        }}
        .metric-accent-coral {{
            box-shadow: inset 0 0 0 1px rgba(209, 96, 61, 0.09), 0 16px 30px rgba(209, 96, 61, 0.08);
        }}
        .metric-accent-gold {{
            box-shadow: inset 0 0 0 1px rgba(233, 163, 59, 0.09), 0 16px 30px rgba(233, 163, 59, 0.08);
        }}
        .section-note {{
            color: var(--ink-soft);
            font-size: 0.92rem;
            margin-top: -0.4rem;
            margin-bottom: 0.7rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str, accent: str = "teal") -> str:
    return f"""
    <div class="metric-card metric-accent-{accent}">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-hint">{html.escape(hint)}</div>
    </div>
    """


def base_figure(title: str, height: int = 370) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=18, color=PALETTE["ink"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="Bahnschrift, Trebuchet MS, Segoe UI, sans-serif", color=PALETTE["ink"]),
        margin=dict(l=20, r=20, t=56, b=20),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        title_font=dict(size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        title_font=dict(size=12),
    )
    return fig


def loss_figure(wide: pd.DataFrame, snapshot: dict) -> go.Figure:
    fig = base_figure("Loss Arc", height=420)

    if "train/loss_ema" in wide.columns:
        train = wide[["step", "train/loss_ema"]].dropna()
        if not train.empty:
            fig.add_trace(
                go.Scatter(
                    x=train["step"],
                    y=train["train/loss_ema"],
                    mode="lines+markers",
                    name="Train EMA",
                    line=dict(color=PALETTE["sky"], width=3),
                    marker=dict(size=6),
                )
            )

    if "val/loss" in wide.columns:
        val = wide[["step", "val/loss"]].dropna()
        if not val.empty:
            fig.add_trace(
                go.Scatter(
                    x=val["step"],
                    y=val["val/loss"],
                    mode="lines+markers",
                    name="Val Loss",
                    line=dict(color=PALETTE["coral"], width=3),
                    marker=dict(size=7),
                    fill="tozeroy",
                    fillcolor="rgba(209, 96, 61, 0.08)",
                )
            )

    best_step = snapshot.get("best_step")
    best_val = snapshot.get("best_val_loss")
    if best_step is not None and best_val is not None:
        fig.add_vline(x=best_step, line_dash="dot", line_color=PALETTE["gold"], opacity=0.9)
        fig.add_annotation(
            x=best_step,
            y=best_val,
            text=f"Best {best_val:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=PALETTE["gold"],
            bgcolor="rgba(255, 249, 241, 0.95)",
            bordercolor="rgba(233, 163, 59, 0.35)",
        )

    fig.update_xaxes(title="Optimizer Step")
    fig.update_yaxes(title="Loss")
    return fig


def throughput_figure(wide: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text="Throughput Pulse", x=0.01, xanchor="left", font=dict(size=18, color=PALETTE["ink"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="Bahnschrift, Trebuchet MS, Segoe UI, sans-serif", color=PALETTE["ink"]),
        margin=dict(l=20, r=20, t=56, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    bar_data = wide[["step", "perf/tok_per_sec"]].dropna() if "perf/tok_per_sec" in wide.columns else pd.DataFrame()
    line_data = wide[["step", "perf/tok_per_step"]].dropna() if "perf/tok_per_step" in wide.columns else pd.DataFrame()

    if not bar_data.empty:
        fig.add_trace(
            go.Bar(
                x=bar_data["step"],
                y=bar_data["perf/tok_per_sec"],
                name="Tok / sec",
                marker_color="rgba(15, 118, 110, 0.75)",
                hovertemplate="Step %{x}<br>%{y:,.0f} tok/s<extra></extra>",
            ),
            secondary_y=False,
        )

    if not line_data.empty:
        fig.add_trace(
            go.Scatter(
                x=line_data["step"],
                y=line_data["perf/tok_per_step"],
                mode="lines+markers",
                name="Tok / step",
                line=dict(color=PALETTE["gold"], width=3),
                marker=dict(size=7),
                hovertemplate="Step %{x}<br>%{y:,.0f} tok/step<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_xaxes(title="Optimizer Step", showgrid=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text="Tokens / second", secondary_y=False, showgrid=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text="Tokens / step", secondary_y=True, showgrid=False)
    return fig


def reasoning_figure(wide: pd.DataFrame) -> go.Figure:
    fig = base_figure("Reasoning Signal", height=380)
    added_trace = False

    if "train/reason_ema" in wide.columns:
        train = wide[["step", "train/reason_ema"]].dropna()
        if not train.empty:
            fig.add_trace(
                go.Scatter(
                    x=train["step"],
                    y=train["train/reason_ema"],
                    mode="lines+markers",
                    name="Train Reason EMA",
                    line=dict(color=PALETTE["teal"], width=3),
                    marker=dict(size=6),
                )
            )
            added_trace = True

    if "val/reason" in wide.columns:
        val = wide[["step", "val/reason"]].dropna()
        if not val.empty:
            fig.add_trace(
                go.Scatter(
                    x=val["step"],
                    y=val["val/reason"],
                    mode="lines+markers",
                    name="Val Reason",
                    line=dict(color=PALETTE["coral"], width=3, dash="dot"),
                    marker=dict(size=7),
                )
            )
            added_trace = True

    if not added_trace:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Reasoning metrics have not been logged yet.",
            showarrow=False,
            font=dict(size=15, color=PALETTE["ink_soft"]),
        )

    fig.update_xaxes(title="Optimizer Step")
    fig.update_yaxes(title="Reasoning Loss")
    return fig


def phase_figure(wide: pd.DataFrame) -> go.Figure:
    fig = base_figure("Efficiency Phase Map", height=420)

    required = {"val/loss", "perf/tok_per_sec"}
    if not required.issubset(set(wide.columns)):
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Need val/loss and tok/sec metrics to draw the phase map.",
            showarrow=False,
            font=dict(size=15, color=PALETTE["ink_soft"]),
        )
        return fig

    phase = wide.copy()
    phase = phase.dropna(subset=["val/loss", "perf/tok_per_sec"])
    if phase.empty:
        return fig

    marker_size = (
        phase["perf/tokens_k"].fillna(phase["perf/tok_per_step"].fillna(1.0)).clip(lower=1.0) ** 0.5
    ) * 3.2

    fig.add_trace(
        go.Scatter(
            x=phase["perf/tok_per_sec"],
            y=phase["val/loss"],
            mode="lines",
            name="Trajectory",
            line=dict(color="rgba(23, 50, 77, 0.25)", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=phase["perf/tok_per_sec"],
            y=phase["val/loss"],
            mode="markers",
            name="Eval point",
            marker=dict(
                size=marker_size,
                color=phase["step"],
                colorscale=[
                    [0.0, PALETTE["sky"]],
                    [0.5, PALETTE["gold"]],
                    [1.0, PALETTE["coral"]],
                ],
                line=dict(color="rgba(255,255,255,0.9)", width=1.2),
                showscale=True,
                colorbar=dict(title="Step", thickness=12),
            ),
            text=[f"Step {int(step)}" for step in phase["step"]],
            hovertemplate="%{text}<br>%{x:,.0f} tok/s<br>Val loss %{y:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Tokens / second")
    fig.update_yaxes(title="Validation loss")
    return fig


def completion_figure(snapshot: dict) -> go.Figure:
    value = snapshot.get("completion_ratio")
    latest_step = snapshot.get("latest_step") or 0
    max_steps = snapshot.get("max_steps") or 0

    if value is None:
        fig = go.Figure(
            go.Indicator(
                mode="number",
                value=latest_step,
                number={"font": {"size": 44}, "valueformat": ",d"},
                title={"text": "Latest Step"},
            )
        )
    else:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value * 100.0,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Completion"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PALETTE["teal"], "thickness": 0.36},
                    "bgcolor": "rgba(23, 50, 77, 0.08)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(63, 167, 214, 0.18)"},
                        {"range": [40, 75], "color": "rgba(233, 163, 59, 0.18)"},
                        {"range": [75, 100], "color": "rgba(15, 118, 110, 0.18)"},
                    ],
                },
            )
        )
        fig.add_annotation(
            x=0.5,
            y=0.06,
            xref="paper",
            yref="paper",
            text=f"{latest_step:,} / {max_steps:,} steps",
            showarrow=False,
            font=dict(size=14, color=PALETTE["ink_soft"]),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="Bahnschrift, Trebuchet MS, Segoe UI, sans-serif", color=PALETTE["ink"]),
        margin=dict(l=20, r=20, t=30, b=10),
        height=250,
    )
    return fig


def checkpoint_figure(checkpoints: pd.DataFrame, snapshot: dict) -> go.Figure:
    fig = base_figure("Checkpoint Trail", height=330)

    step_rows = checkpoints[checkpoints["kind"] == "step"].dropna(subset=["step"]) if not checkpoints.empty else pd.DataFrame()
    if not step_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=step_rows["step"],
                y=["Checkpoint"] * len(step_rows),
                mode="markers+text",
                text=step_rows["name"],
                textposition="top center",
                marker=dict(
                    size=14,
                    color=PALETTE["sky"],
                    line=dict(color="rgba(255,255,255,0.92)", width=1.5),
                    symbol="diamond",
                ),
                hovertemplate="%{text}<br>Step %{x:,}<br>%{customdata[0]} MB<extra></extra>",
                customdata=step_rows[["size_mb"]].to_numpy(),
                name="Saved step",
            )
        )

    if snapshot.get("best_checkpoint_present") and snapshot.get("best_step") is not None:
        best_step = snapshot["best_step"]
        best_loss = snapshot.get("best_val_loss")
        label = f"Best checkpoint @ {best_step:,}"
        if best_loss is not None:
            label = f"{label}<br>Val loss {best_loss:.4f}"
        fig.add_trace(
            go.Scatter(
                x=[best_step],
                y=["Best"],
                mode="markers+text",
                text=[label],
                textposition="top center",
                marker=dict(
                    size=18,
                    color=PALETTE["gold"],
                    line=dict(color="rgba(255,255,255,0.92)", width=1.5),
                    symbol="star",
                ),
                hovertemplate=f"{label}<extra></extra>",
                name="Best checkpoint",
            )
        )

    fig.update_xaxes(title="Optimizer Step")
    fig.update_yaxes(title="", showgrid=False)
    return fig


def format_float(value: float | None, decimals: int = 4, fallback: str = "--") -> str:
    if value is None:
        return fallback
    return f"{value:,.{decimals}f}"


def format_compact_number(value: float | None, fallback: str = "--") -> str:
    if value is None:
        return fallback
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def format_timestamp(timestamp_value: float | None) -> str:
    if timestamp_value is None:
        return "--"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")


def status_sentence(snapshot: dict, run_name: str) -> str:
    latest_step = snapshot.get("latest_step")
    best_step = snapshot.get("best_step")
    best_val = snapshot.get("best_val_loss")
    last_update = format_timestamp(snapshot.get("last_update_ts"))

    parts = [f"{run_name} is currently {snapshot.get('freshness', 'missing')}."]
    if latest_step is not None:
        parts.append(f"Latest logged step: {latest_step:,}.")
    if best_step is not None and best_val is not None:
        parts.append(f"Best validation loss {best_val:.4f} at step {best_step:,}.")
    if last_update != "--":
        parts.append(f"Last event written at {last_update}.")
    return " ".join(parts)


def render_overview(snapshot: dict, wide: pd.DataFrame) -> None:
    top_row = st.columns(3)
    with top_row[0]:
        st.markdown(
            metric_card(
                "Latest Step",
                format_compact_number(snapshot.get("latest_step")),
                f"Freshness: {snapshot.get('freshness', 'missing')}",
                "teal",
            ),
            unsafe_allow_html=True,
        )
    with top_row[1]:
        best_hint = "Lowest recorded validation loss"
        if snapshot.get("best_step") is not None:
            best_hint = f"Reached at step {snapshot['best_step']:,}"
        st.markdown(
            metric_card("Best Val Loss", format_float(snapshot.get("best_val_loss")), best_hint, "coral"),
            unsafe_allow_html=True,
        )
    with top_row[2]:
        throughput = snapshot.get("tok_per_sec")
        tok_per_step = snapshot.get("tok_per_step")
        hint = f"{format_compact_number(tok_per_step)} tok/step"
        st.markdown(
            metric_card("Throughput", f"{format_compact_number(throughput)} tok/s", hint, "gold"),
            unsafe_allow_html=True,
        )

    bottom_row = st.columns(3)
    with bottom_row[0]:
        hint = f"Train vs val delta: {format_float(snapshot.get('generalization_gap'), 3)}"
        st.markdown(
            metric_card("Train Perplexity", format_float(snapshot.get("train_ppl"), 2), hint, "teal"),
            unsafe_allow_html=True,
        )
    with bottom_row[1]:
        improvement = snapshot.get("val_improvement_pct")
        hint = "From first logged validation point"
        if improvement is not None:
            hint = f"{improvement:+.2f}% vs first validation checkpoint"
        st.markdown(
            metric_card(
                "Tokens Seen",
                format_compact_number(snapshot.get("total_tokens")),
                hint,
                "coral",
            ),
            unsafe_allow_html=True,
        )
    with bottom_row[2]:
        eta_text = format_duration(snapshot.get("eta_seconds"))
        hint = f"Last update: {format_timestamp(snapshot.get('last_update_ts'))}"
        st.markdown(metric_card("ETA", eta_text, hint, "gold"), unsafe_allow_html=True)

    chart_left, chart_right = st.columns([2.2, 1.0])
    with chart_left:
        st.plotly_chart(loss_figure(wide, snapshot), use_container_width=True, config=PLOTLY_CONFIG)
    with chart_right:
        st.plotly_chart(completion_figure(snapshot), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown(
            '<div class="section-note">The completion gauge uses the target max steps from the sidebar, so you can retarget runs without touching code.</div>',
            unsafe_allow_html=True,
        )

    st.plotly_chart(phase_figure(wide), use_container_width=True, config=PLOTLY_CONFIG)


def render_dynamics(wide: pd.DataFrame, frame: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(throughput_figure(wide), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        st.plotly_chart(reasoning_figure(wide), use_container_width=True, config=PLOTLY_CONFIG)

    available = summarize_available_metrics(frame)
    metrics_table = pd.DataFrame({"Metric Tag": available})
    st.markdown(
        '<div class="section-note">These are the scalar tags detected in the selected run. Missing charts usually mean the trainer has not emitted that tag yet.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)


def render_checkpoints(checkpoints: pd.DataFrame, snapshot: dict) -> None:
    if checkpoints.empty:
        st.info("No checkpoint files were found for the selected run.")
        return

    st.plotly_chart(checkpoint_figure(checkpoints, snapshot), use_container_width=True, config=PLOTLY_CONFIG)
    display = checkpoints.copy()
    if "updated_ts" in display.columns:
        display = display.drop(columns=["updated_ts"])
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()

    st.sidebar.markdown("## Flight Controls")
    run_dirs = discover_run_dirs("runs")
    if not run_dirs:
        st.error("No TensorBoard runs were found under `runs/`.")
        st.stop()

    options = [str(path) for path in run_dirs]
    default_index = next((index for index, item in enumerate(options) if Path(item).name == "v3_reasoning"), 0)
    selected_run = st.sidebar.selectbox(
        "Run log directory",
        options,
        index=default_index,
        format_func=lambda item: Path(item).name,
    )

    guessed_checkpoint_dir = infer_checkpoint_dir(selected_run)
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint directory",
        value=str(guessed_checkpoint_dir) if guessed_checkpoint_dir else "",
    ).strip()
    max_steps = st.sidebar.number_input("Target max steps", min_value=0, value=60000, step=1000)
    history_points = st.sidebar.slider("Chart window", min_value=10, max_value=500, value=250, step=10)

    if st.sidebar.button("Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    frame, wide, snapshot, checkpoints = load_dashboard_data(
        selected_run,
        checkpoint_dir or None,
        int(max_steps) if max_steps > 0 else None,
    )

    if frame.empty:
        st.warning("The selected run directory exists, but no scalar events were detected yet.")
        st.stop()

    wide = wide.tail(history_points).copy()
    run_name = Path(selected_run).name
    status_class = f"status-{snapshot.get('freshness', 'missing')}"

    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-kicker">Training Observatory</div>
            <h1 class="hero-title">Ryuu Flight Deck</h1>
            <p class="hero-subtitle">{html.escape(status_sentence(snapshot, run_name))}</p>
            <div class="hero-footer">
                <span class="status-pill {status_class}">{html.escape(snapshot.get("freshness", "missing"))}</span>
                <span class="status-pill status-recent">{html.escape(run_name)}</span>
                <span class="status-pill status-live">{len(frame):,} scalar points loaded</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, dynamics_tab, checkpoints_tab = st.tabs(["Overview", "Dynamics", "Checkpoints"])
    with overview_tab:
        render_overview(snapshot, wide)
    with dynamics_tab:
        render_dynamics(wide, frame)
    with checkpoints_tab:
        render_checkpoints(checkpoints, snapshot)


if __name__ == "__main__":
    main()
