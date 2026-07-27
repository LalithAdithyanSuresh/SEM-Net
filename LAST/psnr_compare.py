"""
PSNR Comparison Dashboard
=========================
Compares two SEM-Net training runs by plotting PSNR vs iteration in real-time.

Models:
  A — Baseline  : ./log_inpaint.dat
  B — Updated Spiral : ./updated_spiral/log_inpaint.dat

Run:
  python psnr_compare.py
Then open http://localhost:8055 in your browser.
"""

import os
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ── Log file paths ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_A_PATH = os.path.join(BASE_DIR, "log_inpaint.dat")
LOG_B_PATH = os.path.join(BASE_DIR, "updated_spiral", "log_inpaint.dat")

MODEL_A_LABEL = "Baseline (checkpoints_c2)"
MODEL_B_LABEL = "Updated Spiral (live)"

# ── Log parser ─────────────────────────────────────────────────────────────────
# Format: epoch  iteration  gen_loss  dis_loss  sym_loss  psnr  mae
def parse_log(path: str):
    """Return (iterations, psnr_values) as numpy arrays. Skips malformed lines."""
    iterations, psnrs = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    itr  = int(parts[1])
                    psnr = float(parts[5])
                    # Sanity-check: PSNR must be a reasonable value
                    if 5.0 < psnr < 60.0:
                        iterations.append(itr)
                        psnrs.append(psnr)
                except (ValueError, IndexError):
                    continue
    except OSError:
        pass
    return np.array(iterations, dtype=np.float32), np.array(psnrs, dtype=np.float32)


def bucket_average(iters, psnrs, n: int):
    """
    Average every N consecutive points into one.
    Returns (averaged_iters, averaged_psnrs).
    """
    if n <= 1 or len(iters) == 0:
        return iters, psnrs
    num_buckets = len(iters) // n
    if num_buckets == 0:
        return iters, psnrs
    trim = num_buckets * n
    avg_iters = iters[:trim].reshape(num_buckets, n).mean(axis=1)
    avg_psnrs = psnrs[:trim].reshape(num_buckets, n).mean(axis=1)
    return avg_iters, avg_psnrs


# ── Dash App ───────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="PSNR Comparison")

app.layout = html.Div(
    style={
        "backgroundColor": "#0d1117",
        "minHeight": "100vh",
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        "padding": "24px",
        "color": "#e6edf3",
    },
    children=[

        # ── Title ──────────────────────────────────────────────────────────────
        html.H1(
            "📈 PSNR Training Comparison",
            style={"textAlign": "center", "color": "#58a6ff",
                   "marginBottom": "4px", "fontSize": "26px"},
        ),
        html.P(
            f"Live comparison — refreshes every 5 s",
            style={"textAlign": "center", "color": "#8b949e",
                   "marginTop": "0", "marginBottom": "20px", "fontSize": "13px"},
        ),

        # ── Controls bar ───────────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex", "alignItems": "center", "gap": "16px",
                "backgroundColor": "#161b22", "borderRadius": "8px",
                "padding": "12px 20px", "marginBottom": "20px",
                "border": "1px solid #30363d",
            },
            children=[
                html.Label("Avg window (N):", style={"color": "#8b949e", "fontSize": "14px", "whiteSpace": "nowrap"}),
                dcc.Input(
                    id="avg-n",
                    type="number",
                    value=50,
                    min=1,
                    max=5000,
                    step=1,
                    debounce=True,
                    style={
                        "width": "90px", "padding": "6px 10px",
                        "backgroundColor": "#0d1117", "color": "#e6edf3",
                        "border": "1px solid #30363d", "borderRadius": "6px",
                        "fontSize": "14px",
                    },
                ),
                html.Span("points per dot", style={"color": "#8b949e", "fontSize": "13px"}),

                html.Div(style={"flex": "1"}),   # spacer

                # Status indicators
                html.Div(id="status-a",
                         style={"fontSize": "12px", "color": "#3fb950", "whiteSpace": "nowrap"}),
                html.Div(id="status-b",
                         style={"fontSize": "12px", "color": "#f78166", "whiteSpace": "nowrap"}),
            ],
        ),

        # ── Chart ──────────────────────────────────────────────────────────────
        dcc.Graph(
            id="psnr-chart",
            style={"height": "65vh"},
            config={"displayModeBar": True, "scrollZoom": True},
        ),

        # ── Stats row ──────────────────────────────────────────────────────────
        html.Div(
            id="stats-row",
            style={
                "display": "flex", "gap": "16px", "marginTop": "16px",
                "flexWrap": "wrap",
            },
        ),

        # ── Live refresh interval ───────────────────────────────────────────────
        dcc.Interval(id="interval", interval=5_000, n_intervals=0),
    ],
)


def stat_card(label: str, value: str, color: str):
    return html.Div(
        style={
            "flex": "1", "minWidth": "150px",
            "backgroundColor": "#161b22", "borderRadius": "8px",
            "padding": "12px 16px", "border": f"1px solid {color}33",
            "textAlign": "center",
        },
        children=[
            html.Div(label, style={"color": "#8b949e", "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(value, style={"color": color, "fontSize": "20px", "fontWeight": "700"}),
        ],
    )


@app.callback(
    Output("psnr-chart", "figure"),
    Output("status-a", "children"),
    Output("status-b", "children"),
    Output("stats-row", "children"),
    Input("interval", "n_intervals"),
    Input("avg-n", "value"),
)
def update(_, n_raw):
    n = max(1, int(n_raw or 1))

    iters_a, psnrs_a = parse_log(LOG_A_PATH)
    iters_b, psnrs_b = parse_log(LOG_B_PATH)

    ai_a, ap_a = bucket_average(iters_a, psnrs_a, n)
    ai_b, ap_b = bucket_average(iters_b, psnrs_b, n)

    fig = go.Figure()

    # Model A trace
    if len(ai_a):
        fig.add_trace(go.Scatter(
            x=ai_a, y=ap_a,
            name=MODEL_A_LABEL,
            mode="lines",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            hovertemplate="Iter %{x:.0f}<br>PSNR %{y:.2f} dB<extra>Baseline</extra>",
        ))

    # Model B trace  
    if len(ai_b):
        fig.add_trace(go.Scatter(
            x=ai_b, y=ap_b,
            name=MODEL_B_LABEL,
            mode="lines",
            line=dict(color="#3fb950", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(63,185,80,0.08)",
            hovertemplate="Iter %{x:.0f}<br>PSNR %{y:.2f} dB<extra>Updated Spiral</extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="Inter, sans-serif"),
        xaxis=dict(
            title="Iteration",
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            tickformat=",",
        ),
        yaxis=dict(
            title="PSNR (dB)",
            gridcolor="#21262d",
            zerolinecolor="#30363d",
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=13),
        ),
        margin=dict(l=60, r=20, t=20, b=60),
        hovermode="x unified",
    )

    # Status strings
    def fmt_status(label, iters, psnrs, color_char):
        if len(iters) == 0:
            return f"⚠ {label}: file not found"
        return f"{'🟢' if color_char == 'g' else '🔵'} {label}: {len(iters)} pts | last iter {int(iters[-1])}"

    st_a = fmt_status(MODEL_A_LABEL, iters_a, psnrs_a, "b")
    st_b = fmt_status(MODEL_B_LABEL, iters_b, psnrs_b, "g")

    # Stats cards
    def safe_stat(arr):
        if len(arr) == 0:
            return "—", "—", "—"
        return f"{arr.max():.2f} dB", f"{arr.mean():.2f} dB", f"{arr[-1]:.2f} dB"

    ma, ea, la = safe_stat(psnrs_a)
    mb, eb, lb = safe_stat(psnrs_b)

    cards = [
        stat_card("Baseline — Peak PSNR",   ma, "#58a6ff"),
        stat_card("Baseline — Mean PSNR",   ea, "#58a6ff"),
        stat_card("Baseline — Latest PSNR", la, "#58a6ff"),
        stat_card("Spiral — Peak PSNR",     mb, "#3fb950"),
        stat_card("Spiral — Mean PSNR",     eb, "#3fb950"),
        stat_card("Spiral — Latest PSNR",   lb, "#3fb950"),
    ]

    return fig, st_a, st_b, cards


if __name__ == "__main__":
    print("=" * 55)
    print("  PSNR Comparison Dashboard")
    print(f"  Model A : {LOG_A_PATH}")
    print(f"  Model B : {LOG_B_PATH}")
    print("=" * 55)
    print("  Open  →  http://localhost:8055")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=8055)
