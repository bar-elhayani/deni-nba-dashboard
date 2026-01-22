import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
import base64
import os

# Fixed hex radius (grid resolution). No UI slider.
HEX_SIZE = 22.0

# Court constants (NBA stats-style coordinates, inches-ish)
X_MIN, X_MAX = -250, 250
Y_MIN, Y_MAX = -50, 420

THREE_RADIUS = 237.5       # 23.75 ft * 12
CORNER_X = 220.0
CORNER_Y = 92.5


# -----------------------------
# UI helpers
# -----------------------------
def _hide_xy_axes(fig: go.Figure) -> None:
    fig.update_xaxes(
        title="",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        showline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        title="",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        showline=False,
        fixedrange=True,
    )
    fig.update_layout(dragmode=False)


# -----------------------------
# Court lines (no filled shapes that can cover markers)
# -----------------------------
def add_halfcourt_shapes(fig: go.Figure) -> go.Figure:
    # Court boundary
    fig.add_shape(
        type="rect",
        x0=X_MIN, x1=X_MAX, y0=Y_MIN, y1=Y_MAX,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above"
    )

    # Paint (key)
    fig.add_shape(
        type="rect",
        x0=-80, x1=80, y0=Y_MIN, y1=140,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above"
    )

    # Hoop (rim)
    fig.add_shape(
        type="circle",
        x0=-7.5, x1=7.5, y0=-7.5, y1=7.5,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above"
    )

    # Backboard
    fig.add_shape(
        type="line",
        x0=-30, x1=30, y0=-12, y1=-12,
        line=dict(width=3, color="rgba(20,20,20,1)"),
        layer="above"
    )

    # Free-throw circle (approx)
    fig.add_shape(
        type="circle",
        x0=-60, x1=60, y0=80, y1=200,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above"
    )

    # Restricted area (approx)
    fig.add_shape(
        type="circle",
        x0=-40, x1=40, y0=-5, y1=75,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above"
    )

    # Three-point line (corner lines + arc)
    fig = add_three_point_line(fig)
    return fig


def add_three_point_line(fig: go.Figure) -> go.Figure:
    # Corner 3 lines
    for sx in (-1.0, 1.0):
        fig.add_trace(
            go.Scatter(
                x=[sx * CORNER_X, sx * CORNER_X],
                y=[Y_MIN, CORNER_Y],
                mode="lines",
                line=dict(width=2, color="rgba(20,20,20,1)"),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Arc between the corner endpoints (centered at hoop (0,0))
    theta_left = np.arctan2(CORNER_Y, -CORNER_X)
    theta_right = np.arctan2(CORNER_Y, CORNER_X)
    thetas = np.linspace(theta_right, theta_left, 220)

    xs = THREE_RADIUS * np.cos(thetas)
    ys = THREE_RADIUS * np.sin(thetas)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=2, color="rgba(20,20,20,1)"),
            hoverinfo="skip",
            showlegend=False
        )
    )
    return fig


# -----------------------------
# Hex grid math (flat-top axial coordinates)
# -----------------------------
def xy_to_axial(x: np.ndarray, y: np.ndarray, size: float):
    q = (2.0 / 3.0) * x / size
    r = ((-1.0 / 3.0) * x + (np.sqrt(3) / 3.0) * y) / size
    return q, r


def axial_round(q: np.ndarray, r: np.ndarray):
    cube_x = q
    cube_z = r
    cube_y = -cube_x - cube_z

    rx = np.round(cube_x)
    ry = np.round(cube_y)
    rz = np.round(cube_z)

    x_diff = np.abs(rx - cube_x)
    y_diff = np.abs(ry - cube_y)
    z_diff = np.abs(rz - cube_z)

    mask_x = (x_diff > y_diff) & (x_diff > z_diff)
    mask_y = (y_diff > z_diff)

    rx[mask_x] = -ry[mask_x] - rz[mask_x]
    ry[~mask_x & mask_y] = -rx[~mask_x & mask_y] - rz[~mask_x & mask_y]
    rz[~mask_x & ~mask_y] = -rx[~mask_x & ~mask_y] - ry[~mask_x & ~mask_y]

    return rx.astype(int), rz.astype(int)


def axial_to_xy(q: np.ndarray, r: np.ndarray, size: float):
    x = size * (1.5 * q)
    y = size * (np.sqrt(3) * (r + q / 2.0))
    return x, y


# -----------------------------
# Build a full grid that covers the half-court rectangle
# -----------------------------
def build_full_hex_grid(size: float) -> pd.DataFrame:
    q_max = int(np.ceil(max(abs(X_MIN), abs(X_MAX)) / (1.5 * size))) + 3
    r_max = int(np.ceil(Y_MAX / (np.sqrt(3) * size))) + q_max + 3
    r_min = int(np.floor(Y_MIN / (np.sqrt(3) * size))) - q_max - 3

    qs, rs = [], []
    for q in range(-q_max, q_max + 1):
        for r in range(r_min, r_max + 1):
            qs.append(q)
            rs.append(r)

    qs = np.array(qs, dtype=int)
    rs = np.array(rs, dtype=int)
    xs, ys = axial_to_xy(qs, rs, size)

    mask = (xs >= X_MIN) & (xs <= X_MAX) & (ys >= Y_MIN) & (ys <= Y_MAX)

    grid = pd.DataFrame({
        "Q": qs[mask],
        "R": rs[mask],
        "X_CENTER": xs[mask],
        "Y_CENTER": ys[mask],
    })
    grid["HEX_ID"] = grid["Q"].astype(str) + "_" + grid["R"].astype(str)
    return grid


# -----------------------------
# Aggregate shots into hex bins (FGA, FGM)
# -----------------------------
def aggregate_shots(df: pd.DataFrame, size: float) -> pd.DataFrame:
    x = df["LOC_X"].to_numpy()
    y = df["LOC_Y"].to_numpy()

    q, r = xy_to_axial(x, y, size)
    q_i, r_i = axial_round(q, r)

    tmp = df.copy()
    tmp["Q"] = q_i
    tmp["R"] = r_i

    g = tmp.groupby(["Q", "R"], as_index=False).agg(
        FGA=("SHOT_MADE_FLAG", "count"),
        FGM=("SHOT_MADE_FLAG", "sum"),
    )
    g["HEX_ID"] = g["Q"].astype(str) + "_" + g["R"].astype(str)
    return g


# -----------------------------
# Discrete colorscale for bins
# -----------------------------
def discrete_colorscale(colors):
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        left = i / n
        right = (i + 1) / n
        scale.append([left, c])
        scale.append([right, c])
    scale[0][0] = 0.0
    scale[-1][0] = 1.0
    return scale


# -----------------------------
# HEX MAP
# -----------------------------
def make_hex_map(full_hex: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    z = full_hex[full_hex["FGA"] == 0].copy()
    nz = full_hex[full_hex["FGA"] > 0].copy()
    hex_symbol = "hexagon"

    hover_tmpl = (
        "Shot Attempts: %{customdata[1]}<br>"
        "Made Shots: %{customdata[2]}<br>"
        "Point Percentage: %{customdata[3]:.2%}<extra></extra>"
    )

    z_customdata = [[hid, 0, 0, 0.0] for hid in z["HEX_ID"].to_numpy()]

    fig.add_trace(
        go.Scatter(
            x=z["X_CENTER"],
            y=z["Y_CENTER"],
            mode="markers",
            marker=dict(
                symbol=hex_symbol,
                size=3,
                color="rgba(255,255,255,0.30)",
                line=dict(width=0.6, color="rgba(20,20,20,0.25)"),
            ),
            customdata=z_customdata,
            hovertemplate=hover_tmpl,
            showlegend=False
        )
    )

    wood = "rgb(242, 226, 198)"

    if nz.empty:
        fig.update_layout(
            height=720,
            xaxis=dict(range=[-260, 260]),
            yaxis=dict(range=[-50, 420]),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor=wood,
            paper_bgcolor=wood,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        _hide_xy_axes(fig)
        fig = add_halfcourt_shapes(fig)
        return fig

    sizes = np.clip(8 + 5.0 * nz["FGA"].to_numpy(), 10, 70)

    max_bin = 8
    fgm_bins = np.minimum(nz["FGM"].to_numpy(), max_bin)

    bin_labels = [str(i) for i in range(max_bin)] + [f"{max_bin}+"]
    colors = (
        "rgba(251,133,0,0.88)",
        "rgba(244,162,97,0.88)",
        "rgba(91,124,153,0.88)",
        "rgba(2,48,71,0.88)",
    )

    cs = discrete_colorscale(colors)

    fg_ratio = nz["FGM"].to_numpy() / nz["FGA"].to_numpy()

    nz_customdata = [
        [hid, int(fga), int(fgm), float(r)]
        for hid, fga, fgm, r in zip(
            nz["HEX_ID"].to_numpy(),
            nz["FGA"].to_numpy(),
            nz["FGM"].to_numpy(),
            fg_ratio
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=nz["X_CENTER"],
            y=nz["Y_CENTER"],
            mode="markers",
            marker=dict(
                symbol=hex_symbol,
                size=sizes,
                color=fgm_bins,
                cmin=0,
                cmax=max_bin,
                colorscale=cs,
                colorbar=dict(
                    title="FGM (Makes)",
                    tickmode="array",
                    tickvals=list(range(0, max_bin + 1)),
                    ticktext=bin_labels,
                    x=0.75,
                    thickness=18,
                    len=0.70,
                ),
                line=dict(width=1.2, color="rgba(20,20,20,0.75)"),
                opacity=1.0
            ),
            customdata=nz_customdata,
            hovertemplate=hover_tmpl,
            showlegend=False
        )
    )

    fig.update_layout(
        height=720,
        xaxis=dict(range=[-260, 260]),
        yaxis=dict(range=[-50, 420]),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=wood,
        paper_bgcolor=wood,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _hide_xy_axes(fig)
    fig = add_halfcourt_shapes(fig)
    return fig


# -----------------------------
# ZONE REGIONS: geometry
# -----------------------------
def zone_shapes():
    shapes = {}

    def rect_poly(x0, x1, y0, y1):
        return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]

    def circle_poly(cx, cy, r, n=180):
        t = np.linspace(0, 2 * np.pi, n)
        xs = (cx + r * np.cos(t)).tolist()
        ys = (cy + r * np.sin(t)).tolist()
        return xs + [xs[0]], ys + [ys[0]]

    # Restricted Area
    shapes["Restricted Area"] = dict(kind="circle", x0=-40, x1=40, y0=-5, y1=75, label_xy=(0, 35))

    # In The Paint (Non-RA)
    shapes["In The Paint (Non-RA)"] = dict(kind="rect", x0=-80, x1=80, y0=Y_MIN, y1=140, label_xy=(0, 115))

    # Left / Right Corner 3
    shapes["Left Corner 3"] = dict(kind="rect", x0=X_MIN, x1=-CORNER_X, y0=Y_MIN, y1=CORNER_Y, label_xy=(-235, 35))
    shapes["Right Corner 3"] = dict(kind="rect", x0=CORNER_X, x1=X_MAX, y0=Y_MIN, y1=CORNER_Y, label_xy=(235, 35))

    # Above the Break 3 (path)
    theta_left = np.arctan2(CORNER_Y, -CORNER_X)
    theta_right = np.arctan2(CORNER_Y, CORNER_X)
    thetas = np.linspace(theta_right, theta_left, 260)
    arc_x = THREE_RADIUS * np.cos(thetas)
    arc_y = THREE_RADIUS * np.sin(thetas)

    path = f"M {CORNER_X} {CORNER_Y} "
    path += f"L {X_MAX} {CORNER_Y} L {X_MAX} {Y_MAX} L {X_MIN} {Y_MAX} L {X_MIN} {CORNER_Y} L {-CORNER_X} {CORNER_Y} "
    for x, y in zip(arc_x[::-1], arc_y[::-1]):
        path += f"L {x:.3f} {y:.3f} "
    path += "Z"
    shapes["Above the Break 3"] = dict(kind="path", path=path, label_xy=(0, 360))

    # Mid-Range (inside arc above paint top)
    y_base = 140.0
    x_at_y = float(np.sqrt(max(THREE_RADIUS ** 2 - y_base ** 2, 0.0)))
    theta_l = np.arctan2(y_base, -x_at_y)
    theta_r = np.arctan2(y_base, x_at_y)
    thetas2 = np.linspace(theta_r, theta_l, 220)
    xs = THREE_RADIUS * np.cos(thetas2)
    ys = THREE_RADIUS * np.sin(thetas2)

    path2 = f"M {-x_at_y:.3f} {y_base:.3f} "
    for x, y in zip(xs[::-1], ys[::-1]):
        path2 += f"L {x:.3f} {y:.3f} "
    path2 += f"L {x_at_y:.3f} {y_base:.3f} Z"
    shapes["Mid-Range"] = dict(kind="path", path=path2, label_xy=(0, 210))

    # Backcourt (optional)
    shapes["Backcourt"] = dict(kind="rect", x0=X_MIN, x1=X_MAX, y0=400, y1=Y_MAX, label_xy=(0, 410))

    return shapes


# -----------------------------
# Mode 1: Zone regions (interactive, FG% color + click overlay)
# -----------------------------
def make_zone_map(f: pd.DataFrame, selected_zone: str | None = None) -> tuple[go.Figure, pd.DataFrame]:
    z = f.groupby("SHOT_ZONE_BASIC", as_index=False).agg(
        FGA=("SHOT_MADE_FLAG", "count"),
        FGM=("SHOT_MADE_FLAG", "sum"),
    )
    z["FG%"] = np.where(z["FGA"] > 0, z["FGM"] / z["FGA"], np.nan)

    shapes = zone_shapes()
    known = z[z["SHOT_ZONE_BASIC"].isin(shapes.keys())].copy()
    unknown = z[~z["SHOT_ZONE_BASIC"].isin(shapes.keys())].copy()

    wood = "rgb(242, 226, 198)"
    fig = go.Figure()
    fig.update_layout(
        height=720,
        xaxis=dict(range=[-260, 260], zeroline=False, showgrid=False, title=""),
        yaxis=dict(range=[Y_MIN, Y_MAX], zeroline=False, showgrid=False, title=""),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=wood,
        paper_bgcolor=wood,
        clickmode="event+select",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Colorbar for FG%
    colorscale = [
        [0.0, "#fb8500"],  # Tiger Orange
        [1.0, "#023047"],  # Deep Space Blue
    ]
    fg_vals = known["FG%"].to_numpy()
    valid = fg_vals[np.isfinite(fg_vals)]

    if valid.size == 0:
        fg_min, fg_max = 0.0, 1.0
    else:
        fg_min = float(np.quantile(valid, 0.05))
        fg_max = float(np.quantile(valid, 0.95))

    # safety padding
    if fg_min == fg_max:
        fg_min = max(0.0, fg_min - 0.05)
        fg_max = min(1.0, fg_max + 0.05)

    if fg_min == fg_max:
        fg_min = max(0.0, fg_min - 0.05)
        fg_max = min(1.0, fg_max + 0.05)

    # Dummy invisible trace just to show colorbar
    fig.add_trace(
        go.Scatter(
            x=[1000], y=[1000],
            mode="markers",
            marker=dict(
                size=10,
                color=[fg_min, fg_max],
                cmin=fg_min,
                cmax=fg_max,
                colorscale=colorscale,
                colorbar=dict(
                    title="FG% (Zone)",
                    tickformat=".0%",
                    x=0.75,
                    thickness=18,
                    len=0.70,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.0,
        )
    )

    # Softer fills
    FILL_ALPHA = 0.35

    def fg_to_rgba(v: float) -> str:
        if not np.isfinite(v):
            return f"rgba(200,200,200,{FILL_ALPHA})"
        t = 0.0 if fg_max == fg_min else (v - fg_min) / (fg_max - fg_min)
        t = float(np.clip(t, 0.0, 1.0))
        col = pc.sample_colorscale(colorscale, [t])[0]  # 'rgb(r,g,b)'
        return col.replace("rgb(", "rgba(").replace(")", f",{FILL_ALPHA})")

    max_fga = int(max(known["FGA"].max(), 1)) if not known.empty else 1

    for _, row in known.iterrows():
        zone = row["SHOT_ZONE_BASIC"]
        fga = int(row["FGA"])
        fgm = int(row["FGM"])
        fg = float(row["FG%"]) if fga > 0 else np.nan

        info = shapes[zone]
        fill = fg_to_rgba(fg)

        # Filled region
        if info["kind"] == "rect":
            fig.add_shape(
                type="rect",
                x0=info["x0"], x1=info["x1"], y0=info["y0"], y1=info["y1"],
                line=dict(width=1.6, color="rgba(20,20,20,0.85)"),
                fillcolor=fill,
                layer="below",
            )
        elif info["kind"] == "circle":
            fig.add_shape(
                type="circle",
                x0=info["x0"], x1=info["x1"], y0=info["y0"], y1=info["y1"],
                line=dict(width=1.6, color="rgba(20,20,20,0.85)"),
                fillcolor=fill,
                layer="below",
            )
        else:
            fig.add_shape(
                type="path",
                path=info["path"],
                line=dict(width=1.6, color="rgba(20,20,20,0.85)"),
                fillcolor=fill,
                layer="below",
            )

        # Mid-Range wings fix
        if zone == "Mid-Range":
            fig.add_shape(
                type="rect",
                x0=-CORNER_X, x1=-80, y0=Y_MIN, y1=140,
                line=dict(width=1.2, color="rgba(20,20,20,0.60)"),
                fillcolor=fill,
                layer="below",
            )
            fig.add_shape(
                type="rect",
                x0=80, x1=CORNER_X, y0=Y_MIN, y1=140,
                line=dict(width=1.2, color="rgba(20,20,20,0.60)"),
                fillcolor=fill,
                layer="below",
            )

        lx, ly = info["label_xy"]
        click_y = ly - 35

        # Finger hint
        fig.add_annotation(
            x=lx,
            y=click_y + 12,
            text="👆",
            showarrow=False,
            font=dict(size=18),
            opacity=0.90,
        )

        # Clickable marker (NOT transparent!)
        # If this is transparent, Streamlit often won't fire selection.
        bubble_size = 10 + 34 * np.sqrt(fga / max_fga)

        fig.add_trace(
            go.Scatter(
                x=[lx],
                y=[click_y],
                mode="markers",
                marker=dict(
                    size=bubble_size,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1, color="rgba(0,0,0,0)"),
                ),
                customdata=[[zone, fga, fgm, (fg if np.isfinite(fg) else 0.0)]],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "Shot Attempts: %{customdata[1]}<br>"
                    "Made Shots: %{customdata[2]}<br>"
                    "Point Percentage: %{customdata[3]:.2%}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig = add_halfcourt_shapes(fig)

    # Overlay raw shots for selected zone
    if selected_zone is not None:
        sel = f[f["SHOT_ZONE_BASIC"] == selected_zone].copy()
        if not sel.empty:
            made = sel[sel["SHOT_MADE_FLAG"] == 1]
            miss = sel[sel["SHOT_MADE_FLAG"] == 0]

            if not miss.empty:
                fig.add_trace(
                    go.Scatter(
                        x=miss["LOC_X"],
                        y=miss["LOC_Y"],
                        mode="markers",
                        marker=dict(
                            size=7,
                            color="rgb(200,0,0)",
                            line=dict(width=0.5, color="rgba(0,0,0,0.5)"),
                        ),
                        showlegend=False,
                        hovertemplate="Miss<extra></extra>",
                    )
                )
            if not made.empty:
                fig.add_trace(
                    go.Scatter(
                        x=made["LOC_X"],
                        y=made["LOC_Y"],
                        mode="markers",
                        marker=dict(
                            size=7,
                            color="rgb(0,160,0)",
                            line=dict(width=0.5, color="rgba(0,0,0,0.5)"),
                        ),
                        showlegend=False,
                        hovertemplate="Made<extra></extra>",
                    )
                )
    _hide_xy_axes(fig)

    return fig, unknown

def _read_b64(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _add_wood_background_percent_only(fig: go.Figure, x0=-250, x1=250, y0=-50, y1=420) -> None:
    """
    Adds a wood-court background image ONLY for Zone shots (percent).
    Put an image at: data/images/wood_court.png
    """
    wood_path = os.path.join("data", "images", "wood_court.png")
    b64 = _read_b64(wood_path)

    if b64:
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{b64}",
                xref="x",
                yref="y",
                x=x0,
                y=y1,
                sizex=(x1 - x0),
                sizey=(y1 - y0),
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )
    else:
        # fallback if the file is missing
        fig.update_layout(plot_bgcolor="rgb(235,214,177)", paper_bgcolor="rgb(235,214,177)")


def add_discrete_percent_legend(
    fig: go.Figure,
    *,
    title: str = "Shot Share",
    bins=((0, 5), (5, 15), (15, 30), (30, 100)),
    colors=(
        "rgba(214,64,64,0.88)",   # red
        "rgba(224,196,64,0.88)",  # yellow
        "rgba(110,170,150,0.88)", # teal
        "rgba(64,164,96,0.88)",   # green
    ),
) -> None:
    """
    Adds a right-side discrete color legend (0-100%) using a dummy Scatter trace.
    The map itself stays unchanged.
    """
    n = len(colors)

    # Build a discrete colorscale over z in [0, n-1]
    cs = discrete_colorscale(list(colors))

    # Tick positions at bin centers (0..n-1)
    tickvals = list(range(n))
    ticktext = [f"{a}–{b}%" for a, b in bins]

    fig.add_trace(
        go.Scatter(
            x=[1000], y=[1000],
            mode="markers",
            marker=dict(
                size=12,
                color=tickvals,   # just to "activate" all bins
                cmin=0,
                cmax=n - 1,
                colorscale=cs,
                colorbar=dict(
                    title=title,
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    ticks="outside",
                    len=0.70,
                    thickness=18,
                    x=0.75,  # push a bit to the right
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.0,  # invisible on court
        )
    )




# -----------------------------
# Mode 2: Zone share labels (percent only on the court)
# -----------------------------
def make_zone_share_label_map(f: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    z = f.groupby("SHOT_ZONE_BASIC", as_index=False).agg(
        FGA=("SHOT_MADE_FLAG", "count"),
        FGM=("SHOT_MADE_FLAG", "sum"),
    )

    shapes = zone_shapes()
    known = z[z["SHOT_ZONE_BASIC"].isin(shapes.keys())].copy()
    unknown = z[~z["SHOT_ZONE_BASIC"].isin(shapes.keys())].copy()

    # Compute share of shot attempts (FGA) out of total attempts
    total_fga = float(known["FGA"].sum()) if not known.empty else 0.0
    known["SHARE"] = (known["FGA"] / total_fga) if total_fga > 0 else 0.0

    fig = go.Figure()
    fig.update_layout(
        height=720,
        xaxis=dict(range=[-260, 260]),
        yaxis=dict(range=[Y_MIN, Y_MAX]),
        margin=dict(l=10, r=50, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _hide_xy_axes(fig)

    # NBA-like wood background (percent-only view)
    _add_wood_background_percent_only(fig, x0=X_MIN, x1=X_MAX, y0=Y_MIN, y1=Y_MAX)

    # Discrete NBA-like coloring by SHARE (4 colors)
    def fg_to_color(v: float) -> str:
        if not np.isfinite(v):
            return "rgba(140,140,140,0.35)"

        if v < 0.05:
            return "rgba(251,133,0,0.88)"  # (<5%)

        if v < 0.15:
            return "rgba(244,162,97,0.88)"  # (5–15%)

        if v < 0.30:
            return "rgba(91,124,153,0.88)"

        return "rgba(2,48,71,0.88)"  # (30%+)


    # Right-side legend for the discrete % colors
    add_discrete_percent_legend(
        fig,
        title="Shot Share (%)",
        bins=((0, 5), (5, 15), (15, 30), (30, 100)),
        colors=(
            "rgba(251,133,0,0.88)",
            "rgba(244,162,97,0.88)",
            "rgba(91,124,153,0.88)",
            "rgba(2,48,71,0.88)",
        ),
    )


    for _, row in known.iterrows():
        zone = row["SHOT_ZONE_BASIC"]
        fga = int(row["FGA"])
        share = float(row["SHARE"]) if np.isfinite(row["SHARE"]) else np.nan

        info = shapes[zone]
        fill = fg_to_color(share)

        # Zone filled region (STRONG fill like NBA)
        if info["kind"] == "rect":
            fig.add_shape(
                type="rect",
                x0=info["x0"], x1=info["x1"], y0=info["y0"], y1=info["y1"],
                line=dict(width=2.0, color="rgba(0,0,0,0.75)"),
                fillcolor=fill,
                layer="above",
            )
        elif info["kind"] == "circle":
            fig.add_shape(
                type="circle",
                x0=info["x0"], x1=info["x1"], y0=info["y0"], y1=info["y1"],
                line=dict(width=2.0, color="rgba(0,0,0,0.75)"),
                fillcolor=fill,
                layer="above",
            )
        else:
            fig.add_shape(
                type="path",
                path=info["path"],
                line=dict(width=2.0, color="rgba(0,0,0,0.75)"),
                fillcolor=fill,
                layer="above",
            )

        # Mid-Range wings fix (keep same geometry behavior)
        if zone == "Mid-Range":
            fig.add_shape(
                type="rect",
                x0=-CORNER_X, x1=-80, y0=Y_MIN, y1=140,
                line=dict(width=2.0, color="rgba(0,0,0,0.75)"),
                fillcolor=fill,
                layer="above",
            )
            fig.add_shape(
                type="rect",
                x0=80, x1=CORNER_X, y0=Y_MIN, y1=140,
                line=dict(width=2.0, color="rgba(0,0,0,0.75)"),
                fillcolor=fill,
                layer="above",
            )

        lx, ly = info["label_xy"]

        pct_txt = "—" if not np.isfinite(share) else f"{share*100:.2f}%"

        # Label shows attempts + share (NOT makes / FG%)
        fig.add_annotation(
            x=lx,
            y=ly,
            text=f"{fga} attempts<br>{pct_txt} of shots",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            align="center",
            font=dict(size=14, color="white", family="Arial"),
            bgcolor="rgba(80,80,80,0.55)",
            bordercolor="rgba(255,255,255,0.20)",
            borderwidth=0,
            borderpad=4,
            opacity=1.0,
        )

    # Draw court lines AFTER zones so they appear above (white lines)
    fig = add_halfcourt_shapes(fig)

    # Turn court lines to white (only affects this fig because shapes are inside it)
    new_shapes = []
    for s in (fig.layout.shapes or []):
        s = s.to_plotly_json()
        if "line" in s and "color" in s["line"]:
            s["line"]["color"] = "rgba(255,255,255,0.95)"
        new_shapes.append(s)
    fig.update_layout(shapes=new_shapes)

    return fig, unknown

# -----------------------------
# STREAMLIT ENTRY
# -----------------------------
def render_shot_map(shots_df: pd.DataFrame) -> None:
    st.header("Shot Map – Deni Avdija")
    st.caption(
        "### This page explores where Deni Avdija takes his shots from on the court.\n\n"
        "### It helps identify his most common shooting areas, his efficiency in different zones, and how his shot selection changes across game situations."
    )

    df = shots_df.copy()
    df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce")
    df["LOC_X"] = pd.to_numeric(df["LOC_X"], errors="coerce")
    df["LOC_Y"] = pd.to_numeric(df["LOC_Y"], errors="coerce")
    df["SHOT_MADE_FLAG"] = pd.to_numeric(df["SHOT_MADE_FLAG"], errors="coerce")

    df = df.dropna(subset=["SEASON", "PERIOD", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG"])
    df["PERIOD"] = df["PERIOD"].astype(int)
    df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

    seasons = sorted(df["SEASON"].unique().tolist())
    if len(seasons) == 0:
        st.warning("No seasons found in data.")
        return

    c1, c2, c3 = st.columns([3, 2, 3])
    with c1:
        selected_seasons = st.multiselect("Season(s)", seasons, default=[seasons[-1]])
    with c2:
        quarter_choice = st.selectbox("Quarter + Total", ["Total", 1, 2, 3, 4], index=0)
    with c3:
        view = st.radio(
            "View",
            ["Hex bins", "Zone shots", "Zone shots (percent)"],
            horizontal=True,
            index=0
        )

    if quarter_choice == "Total":
        f = df[df["SEASON"].isin(selected_seasons)].copy()
    else:
        f = df[(df["SEASON"].isin(selected_seasons)) & (df["PERIOD"] == int(quarter_choice))].copy()

    if f.empty:
        st.warning("No shots for selected filters.")
        return

    total_fga = int(f.shape[0])
    total_fgm = int(f["SHOT_MADE_FLAG"].sum())
    fg_pct = total_fgm / total_fga if total_fga > 0 else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Shot Attempts", f"{total_fga:,}")
    k2.metric("Made Shots", f"{total_fgm:,}")
    k3.metric("FG%", f"{fg_pct:.3f}")

    if view == "Hex bins":
        grid = build_full_hex_grid(HEX_SIZE)
        agg = aggregate_shots(f, HEX_SIZE)

        full_hex = grid.merge(agg[["HEX_ID", "FGA", "FGM"]], on="HEX_ID", how="left")
        full_hex["FGA"] = full_hex["FGA"].fillna(0).astype(int)
        full_hex["FGM"] = full_hex["FGM"].fillna(0).astype(int)

        fig = make_hex_map(full_hex)
        st.plotly_chart(fig, width="stretch", on_select="rerun", key="shot_hex_map_full")
        return

    # Zone-based views
    if "SHOT_ZONE_BASIC" not in f.columns:
        st.warning("SHOT_ZONE_BASIC column is missing, cannot show Zone views.")
        return

    if view == "Zone shots (percent)":
        fig, unknown = make_zone_share_label_map(f)
        st.plotly_chart(fig, width="stretch", key="shot_zone_share_labels")
        if not unknown.empty:
            st.subheader("Zones not drawn (no matching geometry)")
            st.dataframe(unknown, use_container_width=True)
        return

    # Interactive zone regions (click to overlay)
    if "zone_plot_nonce" not in st.session_state:
        st.session_state["zone_plot_nonce"] = 0
    if "selected_zone_basic" not in st.session_state:
        st.session_state["selected_zone_basic"] = None

    st.caption("Click inside a zone bubble to show all shots (green=made, red=missed).")

    if st.button("Clear selection"):
        st.session_state["selected_zone_basic"] = None
        st.session_state["zone_plot_nonce"] += 1
        st.rerun()

    plot_key = f"shot_zone_map_{st.session_state['zone_plot_nonce']}"

    selected_zone = st.session_state.get("selected_zone_basic", None)
    fig, unknown = make_zone_map(f, selected_zone=selected_zone)

    st.plotly_chart(fig, width="stretch", on_select="rerun", key=plot_key)

    selection = st.session_state.get(plot_key, {}).get("selection", None)
    new_selected = None
    if selection and "points" in selection and len(selection["points"]) > 0:
        cd = selection["points"][0].get("customdata", None)
        if cd and len(cd) > 0:
            new_selected = cd[0]

    if new_selected is not None and new_selected != st.session_state.get("selected_zone_basic", None):
        st.session_state["selected_zone_basic"] = new_selected
        st.rerun()

    if not unknown.empty:
        st.subheader("Zones not drawn (no matching geometry)")
        st.dataframe(unknown, use_container_width=True)
