import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================================================
# IDs / Images
# =========================================================
PLAYER_ID_MAP = {
    "deni avdija": 1630166,
    "lebron james": 2544,
    "omri casspi": 201956,
}


def nba_headshot_url(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


# =========================================================
# Court constants
# =========================================================
X_MIN, X_MAX = -250, 250
Y_MIN, Y_MAX = -50, 420

THREE_RADIUS = 237.5
CORNER_X = 220.0
CORNER_Y = 92.5

DENI_CANONICAL_SEASON = "2025-26"


# =========================================================
# Rankings (league-wide CSV lookups)
# =========================================================
def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def _norm_metric(m: str) -> str:
    return _norm_str(m).upper()


def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _detect_season_col(df: pd.DataFrame) -> str | None:
    return _find_first_col(df, ["SEASON", "season", "Season", "SEASON_ID", "YEAR", "year"])


def _detect_type_col(df: pd.DataFrame) -> str | None:
    return _find_first_col(df, ["DATA_TYPE", "data_type", "TYPE", "type"])


def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _get_season_from_row(row: pd.Series) -> str | None:
    for col in ["SEASON", "season", "Season", "SEASON_ID", "YEAR", "year"]:
        if col in row.index:
            v = _norm_str(row[col])
            if v:
                return v
    return None


def _resolve_rankings_path(filename: str) -> str:
    """
    Resolve CSV path robustly:
    1) same folder as this component file (components/)
    2) project root (one level up)
    3) data/ under project root
    4) current working dir
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))

    candidates = [
        os.path.join(here, filename),
        os.path.join(project_root, filename),
        os.path.join(project_root, "data", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data", filename),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not find rankings CSV '{filename}'. Tried:\n" + "\n".join(candidates)
    )


@st.cache_data(show_spinner=False)
def _load_rankings_lookup(filename: str = "deni_season_rankings.csv"):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "data", filename)

    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    ...
    df.columns = [str(c).strip().upper() for c in df.columns]

    req = ["SEASON", "METRIC", "ABS_RANK", "TOTAL_PLAYERS", "PERCENTILE"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Rankings CSV missing columns: {missing} in {csv_path}")

    df["SEASON"] = df["SEASON"].astype(str).str.strip()
    df["METRIC"] = df["METRIC"].astype(str).str.strip().str.upper()

    df["ABS_RANK"] = pd.to_numeric(df["ABS_RANK"], errors="coerce")
    df["TOTAL_PLAYERS"] = pd.to_numeric(df["TOTAL_PLAYERS"], errors="coerce")
    df["PERCENTILE"] = pd.to_numeric(df["PERCENTILE"], errors="coerce")

    df = df.dropna(subset=["SEASON", "METRIC", "ABS_RANK", "TOTAL_PLAYERS", "PERCENTILE"]).copy()

    lookup: dict[tuple[str, str], tuple[int, int, float]] = {}

    for _, r in df.iterrows():
        season = str(r["SEASON"]).strip()
        metric = str(r["METRIC"]).strip().upper()

        abs_rank = int(float(r["ABS_RANK"]))
        total = int(float(r["TOTAL_PLAYERS"]))

        pct = float(r["PERCENTILE"])
        if pct <= 1.5:
            pct *= 100.0
        pct = max(0.0, min(100.0, pct))

        lookup[(season, metric)] = (abs_rank, total, pct)

    return lookup


def _ranking_line(
    rankings_lookup: dict[tuple[str, str], tuple[int, int, float]] | None,
    season: str | None,
    metric: str,
) -> str:
    if not rankings_lookup or not season:
        return ""
    key = (str(season).strip(), _norm_metric(metric))
    if key not in rankings_lookup:
        return ""
    abs_rank, total, pct = rankings_lookup[key]
    return f"Rank {abs_rank} out of {total}"


# =========================================================
# Court drawing
# =========================================================
def _add_three_point_line(fig: go.Figure) -> go.Figure:
    for sx in (-1.0, 1.0):
        fig.add_trace(
            go.Scatter(
                x=[sx * CORNER_X, sx * CORNER_X],
                y=[Y_MIN, CORNER_Y],
                mode="lines",
                line=dict(width=2, color="rgba(20,20,20,1)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

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
            showlegend=False,
        )
    )
    return fig


def _add_halfcourt_shapes(fig: go.Figure) -> go.Figure:
    fig.add_shape(
        type="rect",
        x0=X_MIN,
        x1=X_MAX,
        y0=Y_MIN,
        y1=Y_MAX,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above",
    )

    fig.add_shape(
        type="rect",
        x0=-80,
        x1=80,
        y0=Y_MIN,
        y1=140,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above",
    )

    fig.add_shape(
        type="circle",
        x0=-7.5,
        x1=7.5,
        y0=-7.5,
        y1=7.5,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above",
    )

    fig.add_shape(
        type="line",
        x0=-30,
        x1=30,
        y0=-12,
        y1=-12,
        line=dict(width=3, color="rgba(20,20,20,1)"),
        layer="above",
    )

    fig.add_shape(
        type="circle",
        x0=-60,
        x1=60,
        y0=80,
        y1=200,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above",
    )

    fig.add_shape(
        type="circle",
        x0=-40,
        x1=40,
        y0=-5,
        y1=75,
        line=dict(width=2, color="rgba(20,20,20,1)"),
        layer="above",
    )

    fig = _add_three_point_line(fig)
    return fig


# =========================================================
# Rows picking / meta / formatting
# =========================================================
def _player_label(row: pd.Series, fallback: str) -> str:
    for col in ["PLAYER_NAME", "player_name", "Player", "name"]:
        if col in row.index:
            v = str(row[col])
            if v and v.lower() != "nan":
                return v
    return fallback


def _meta_line(row: pd.Series) -> str:
    season = None
    team = None
    pos = None

    for col in ["SEASON", "season", "Season", "YEAR", "year"]:
        if col in row.index:
            v = str(row[col])
            if v and v.lower() != "nan":
                season = v
                break

    for col in ["TEAM", "team", "TEAM_NAME", "team_name", "Tm"]:
        if col in row.index:
            v = str(row[col])
            if v and v.lower() != "nan":
                team = v
                break

    for col in ["POS", "pos", "POSITION", "position"]:
        if col in row.index:
            v = str(row[col])
            if v and v.lower() != "nan":
                pos = v
                break

    parts = [p for p in [pos, team, season] if p]
    return " | ".join(parts)


def _fmt_stat(col: str, val: float) -> str:
    cu = col.upper()
    if "PCT" in cu or cu.endswith("%"):
        if val <= 1.5:
            return f"{val * 100:.1f}%"
        return f"{val:.1f}%"
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.2f}"


def _get_player_id_from_row(row: pd.Series, fallback_name: str | None = None) -> int | None:
    for c in ["PLAYER_ID", "player_id", "PERSON_ID", "person_id"]:
        if c in row.index:
            try:
                return int(float(row[c]))
            except Exception:
                pass

    name_val = None
    for c in ["PLAYER_NAME", "player_name", "Player", "name"]:
        if c in row.index:
            name_val = str(row[c])
            break

    key = _normalize_name(name_val) if name_val else _normalize_name(fallback_name)
    if not key:
        return None

    return PLAYER_ID_MAP.get(key)


def _pick_row_by_name(
    df: pd.DataFrame,
    target_name_substrings: list[str],
    season_prefer: str | None = None,
    type_prefer_substrings: list[str] | None = None,
) -> pd.Series | None:
    if df is None or df.empty:
        return None

    name_col = _find_first_col(df, ["PLAYER_NAME", "player_name", "Player", "name"])
    if name_col is None:
        return df.iloc[0]

    names = df[name_col].astype(str).str.lower()
    mask = np.zeros(len(df), dtype=bool)
    for sub in target_name_substrings:
        mask |= names.str.contains(sub.lower(), na=False)

    cand = df.loc[mask].copy()
    if cand.empty:
        return None

    if season_prefer is not None:
        season_col = _detect_season_col(cand)
        if season_col is not None:
            s = cand[season_col].astype(str)
            exact = cand.loc[s == str(season_prefer)]
            if not exact.empty:
                cand = exact

    if type_prefer_substrings:
        type_col = _detect_type_col(cand)
        if type_col is not None:
            t = cand[type_col].astype(str).str.lower()
            tmask = np.zeros(len(cand), dtype=bool)
            for sub in type_prefer_substrings:
                tmask |= t.str.contains(sub.lower(), na=False)
            exact = cand.loc[tmask]
            if not exact.empty:
                cand = exact

    gp_col = _find_first_col(cand, ["GP", "GAMES"])
    if gp_col is not None:
        gp = pd.to_numeric(cand[gp_col], errors="coerce").fillna(-1)
        idx = int(gp.values.argmax())
        return cand.iloc[idx]

    return cand.iloc[0]


def _pick_deni_row_fixed(deni_lebron_compare: pd.DataFrame, deni_casspi_compare: pd.DataFrame) -> pd.Series | None:
    deni_subs = ["deni", "avdija"]

    r1 = _pick_row_by_name(
        deni_lebron_compare,
        deni_subs,
        season_prefer=DENI_CANONICAL_SEASON,
        type_prefer_substrings=["season", "peak"],
    )
    if r1 is not None:
        return r1

    r2 = _pick_row_by_name(
        deni_casspi_compare,
        deni_subs,
        season_prefer=DENI_CANONICAL_SEASON,
        type_prefer_substrings=["season", "peak"],
    )
    if r2 is not None:
        return r2

    r3 = _pick_row_by_name(deni_lebron_compare, deni_subs)
    if r3 is not None:
        return r3

    return _pick_row_by_name(deni_casspi_compare, deni_subs)


def _pick_other_row(compare_df: pd.DataFrame, other_name: str) -> pd.Series | None:
    name = other_name.lower()
    if "lebron" in name:
        return _pick_row_by_name(compare_df, ["lebron", "james"])
    if "casspi" in name or "omri" in name:
        return _pick_row_by_name(compare_df, ["casspi", "omri"])
    return _pick_row_by_name(compare_df, [name])


# =========================================================
# Shots
# =========================================================
def _filter_shots_by_seasons(shots_df: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    if shots_df is None or shots_df.empty:
        return pd.DataFrame()
    season_col = _detect_season_col(shots_df)
    if season_col is None or not seasons:
        return shots_df.copy()
    s = shots_df[season_col].astype(str)
    return shots_df.loc[s.isin([str(x) for x in seasons])].copy()


def _make_shot_scatter(shots_df: pd.DataFrame) -> go.Figure:
    wood = "rgb(242, 226, 198)"
    fig = go.Figure()
    fig.update_layout(
        height=520,
        xaxis=dict(range=[-260, 260], zeroline=False, showgrid=False, title=""),
        yaxis=dict(range=[Y_MIN, Y_MAX], zeroline=False, showgrid=False, title=""),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=wood,
        paper_bgcolor=wood,
        showlegend=True,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if shots_df is None or shots_df.empty:
        return _add_halfcourt_shapes(fig)

    for c in ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"]:
        if c not in shots_df.columns:
            return _add_halfcourt_shapes(fig)

    df = shots_df.copy()
    df["LOC_X"] = pd.to_numeric(df["LOC_X"], errors="coerce")
    df["LOC_Y"] = pd.to_numeric(df["LOC_Y"], errors="coerce")
    df["SHOT_MADE_FLAG"] = pd.to_numeric(df["SHOT_MADE_FLAG"], errors="coerce")
    df = df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"])
    df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

    made = df[df["SHOT_MADE_FLAG"] == 1]
    miss = df[df["SHOT_MADE_FLAG"] == 0]

    if not miss.empty:
        fig.add_trace(
            go.Scatter(
                x=miss["LOC_X"],
                y=miss["LOC_Y"],
                mode="markers",
                name="Missed",
                marker=dict(size=6, color="rgb(200,0,0)", line=dict(width=0.5, color="rgba(0,0,0,0.5)")),
                hovertemplate="Missed<extra></extra>",
            )
        )

    if not made.empty:
        fig.add_trace(
            go.Scatter(
                x=made["LOC_X"],
                y=made["LOC_Y"],
                mode="markers",
                name="Made",
                marker=dict(size=6, color="rgb(0,160,0)", line=dict(width=0.5, color="rgba(0,0,0,0.5)")),
                hovertemplate="Made<extra></extra>",
            )
        )

    return _add_halfcourt_shapes(fig)


# =========================================================
# Profile card (league ranking lines) + GREEN ranking text
# =========================================================
def _render_compact_profile_card(
    player_name: str,
    meta: str,
    row: pd.Series,
    id_fallback_name: str,
    rankings_lookup: dict[tuple[str, str], tuple[int, int, float]] | None = None,
    season_override: str | None = None,
):
    season = season_override if season_override else _get_season_from_row(row)

    pid = _get_player_id_from_row(row, fallback_name=id_fallback_name)
    img_url = nba_headshot_url(pid) if pid is not None else None

    with st.container(border=True):
        st.markdown(
            """
            <style>
              .rank-green { color: rgba(46, 204, 113, 0.95); font-size: 0.85rem; margin-top: -6px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if img_url:
            c_img, c_txt = st.columns([1, 3], vertical_alignment="center")
            with c_img:
                st.image(img_url, width=140)
            with c_txt:
                st.markdown(f"### {player_name}")
                if meta:
                    st.caption(meta)
        else:
            st.markdown(f"### {player_name}")
            if meta:
                st.caption(meta)

        keys = ["PTS", "REB", "AST", "STL", "BLK", "TS_PCT", "USG_PCT", "OFF_RATING", "DEF_RATING", "PIE"]

        stats = []
        for k in keys:
            if k in row.index:
                v = _safe_float(row[k])
                if v is not None:
                    stats.append((k, float(v)))

        stats = stats[:10]
        if not stats:
            st.info("No profile stats available.")
            return

        c1, c2 = st.columns(2)
        half = (len(stats) + 1) // 2
        left_stats = stats[:half]
        right_stats = stats[half:]

        def _render_metric(col, metric: str, value: float):
            col.metric(metric, _fmt_stat(metric, value))
            rline = _ranking_line(rankings_lookup, season, metric)
            if rline:
                col.markdown(f"<div class='rank-green'>{rline}</div>", unsafe_allow_html=True)

        with c1:
            for k, v in left_stats:
                _render_metric(c1, k, v)

        with c2:
            for k, v in right_stats:
                _render_metric(c2, k, v)


# =========================================================
# Tornado data
# =========================================================
def _build_metrics_table(deni_row: pd.Series, other_row: pd.Series) -> pd.DataFrame:
    ignore = {
        "PLAYER_NAME", "player_name", "Player", "name",
        "TEAM", "team", "TEAM_NAME", "team_name", "Tm",
        "SEASON", "season", "Season", "YEAR", "year",
        "POS", "pos", "POSITION", "position",
        "AGE", "age",
        "DATA_TYPE", "data_type", "TYPE", "type",
    }

    shared = []
    for c in deni_row.index:
        if c in ignore:
            continue
        if c not in other_row.index:
            continue
        dv = _safe_float(deni_row[c])
        ov = _safe_float(other_row[c])
        if dv is None or ov is None:
            continue
        shared.append(c)

    if not shared:
        return pd.DataFrame()

    priority = [
        "PTS", "REB", "AST", "STL", "BLK",
        "MIN", "GP", "GAMES",
        "TS_PCT", "FG_PCT", "FG3_PCT", "FT_PCT",
        "USG_PCT", "OFF_RATING", "DEF_RATING", "PIE",
    ]
    ordered = [c for c in priority if c in shared] + [c for c in shared if c not in priority]

    rows = []
    for c in ordered:
        rows.append(
            {"metric": c, "deni": float(_safe_float(deni_row[c])), "other": float(_safe_float(other_row[c]))}
        )
    return pd.DataFrame(rows)


# =========================================================
# Tornado VIEW 1: Original Mirror (winner colored, loser gray)
# =========================================================
def _render_tornado_mirror_chart(metrics_df: pd.DataFrame, deni_name: str, other_name: str) -> None:
    if metrics_df is None or metrics_df.empty:
        st.info("Not enough shared metrics to compare.")
        return

    KEEP = ["PTS", "REB", "AST", "TS_PCT", "USG_PCT", "OFF_RATING", "DEF_RATING", "PIE"]

    df = metrics_df.copy()
    df["metric"] = df["metric"].astype(str)
    df = df[df["metric"].isin(KEEP)].copy()
    if df.empty:
        st.info("Not enough shared metrics to compare (after filtering).")
        return

    df["metric"] = pd.Categorical(df["metric"], categories=KEEP, ordered=True)
    df = df.sort_values("metric").reset_index(drop=True)

    lower_is_better = {"DEF_RATING"}

    deni_val = pd.to_numeric(df["deni"], errors="coerce").astype(float)
    other_val = pd.to_numeric(df["other"], errors="coerce").astype(float)

    is_lower = df["metric"].astype(str).isin(lower_is_better)
    deni_win = np.where(is_lower, deni_val <= other_val, deni_val >= other_val)
    other_win = ~deni_win

    deni_green_win = "rgba(76, 175, 80, 0.75)"
    other_purple_win = "rgba(156, 139, 201, 0.75)"
    loser_gray = "rgba(180, 180, 180, 0.25)"

    eps = 1e-12
    denom = np.maximum(np.abs(deni_val), np.abs(other_val))
    denom = np.maximum(denom, eps)
    deni_norm = deni_val / denom
    other_norm = other_val / denom

    df_plot = df.copy()
    df_plot["deni_val"] = deni_val
    df_plot["other_val"] = other_val
    df_plot["deni_norm"] = deni_norm
    df_plot["other_norm"] = other_norm
    df_plot["deni_win"] = deni_win
    df_plot["other_win"] = other_win

    df_plot = df_plot.iloc[::-1].reset_index(drop=True)

    deni_norm_plot = df_plot["deni_norm"].to_numpy(dtype=float)
    other_norm_plot = df_plot["other_norm"].to_numpy(dtype=float)

    deni_colors_plot = [deni_green_win if bool(w) else loser_gray for w in df_plot["deni_win"].tolist()]
    other_colors_plot = [other_purple_win if bool(w) else loser_gray for w in df_plot["other_win"].tolist()]

    def _fmt_inside(v: float, m: str) -> str:
        if not np.isfinite(v):
            return ""
        m_up = str(m).upper()
        if "PCT" in m_up:
            return f"{v * 100:.2f}%" if v <= 1.5 else f"{v:.2f}%"
        return f"{v:.2f}"

    deni_text = [
        _fmt_inside(v, m)
        for v, m in zip(df_plot["deni_val"].to_numpy(float), df_plot["metric"].astype(str).tolist())
    ]
    other_text = [
        _fmt_inside(v, m)
        for v, m in zip(df_plot["other_val"].to_numpy(float), df_plot["metric"].astype(str).tolist())
    ]

    LABELS = {
        "PTS": "Points",
        "REB": "Rebounds",
        "AST": "Assists",
        "TS_PCT": "True Shooting %",
        "USG_PCT": "Usage %",
        "OFF_RATING": "Offensive Rating",
        "DEF_RATING": "Defensive Rating",
        "PIE": "PIE",
    }
    EXPLAIN = {
        "PTS": "Average points per game.",
        "REB": "Average rebounds per game (recovering missed shots).",
        "AST": "Average assists per game (passes leading directly to a made shot).",
        "TS_PCT": "True Shooting %: scoring efficiency including 2PT, 3PT and free throws (higher is better).",
        "USG_PCT": "Usage %: how often the player finishes a team possession (shot, FT, or turnover).",
        "OFF_RATING": "Offensive Rating: team points scored per 100 possessions with the player on court (higher is better).",
        "DEF_RATING": "Defensive Rating: team points allowed per 100 possessions with the player on court (lower is better).",
        "PIE": "Player Impact Estimate: overall impact using many box-score stats (higher is better).",
    }

    metric_codes = df_plot["metric"].astype(str).tolist()
    y_labels = [LABELS.get(m, m) for m in metric_codes]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=-deni_norm_plot,
            orientation="h",
            name=deni_name,
            showlegend=False,
            marker=dict(color=deni_colors_plot, line=dict(width=0)),
            text=deni_text,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="rgba(0,0,0,0.70)", size=11, family="Arial"),
            hovertemplate=f"{deni_name}<br>%{{y}}: %{{text}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=other_norm_plot,
            orientation="h",
            name=other_name,
            showlegend=False,
            marker=dict(color=other_colors_plot, line=dict(width=0)),
            text=other_text,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="rgba(0,0,0,0.70)", size=11, family="Arial"),
            hovertemplate=f"{other_name}<br>%{{y}}: %{{text}}<extra></extra>",
        )
    )

    max_abs = float(np.nanmax(np.abs(np.concatenate([deni_norm_plot, other_norm_plot])))) if len(df_plot) else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    fig.update_traces(cliponaxis=False)

    hover_x = np.full(len(y_labels), -max_abs * 1.18, dtype=float)
    hover_texts = [
        f"<b>{LABELS.get(code, code)}</b><br>{EXPLAIN.get(code, 'No description available.')}" for code in metric_codes
    ]

    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=y_labels,
            mode="markers",
            marker=dict(size=36, color="rgba(0,0,0,0)"),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
            showlegend=False,
        )
    )

    fig.update_layout(
        height=520,
        barmode="overlay",
        bargap=0.32,
        margin=dict(l=190, r=10, t=10, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            showticklabels=False,
            ticks="",
            range=[-max_abs * 1.14, max_abs * 1.10],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            title="",
        ),
        yaxis=dict(title="", automargin=True, showgrid=False),
        plot_bgcolor="rgba(255,255,255,0.0)",
        paper_bgcolor="rgba(255,255,255,0.0)",
    )

    fig.add_annotation(
        x=-max_abs * 1.02,
        y=-0.20,
        xref="x",
        yref="paper",
        text=f"<b>{deni_name}</b>",
        showarrow=False,
        font=dict(size=16, color="rgba(46, 204, 113, 0.95)"),
        align="left",
    )
    fig.add_annotation(
        x=max_abs * 1.02,
        y=-0.20,
        xref="x",
        yref="paper",
        text=f"<b>{other_name}</b>",
        showarrow=False,
        font=dict(size=16, color="rgba(155, 89, 182, 0.95)"),
        align="right",
    )

    st.plotly_chart(fig, width="stretch")


# =========================================================
# Tornado VIEW 2: Grouped (same baseline, always green/purple + black separators)
# =========================================================
def _render_tornado_grouped_chart(metrics_df: pd.DataFrame, deni_name: str, other_name: str) -> None:
    if metrics_df is None or metrics_df.empty:
        st.info("Not enough shared metrics to compare.")
        return

    KEEP = ["PTS", "REB", "AST", "TS_PCT", "USG_PCT", "OFF_RATING", "DEF_RATING", "PIE"]

    df = metrics_df.copy()
    df["metric"] = df["metric"].astype(str)
    df = df[df["metric"].isin(KEEP)].copy()
    if df.empty:
        st.info("Not enough shared metrics to compare (after filtering).")
        return

    df["metric"] = pd.Categorical(df["metric"], categories=KEEP, ordered=True)
    df = df.sort_values("metric").reset_index(drop=True)

    deni_val = pd.to_numeric(df["deni"], errors="coerce").astype(float)
    other_val = pd.to_numeric(df["other"], errors="coerce").astype(float)

    eps = 1e-12
    denom = np.maximum(np.abs(deni_val), np.abs(other_val))
    denom = np.maximum(denom, eps)
    deni_norm = deni_val / denom
    other_norm = other_val / denom

    deni_green = "rgba(76, 175, 80, 0.75)"
    other_purple = "rgba(156, 139, 201, 0.75)"

    LABELS = {
        "PTS": "Points",
        "REB": "Rebounds",
        "AST": "Assists",
        "TS_PCT": "True Shooting %",
        "USG_PCT": "Usage %",
        "OFF_RATING": "Offensive Rating",
        "DEF_RATING": "Defensive Rating",
        "PIE": "PIE",
    }
    EXPLAIN = {
        "PTS": "Average points per game.",
        "REB": "Average rebounds per game (recovering missed shots).",
        "AST": "Average assists per game (passes leading directly to a made shot).",
        "TS_PCT": "True Shooting %: scoring efficiency including 2PT, 3PT and free throws.",
        "USG_PCT": "Usage %: how often the player finishes a team possession (shot, FT, or turnover).",
        "OFF_RATING": "Offensive Rating: team points scored per 100 possessions with the player on court.",
        "DEF_RATING": "Defensive Rating: team points allowed per 100 possessions with the player on court (lower is better).",
        "PIE": "Player Impact Estimate: overall impact using many box-score stats.",
    }

    metric_codes = df["metric"].astype(str).tolist()
    metric_labels = [LABELS.get(m, m) for m in metric_codes]

    def _fmt_inside(v: float, m: str) -> str:
        if not np.isfinite(v):
            return ""
        m_up = str(m).upper()
        if "PCT" in m_up:
            return f"{v * 100:.2f}%" if v <= 1.5 else f"{v:.2f}%"
        return f"{v:.2f}"

    y_metric = []
    y_player = []

    deni_x_aligned = []
    deni_text_aligned = []
    deni_color_aligned = []

    other_x_aligned = []
    other_text_aligned = []
    other_color_aligned = []

    for i, code in enumerate(metric_codes):
        lab = metric_labels[i]

        # Deni row
        y_metric.append(lab)
        y_player.append(deni_name)

        deni_x_aligned.append(float(deni_norm.iloc[i]))
        deni_text_aligned.append(_fmt_inside(float(deni_val.iloc[i]), code))
        deni_color_aligned.append(deni_green)

        other_x_aligned.append(0.0)
        other_text_aligned.append("")
        other_color_aligned.append("rgba(0,0,0,0)")

        # Other row
        y_metric.append(lab)
        y_player.append(other_name)

        deni_x_aligned.append(0.0)
        deni_text_aligned.append("")
        deni_color_aligned.append("rgba(0,0,0,0)")

        other_x_aligned.append(float(other_norm.iloc[i]))
        other_text_aligned.append(_fmt_inside(float(other_val.iloc[i]), code))
        other_color_aligned.append(other_purple)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=[y_metric, y_player],
            x=deni_x_aligned,
            orientation="h",
            name=deni_name,
            showlegend=False,
            marker=dict(color=deni_color_aligned, line=dict(width=0)),
            text=deni_text_aligned,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="rgba(0,0,0,0.70)", size=11, family="Arial"),
            hovertemplate=f"{deni_name}<br>%{{y}}: %{{text}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=[y_metric, y_player],
            x=other_x_aligned,
            orientation="h",
            name=other_name,
            showlegend=False,
            marker=dict(color=other_color_aligned, line=dict(width=0)),
            text=other_text_aligned,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="rgba(0,0,0,0.70)", size=11, family="Arial"),
            hovertemplate=f"{other_name}<br>%{{y}}: %{{text}}<extra></extra>",
        )
    )
    # --- Force legend colors (dummy traces) ---
    deni_green_win = "rgba(76, 175, 80, 0.75)"
    other_purple_win = "rgba(156, 139, 201, 0.75)"

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=deni_green_win),
            name=deni_name,
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=other_purple_win),
            name=other_name,
            showlegend=True,
            hoverinfo="skip",
        )
    )

    max_abs = float(np.nanmax(np.abs(np.concatenate([deni_norm.to_numpy(), other_norm.to_numpy()])))) if len(df) else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    fig.update_traces(cliponaxis=False)

    hover_x = np.full(len(metric_labels), -max_abs * 0.12, dtype=float)
    hover_texts = [
        f"<b>{LABELS.get(code, code)}</b><br>{EXPLAIN.get(code, 'No description available.')}"
        for code in metric_codes
    ]
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=[metric_labels, [deni_name] * len(metric_labels)],
            mode="markers",
            marker=dict(size=34, color="rgba(0,0,0,0)"),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
            showlegend=False,
        )
    )

    fig.update_layout(
        height=640,
        barmode="overlay",
        bargap=0.25,
        margin=dict(l=190, r=10, t=10, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            showticklabels=False,
            ticks="",
            range=[-max_abs * 0.15, max_abs * 1.10],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            title="",
        ),
        yaxis=dict(
            title="",
            automargin=True,
            showgrid=False,
            showdividers=True,
            dividercolor="rgba(0,0,0,0.85)",
            dividerwidth=1,
        ),
        plot_bgcolor="rgba(255,255,255,0.0)",
        paper_bgcolor="rgba(255,255,255,0.0)",
    )

    st.plotly_chart(fig, width="stretch")


# =========================================================
# MAIN RENDER
# =========================================================
def render_deni_labron_casspi(
    deni_shots: pd.DataFrame,
    lebron_shots: pd.DataFrame,
    casspi_shots: pd.DataFrame,
    deni_lebron_compare: pd.DataFrame,
    deni_casspi_compare: pd.DataFrame,
):
    st.header("Deni Avdija Comparisons")

    deni_csv = _resolve_rankings_path("deni_season_rankings.csv")
    lebron_csv = _resolve_rankings_path("lebron_season_rankings.csv")
    casspi_csv = _resolve_rankings_path("casspi_season_rankings.csv")

    deni_rank_lookup = _load_rankings_lookup(deni_csv)
    lebron_rank_lookup = _load_rankings_lookup(lebron_csv)
    casspi_rank_lookup = _load_rankings_lookup(casspi_csv)

    mode = st.radio(
        "Choose comparison",
        ["Deni vs LeBron (Year 6)", "Deni vs Omri Casspi (Peak Season)"],
        horizontal=True,
    )

    deni_row = _pick_deni_row_fixed(deni_lebron_compare, deni_casspi_compare)
    if deni_row is None:
        st.error("Deni row not found in compare tables.")
        return

    if mode == "Deni vs LeBron (Year 6)":
        other_row = _pick_other_row(deni_lebron_compare, "LeBron James")
        other_shots = lebron_shots
        other_default_seasons = ["2008-09"]
        other_id_fallback = "lebron james"
        other_caption = "LeBron James – Season 2008–09 (Year 6)"
        other_rank_lookup = lebron_rank_lookup
    else:
        other_row = _pick_other_row(deni_casspi_compare, "Omri Casspi")
        other_shots = casspi_shots
        other_default_seasons = ["2015-16"]
        other_id_fallback = "omri casspi"
        other_caption = "Omri Casspi – Season 2015–16 (Peak)"
        other_rank_lookup = casspi_rank_lookup

    if other_row is None:
        st.error("Other player row not found in compare table.")
        return

    deni_name = _player_label(deni_row, "Deni Avdija")
    other_name = _player_label(other_row, "LeBron James" if mode.startswith("Deni vs LeBron") else "Omri Casspi")

    # ===== TOP CARDS =====
    c_left, c_right = st.columns(2)
    with c_left:
        _render_compact_profile_card(
            player_name=deni_name,
            meta=_meta_line(deni_row),
            row=deni_row,
            id_fallback_name="deni avdija",
            rankings_lookup=deni_rank_lookup,
            season_override=DENI_CANONICAL_SEASON,
        )

    other_meta = _meta_line(other_row)
    if not other_meta:
        other_meta = other_default_seasons[0]

    with c_right:
        _render_compact_profile_card(
            player_name=other_name,
            meta=other_meta,
            row=other_row,
            id_fallback_name=other_id_fallback,
            rankings_lookup=other_rank_lookup,
            season_override=other_default_seasons[0],
        )

    # ===== TORNADO (FULL WIDTH) =====
    view_mode = st.selectbox(
        "Tornado view",
        ["Mirror", "Grouped"],
        index=0,
        key=f"tornado_view_mode_{mode}",
    )

    metrics_df = _build_metrics_table(deni_row, other_row)

    if view_mode == "Mirror":
        _render_tornado_mirror_chart(metrics_df, deni_name, other_name)
    else:
        _render_tornado_grouped_chart(metrics_df, deni_name, other_name)

    # ===== SHOT CHARTS =====
    s_left, s_right = st.columns(2)
    with s_left:
        st.caption("Deni Avdija – Season 2025–26")
        deni_filtered = _filter_shots_by_seasons(deni_shots, [DENI_CANONICAL_SEASON])
        st.plotly_chart(_make_shot_scatter(deni_filtered), width="stretch")

    with s_right:
        st.caption(other_caption)
        other_filtered = _filter_shots_by_seasons(other_shots, other_default_seasons)
        st.plotly_chart(_make_shot_scatter(other_filtered), width="stretch")
