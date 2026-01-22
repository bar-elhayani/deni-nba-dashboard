import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import unicodedata


# -----------------------------
# Groups / Cohorts (menu)
# -----------------------------
TOP_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Nikola Jokic",
    "Kevin Durant",
    "Giannis Antetokounmpo",
    "Luka Doncic",
    "Joel Embiid",
    "James Harden",
    "Donovan Mitchell",
    "Shai Gilgeous-Alexander",
]

ALLSTAR_2025_NON_STARTERS = [
    "Damian Lillard",
    "Darius Garland",
    "Jaylen Brown",
    "Cade Cunningham",
    "Tyler Herro",
    "Anthony Edwards",
    "James Harden",
    "Jaren Jackson Jr.",
    "Victor Wembanyama",
    "Alperen Sengun",
]

ALLSTAR_2024_NON_STARTERS = [
    "Bam Adebayo",
    "Paolo Banchero",
    "Jaylen Brown",
    "Jalen Brunson",
    "Tyrese Maxey",
    "Devin Booker",
    "Stephen Curry",
    "Anthony Edwards",
    "Paul George",
    "Kawhi Leonard",
]

ALLSTAR_2023_NON_STARTERS = [
    "Bam Adebayo",
    "Jaylen Brown",
    "DeMar DeRozan",
    "Tyrese Haliburton",
    "Jrue Holiday",
    "Shai Gilgeous-Alexander",
    "Damian Lillard",
    "Ja Morant",
    "Lauri Markkanen",
    "Domantas Sabonis",
]

ALLSTAR_2022_NON_STARTERS = [
    "Jimmy Butler",
    "Darius Garland",
    "James Harden",
    "Zach LaVine",
    "Jayson Tatum",
    "Devin Booker",
    "Luka Doncic",
    "Donovan Mitchell",
    "Chris Paul",
    "Karl-Anthony Towns",
]

ALLSTAR_2021_NON_STARTERS = [
    "Jaylen Brown",
    "James Harden",
    "Zach LaVine",
    "Julius Randle",
    "Ben Simmons",
    "Anthony Davis",
    "Paul George",
    "Rudy Gobert",
    "Damian Lillard",
    "Donovan Mitchell",
]

DRAFT_2020_TOP10 = [
    "Anthony Edwards",
    "James Wiseman",
    "LaMelo Ball",
    "Patrick Williams",
    "Isaac Okoro",
    "Onyeka Okongwu",
    "Killian Hayes",
    "Obi Toppin",
    "Deni Avdija",
    "Jalen Smith",
]

PLAYER_GROUPS = {
    "TOP 10 PLAYERS": TOP_PLAYERS,
    "All-Star 2025 (Non-starters)": ALLSTAR_2025_NON_STARTERS,
    "All-Star 2024 (Non-starters)": ALLSTAR_2024_NON_STARTERS,
    "All-Star 2023 (Non-starters)": ALLSTAR_2023_NON_STARTERS,
    "All-Star 2022 (Non-starters)": ALLSTAR_2022_NON_STARTERS,
    "All-Star 2021 (Non-starters)": ALLSTAR_2021_NON_STARTERS,
    "Draft 2020 (Top 10 picks)": DRAFT_2020_TOP10,
}

DENI_ID = 1630166


# -----------------------------
# Offense metric options (Y)
# -----------------------------
OFFENSE_METRICS = {
    "Usage (USG/approx)": "USAGE_SCORE",
    "FER (Efficiency Rating)": "FER_SCORE",
    "True Shooting %": "TS_SCORE",
    "+/- (Plus-Minus)": "PM_SCORE",
}


def nba_headshot_url(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - mu) / sigma


def pick_usage_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "USG_PCT", "USAGE_PCT", "USAGE", "USG", "USG%",
        "USAGE_RATE", "USAGE_RATE_PCT",
    ]
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def load_season_data(base_df: pd.DataFrame, adv_df: pd.DataFrame, season: str) -> pd.DataFrame:
    b = base_df[base_df["SEASON"] == season].copy()
    a = adv_df[adv_df["SEASON"] == season].copy()

    adv_keep = [
        "PLAYER_ID", "SEASON",
        "USG_PCT", "USAGE_PCT", "USAGE", "USG", "USG%", "USAGE_RATE", "USAGE_RATE_PCT",
        "DEF_RATING", "NET_RATING", "OFF_RATING",
        "DREB_PCT", "STL_PCT", "BLK_PCT",
    ]
    adv_keep = [c for c in adv_keep if c in a.columns]
    a = a[adv_keep].copy()

    merged = b.merge(a, on=["PLAYER_ID", "SEASON"], how="left")
    return merged


def compute_usage_score(league_df: pd.DataFrame) -> pd.Series:
    """
    Higher = more usage.
    Prefer an official usage column if exists; else approximate usage from box-score.
    """
    df = league_df.copy()

    usage_col = pick_usage_column(df)
    if usage_col is not None:
        return zscore(df[usage_col])

    need = ["FGA", "TOV"]
    if not all(c in df.columns for c in need):
        return pd.Series(np.zeros(len(df)), index=df.index)

    fga = pd.to_numeric(df["FGA"], errors="coerce").fillna(0.0)
    tov = pd.to_numeric(df["TOV"], errors="coerce").fillna(0.0)
    fta = pd.to_numeric(df["FTA"], errors="coerce").fillna(0.0) if "FTA" in df.columns else 0.0

    approx = fga + 0.44 * fta + tov
    approx.index = df.index
    return zscore(approx)


def compute_true_shooting_pct(league_df: pd.DataFrame) -> pd.Series:
    """
    TS% = PTS / (2 * (FGA + 0.44*FTA))
    If there is an existing TS column, use it. Otherwise compute if possible.
    Returns raw TS% (0-1 or 0-100 depending on source); caller z-scores it.
    """
    df = league_df.copy()

    # Try common column names first
    ts_candidates = ["TS_PCT", "TS%", "TRUE_SHOOTING", "TRUE_SHOOTING_PCT", "TS"]
    for c in ts_candidates:
        if c in df.columns and df[c].notna().any():
            s = pd.to_numeric(df[c], errors="coerce")
            s.index = df.index
            # If it looks like percent (0-100), convert to 0-1 for consistency
            mx = float(np.nanmax(s.to_numpy(dtype=float))) if len(s) else np.nan
            if np.isfinite(mx) and mx > 1.5:
                s = s / 100.0
            return s

    need = ["PTS", "FGA"]
    if not all(c in df.columns for c in need):
        return pd.Series(np.nan, index=df.index)

    pts = pd.to_numeric(df["PTS"], errors="coerce")
    fga = pd.to_numeric(df["FGA"], errors="coerce")
    fta = pd.to_numeric(df["FTA"], errors="coerce") if "FTA" in df.columns else pd.Series(np.nan, index=df.index)

    denom = 2.0 * (fga + 0.44 * fta.fillna(0.0))
    denom = denom.replace(0.0, np.nan)
    ts = pts / denom
    ts.index = df.index
    return ts


def compute_fer_efficiency(league_df: pd.DataFrame) -> pd.Series:
    """
    FER (simple efficiency rating, per game):
    EFF = PTS + REB + AST + STL + BLK
          - (FGA - FGM) - (FTA - FTM) - TOV
    If GP exists, convert totals to per-game. If stats are already per-game, GP division won't harm much
    because totals-per-game vs per-game differs by a constant scale; z-score normalizes anyway.
    """
    df = league_df.copy()

    required = ["PTS", "REB", "AST", "STL", "BLK", "FGA", "FGM", "TOV"]
    if not all(c in df.columns for c in required):
        return pd.Series(np.nan, index=df.index)

    pts = pd.to_numeric(df["PTS"], errors="coerce")
    reb = pd.to_numeric(df["REB"], errors="coerce")
    ast = pd.to_numeric(df["AST"], errors="coerce")
    stl = pd.to_numeric(df["STL"], errors="coerce")
    blk = pd.to_numeric(df["BLK"], errors="coerce")
    fga = pd.to_numeric(df["FGA"], errors="coerce")
    fgm = pd.to_numeric(df["FGM"], errors="coerce")
    tov = pd.to_numeric(df["TOV"], errors="coerce")

    fta = pd.to_numeric(df["FTA"], errors="coerce") if "FTA" in df.columns else pd.Series(0.0, index=df.index)
    ftm = pd.to_numeric(df["FTM"], errors="coerce") if "FTM" in df.columns else pd.Series(0.0, index=df.index)

    eff = (pts + reb + ast + stl + blk) - (fga - fgm) - (fta - ftm) - tov
    eff.index = df.index

    if "GP" in df.columns:
        gp = pd.to_numeric(df["GP"], errors="coerce").replace(0, np.nan)
        # If totals, convert to per-game; if already per-game, division scales but z-score handles it.
        eff = eff / gp

    return eff


def compute_plus_minus(league_df: pd.DataFrame) -> pd.Series:
    """
    Try to find +/- column. Common names: PLUS_MINUS, +/-, PLUSMINUS, PM
    """
    df = league_df.copy()
    pm_candidates = ["PLUS_MINUS", "+/-", "PLUSMINUS", "PM", "PLUS_MINUS_PG"]
    for c in pm_candidates:
        if c in df.columns and df[c].notna().any():
            s = pd.to_numeric(df[c], errors="coerce")
            s.index = df.index
            return s
    return pd.Series(np.nan, index=df.index)


def compute_offense_score(league_df: pd.DataFrame, offense_metric_key: str) -> pd.Series:
    """
    Higher = better offensive output for the chosen metric.
    Returns a z-scored series.
    """
    df = league_df.copy()

    if offense_metric_key == "USAGE_SCORE":
        return compute_usage_score(df)

    if offense_metric_key == "TS_SCORE":
        ts = compute_true_shooting_pct(df)
        return zscore(ts)

    if offense_metric_key == "FER_SCORE":
        fer = compute_fer_efficiency(df)
        return zscore(fer)

    if offense_metric_key == "PM_SCORE":
        pm = compute_plus_minus(df)
        return zscore(pm)

    return pd.Series(np.zeros(len(df)), index=df.index)


def compute_defense_score(league_df: pd.DataFrame, defense_metric: str) -> pd.Series:
    """
    Higher = better defense.
    DEF_RATING is lower-better, so we flip sign.
    """
    df = league_df.copy()

    if defense_metric not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)

    s = pd.to_numeric(df[defense_metric], errors="coerce")
    s.index = df.index

    if defense_metric == "DEF_RATING":
        return -zscore(s)

    return zscore(s)


def build_usage_def_plot_df(
    merged_season_df: pd.DataFrame,
    group_label: str,
    defense_metric: str,
    offense_metric_key: str,
) -> tuple[pd.DataFrame, float]:
    group_players = PLAYER_GROUPS.get(group_label, TOP_PLAYERS)

    league_df = merged_season_df.copy()
    if "GP" in league_df.columns:
        league_df = league_df[pd.to_numeric(league_df["GP"], errors="coerce") >= 10].copy()

    if "PLAYER_NAME" in league_df.columns:
        league_df["NAME_NORM"] = league_df["PLAYER_NAME"].apply(normalize_name)
    else:
        league_df["NAME_NORM"] = ""

    group_norm = set(normalize_name(x) for x in group_players)
    selected = league_df[league_df["NAME_NORM"].isin(group_norm)].copy()

    deni_row = league_df[league_df["PLAYER_ID"] == DENI_ID].copy()
    if not deni_row.empty:
        selected = pd.concat([selected, deni_row], axis=0)

    selected = selected.drop_duplicates(subset=["PLAYER_ID"]).copy()

    # Full-league scoring (z-scores over league, then map by PLAYER_ID)
    off_full = compute_offense_score(league_df, offense_metric_key=offense_metric_key)
    def_full = compute_defense_score(league_df, defense_metric=defense_metric)

    pid_series = pd.to_numeric(league_df["PLAYER_ID"], errors="coerce")

    off_by_pid = pd.Series(off_full.to_numpy(), index=pid_series.to_numpy())
    def_by_pid = pd.Series(def_full.to_numpy(), index=pid_series.to_numpy())

    selected["OFF_SCORE"] = pd.to_numeric(selected["PLAYER_ID"], errors="coerce").map(off_by_pid)
    selected["DEF_SCORE"] = pd.to_numeric(selected["PLAYER_ID"], errors="coerce").map(def_by_pid)

    # Keep old usage score too (for table/hover), even when Y is not usage
    usage_full = compute_usage_score(league_df)
    usage_by_pid = pd.Series(usage_full.to_numpy(), index=pid_series.to_numpy())
    selected["USAGE_SCORE"] = pd.to_numeric(selected["PLAYER_ID"], errors="coerce").map(usage_by_pid)

    selected["IS_DENI"] = selected["PLAYER_ID"] == DENI_ID
    selected["IMG_URL"] = selected["PLAYER_ID"].apply(nba_headshot_url)

    numeric_cols = [
        "OFF_SCORE", "DEF_SCORE", "USAGE_SCORE",
        "USG_PCT", "DEF_RATING", "NET_RATING", "OFF_RATING",
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "GP",
        "FGA", "FGM", "FTA", "FTM",
        "PLUS_MINUS", "+/-", "PM",
    ]
    for c in numeric_cols:
        if c in selected.columns:
            selected[c] = pd.to_numeric(selected[c], errors="coerce")

    selected = selected.loc[:, ~selected.columns.duplicated()].copy()

    league_mean_x = float(np.nanmean(def_full.to_numpy(dtype=float))) if len(def_full) else 0.0
    return selected, league_mean_x


def make_player_image_scatter_xy(
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    x_mean_line: float | None = None,
) -> go.Figure:
    df = plot_df.copy()
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return go.Figure()

    x = df[x_col].to_numpy(dtype=float)
    y_true = df[y_col].to_numpy(dtype=float)

    # Force axes to be centered (symmetric ranges around 0)
    x_abs = float(np.nanmax(np.abs(x))) if len(x) else 1.0
    y_abs = float(np.nanmax(np.abs(y_true))) if len(y_true) else 1.0
    x_pad = 0.15 * x_abs
    y_pad = 0.15 * y_abs
    x_lim = max(1e-9, x_abs + x_pad)
    y_lim = max(1e-9, y_abs + y_pad)
    x_range_fixed = [-x_lim, x_lim]
    y_range_fixed = [-y_lim, y_lim]

    x_range = (x_range_fixed[1] - x_range_fixed[0]) if (x_range_fixed[1] - x_range_fixed[0]) != 0 else 1.0
    y_range = (y_range_fixed[1] - y_range_fixed[0]) if (y_range_fixed[1] - y_range_fixed[0]) != 0 else 1.0

    # Collision-avoidance
    x_thr = 0.035 * x_range
    y_thr = 0.045 * y_range
    y_step = 0.030 * y_range

    order = np.argsort(y_true)
    y_display = y_true.copy()

    placed = []
    for idx in order:
        xi = float(x[idx])
        yi = float(y_true[idx])

        found = False
        for k in range(0, 12):
            if k == 0:
                cand = yi
            else:
                sign = 1 if (k % 2 == 1) else -1
                step_mul = (k + 1) // 2
                cand = yi + sign * step_mul * y_step

            ok = True
            for (px, py) in placed:
                if abs(xi - px) < x_thr and abs(cand - py) < y_thr:
                    ok = False
                    break

            if ok:
                y_display[idx] = cand
                placed.append((xi, cand))
                found = True
                break

        if not found:
            pid = int(df["PLAYER_ID"].iloc[idx]) if "PLAYER_ID" in df.columns else idx
            rng = np.random.default_rng(pid)
            y_display[idx] = yi + float(rng.uniform(-0.02 * y_range, 0.02 * y_range))
            placed.append((xi, y_display[idx]))

    hover_cols = [
        "PLAYER_NAME", "GP", "MIN",
        "USG_PCT", "DEF_RATING", "NET_RATING", "OFF_RATING",
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "USAGE_SCORE", "OFF_SCORE", "DEF_SCORE",
    ]
    hover_cols = [c for c in hover_cols if c in df.columns]
    customdata = df[hover_cols].to_numpy()

    lines = []
    for i, c in enumerate(hover_cols):
        if c == "PLAYER_NAME":
            lines.append(f"<b>%{{customdata[{i}]}}</b>")
        elif c in {"USAGE_SCORE", "OFF_SCORE", "DEF_SCORE"}:
            lines.append(f"{c}: %{{customdata[{i}]:.3f}}")
        elif c.endswith("_PCT") or c == "USG%":
            lines.append(f"{c}: %{{customdata[{i}]:.3f}}")
        else:
            lines.append(f"{c}: %{{customdata[{i}]}}")
    hovertemplate = "<br>".join(lines) + "<extra></extra>"

    fig = go.Figure()

    # Invisible hover hitbox
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_display,
            mode="markers",
            marker=dict(size=42, color="rgba(0,0,0,0)"),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
        )
    )

    # Highlight Deni
    deni_mask = df["IS_DENI"].to_numpy(dtype=bool)
    if np.any(deni_mask):
        deni_x = x[deni_mask][0]
        deni_y = y_display[deni_mask][0]
        fig.add_trace(
            go.Scatter(
                x=[float(deni_x)],
                y=[float(deni_y)],
                mode="markers",
                marker=dict(
                    size=78,
                    color="rgba(255, 215, 0, 0.10)",
                    line=dict(width=3, color="rgba(255, 215, 0, 0.85)"),
                    symbol="circle",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Add images
    images = []
    for xi, yi, url in zip(x, y_display, df["IMG_URL"]):
        images.append(
            dict(
                source=url,
                xref="x",
                yref="y",
                x=float(xi),
                y=float(yi),
                xanchor="center",
                yanchor="middle",
                sizex=0.0,
                sizey=0.0,
                sizing="contain",
                opacity=1.0,
                layer="above",
            )
        )
    fig.update_layout(images=images)

    # Axes at 0 and mean line
    shapes = []

    shapes.append(
        dict(
            type="line", xref="x", yref="y",
            x0=0, x1=0,
            y0=y_range_fixed[0], y1=y_range_fixed[1],
            line=dict(color="rgba(0,0,0,0.85)", width=2),
        )
    )
    shapes.append(
        dict(
            type="line", xref="x", yref="y",
            x0=x_range_fixed[0], x1=x_range_fixed[1],
            y0=0, y1=0,
            line=dict(color="rgba(0,0,0,0.85)", width=2),
        )
    )

    if x_mean_line is not None and np.isfinite(x_mean_line):
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=float(x_mean_line), x1=float(x_mean_line),
                y0=y_range_fixed[0], y1=y_range_fixed[1],
                line=dict(color="rgba(0,0,0,0.85)", width=2),
            )
        )

    fig.update_layout(shapes=shapes)

    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            range=x_range_fixed,
            title=dict(text=x_title, font=dict(color="black", size=16)),
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="black", size=14),
        ),
        yaxis=dict(
            range=y_range_fixed,
            title=dict(text=y_title, font=dict(color="black", size=16)),
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="black", size=14),
        ),
        plot_bgcolor="rgba(255,255,255,0.0)",
        paper_bgcolor="rgba(255,255,255,0.0)",
    )

    # Size images by axis ranges (bigger + more stable)
    y_disp_min = np.nanmin(y_display)
    y_disp_max = np.nanmax(y_display)
    y_disp_range = (y_disp_max - y_disp_min) if np.isfinite(y_disp_max - y_disp_min) and (
        y_disp_max - y_disp_min) != 0 else y_range

    BASE_SIZE_X = 0.28
    BASE_SIZE_Y = 0.36

    img_sizex = max(BASE_SIZE_X, 0.10 * x_range)
    img_sizey = max(BASE_SIZE_Y, 0.14 * y_disp_range)

    for i in range(len(fig.layout.images)):
        is_deni = bool(df["IS_DENI"].iloc[i])
        fig.layout.images[i].sizex = img_sizex * (1.25 if is_deni else 1.0)
        fig.layout.images[i].sizey = img_sizey * (1.25 if is_deni else 1.0)

    return fig


def render_usage_vs_defense(base_df: pd.DataFrame, adv_df: pd.DataFrame) -> None:
    st.header("Usage vs Defense – Deni vs Player Groups")
    st.caption(
        "### This page examines Deni Avdija’s offensive and defensive impact relative to other players.\n\n"
        "### It shows how he compares to league peers across key metrics, and where he stands in terms of overall balance between offense and defense."
    )

    deni_seasons = sorted(base_df.loc[base_df["PLAYER_ID"] == DENI_ID, "SEASON"].unique().tolist())
    if len(deni_seasons) == 0:
        st.error("Deni seasons not found in league_all_seasons_base.csv.")
        return

    c1, c2, c3, c4 = st.columns([2, 3, 3, 3])
    with c1:
        season = st.selectbox("Season", deni_seasons, index=len(deni_seasons) - 1)
    with c2:
        group_label = st.selectbox("Compare group", list(PLAYER_GROUPS.keys()), index=0)
    with c3:
        merged_tmp = load_season_data(base_df, adv_df, season)
        defense_options = []
        for m in ["DEF_RATING", "STL", "BLK", "DREB_PCT", "STL_PCT", "BLK_PCT", "NET_RATING"]:
            if m in merged_tmp.columns:
                defense_options.append(m)
        if len(defense_options) == 0:
            defense_options = ["DEF_RATING"]
        defense_metric = st.selectbox("Defense metric (X)", defense_options, index=0)
    with c4:
        offense_labels = list(OFFENSE_METRICS.keys())
        offense_label = st.selectbox("Offense metric (Y)", offense_labels, index=0)
        offense_metric_key = OFFENSE_METRICS[offense_label]

    merged = load_season_data(base_df, adv_df, season)
    plot_df, league_mean_x = build_usage_def_plot_df(
        merged,
        group_label=group_label,
        defense_metric=defense_metric,
        offense_metric_key=offense_metric_key,
    )

    st.caption(f"Season: {season} | Group: {group_label} | Players shown: {len(plot_df)}")

    x_title = "Defense (higher = better)"
    if defense_metric == "DEF_RATING":
        x_title = "Defense Rating (converted: higher = better)"

    if offense_metric_key == "USAGE_SCORE":
        y_title = "Usage Score (higher = more usage)"
    elif offense_metric_key == "TS_SCORE":
        y_title = "True Shooting (z-score, higher = better)"
    elif offense_metric_key == "FER_SCORE":
        y_title = "FER / Efficiency (z-score, higher = better)"
    elif offense_metric_key == "PM_SCORE":
        y_title = "+/- (z-score, higher = better)"
    else:
        y_title = "Offense (z-score)"

    fig = make_player_image_scatter_xy(
        plot_df,
        x_col="DEF_SCORE",
        y_col="OFF_SCORE",
        x_title=x_title,
        y_title=y_title,
        x_mean_line=league_mean_x,
    )

    if len(fig.data) == 0:
        st.warning("No valid points for this season/metric (missing data).")
        return

    st.plotly_chart(fig, width="stretch")

    with st.expander("Show players data table"):
        st.caption("Rank values indicate Deni’s position relative to the other players shown in this table.")

        cols_show = [
            "PLAYER_ID", "PLAYER_NAME", "DEF_SCORE", "OFF_SCORE", "USAGE_SCORE",
            "USG_PCT", "DEF_RATING", "NET_RATING", "OFF_RATING",
            "PTS", "REB", "AST", "STL", "BLK", "TOV",
            "GP", "MIN"
        ]
        cols_show = [c for c in cols_show if c in plot_df.columns]
        cols_show = list(dict.fromkeys(cols_show))

        table_df = plot_df[cols_show].copy()

        # Sort like before
        sort_primary = "OFF_SCORE" if "OFF_SCORE" in table_df.columns else (
            "USAGE_SCORE" if "USAGE_SCORE" in table_df.columns else None)
        if sort_primary is not None and "DEF_SCORE" in table_df.columns:
            table_df = table_df.sort_values([sort_primary, "DEF_SCORE"], ascending=False)
        elif sort_primary is not None:
            table_df = table_df.sort_values([sort_primary], ascending=False)

        # Identify Deni row
        deni_mask = (pd.to_numeric(table_df["PLAYER_ID"],
                                   errors="coerce") == DENI_ID) if "PLAYER_ID" in table_df.columns else pd.Series(False,
                                                                                                                  index=table_df.index)
        if not deni_mask.any():
            st.dataframe(table_df.drop(columns=["PLAYER_ID"], errors="ignore"), width="stretch")
        else:
            deni_idx = table_df.index[deni_mask][0]

            # Metrics to augment (exclude identifiers/text)
            metric_cols = [c for c in table_df.columns if c not in {"PLAYER_ID", "PLAYER_NAME"}]

            # Ranking rules: DEF_RATING lower is better; TOV lower is better; otherwise higher is better
            def _ascending_for(col: str) -> bool:
                return col in {"DEF_RATING", "TOV"}

            # Format helpers
            def _fmt_val(col: str, v):
                if pd.isna(v):
                    return ""
                try:
                    x = float(v)
                except Exception:
                    return str(v)

                if col.endswith("_SCORE"):
                    return f"{x:.3f}"
                if col in {"USG_PCT", "TS_PCT"}:
                    return f"{x:.3f}"
                if col in {"PTS", "REB", "AST", "STL", "BLK", "TOV", "GP"}:
                    return f"{x:.1f}" if col != "GP" else f"{int(round(x))}"
                return f"{x:.2f}"

            # Create display copy as strings
            disp = table_df.copy()

            # Compute ranks and inject ONLY into Deni cells
            for c in metric_cols:
                s = pd.to_numeric(table_df[c], errors="coerce")
                ranks = s.rank(method="min", ascending=_ascending_for(c), na_option="bottom")
                deni_rank_val = ranks.loc[deni_idx]
                if pd.isna(deni_rank_val):
                    deni_rank = ""
                else:
                    deni_rank = str(int(deni_rank_val))

                # Convert entire column to formatted strings (so HTML table looks clean)
                disp[c] = [_fmt_val(c, v) for v in table_df[c].to_list()]

                # Inject rank next to value ONLY for Deni row
                base_val = disp.at[deni_idx, c]
                if base_val != "" and deni_rank != "":
                    disp.at[deni_idx, c] = (
                        f'{base_val} <span style="color: rgba(0,100,50,0.95); font-size: 12px; font-weight: 700; margin-left: 8px;">{deni_rank}</span>'
                    )

            # Bold + highlight Deni row (PLAYER_NAME cell and full row background)
            disp["PLAYER_NAME"] = disp["PLAYER_NAME"].astype(str)
            disp.at[deni_idx, "PLAYER_NAME"] = f"<b>{disp.at[deni_idx, 'PLAYER_NAME']}</b>"

            # Drop PLAYER_ID from display
            disp_show = disp.drop(columns=["PLAYER_ID"], errors="ignore")

            # Build HTML table with row highlight
            def _row_style(idx):
                if idx == deni_idx:
                    return ' style="background-color: rgba(255, 215, 0, 0.18); font-weight: 700;"'
                return ""

            headers = "".join(
                [f"<th style='text-align:left; padding:8px; border-bottom:1px solid rgba(0,0,0,0.08);'>{c}</th>" for c
                 in disp_show.columns])

            rows_html = []
            for idx, row in disp_show.iterrows():
                tds = []
                for c in disp_show.columns:
                    align = "left" if c == "PLAYER_NAME" else "right"
                    tds.append(
                        f"<td style='text-align:{align}; padding:8px; border-bottom:1px solid rgba(0,0,0,0.06); white-space:nowrap;'>{row[c]}</td>"
                    )
                rows_html.append(f"<tr{_row_style(idx)}>" + "".join(tds) + "</tr>")

            html = f"""
            <div style="overflow-x:auto; width:100%;">
              <table style="border-collapse:collapse; width:100%; font-size: 14px;">
                <thead><tr>{headers}</tr></thead>
                <tbody>
                  {''.join(rows_html)}
                </tbody>
              </table>
            </div>
            """

            st.markdown(html, unsafe_allow_html=True)
