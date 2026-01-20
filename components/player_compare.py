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

# 5 latest All-Star games (non-starters / reserves)
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

# Extra group: Deni's draft year (2020) top 10 picks
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
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def load_season_data(base_df: pd.DataFrame, adv_df: pd.DataFrame, season: str) -> pd.DataFrame:
    b = base_df[base_df["SEASON"] == season].copy()
    a = adv_df[adv_df["SEASON"] == season].copy()

    adv_keep = [
        "PLAYER_ID", "SEASON",
        "TS_PCT", "EFG_PCT",
        "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_TO",
    ]
    adv_keep = [c for c in adv_keep if c in a.columns]
    a = a[adv_keep].copy()

    merged = b.merge(a, on=["PLAYER_ID", "SEASON"], how="left")
    return merged


def compute_overall_score(league_df: pd.DataFrame) -> pd.Series:
    df = league_df.copy()
    if "GP" in df.columns:
        df = df[df["GP"] >= 10].copy()

    pos_metrics = ["PTS", "AST", "REB", "TS_PCT", "STL", "BLK"]
    neg_metrics = ["TOV"]

    pos_metrics = [m for m in pos_metrics if m in df.columns]
    neg_metrics = [m for m in neg_metrics if m in df.columns]

    parts = []
    for m in pos_metrics:
        parts.append(zscore(df[m]))
    for m in neg_metrics:
        parts.append(-zscore(df[m]))

    if len(parts) == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)

    overall = sum(parts) / float(len(parts))
    overall.index = df.index
    return overall


def build_plot_df(merged_season_df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    group_players = PLAYER_GROUPS.get(group_label, TOP_PLAYERS)

    league_df = merged_season_df.copy()
    if "GP" in league_df.columns:
        league_df = league_df[league_df["GP"] >= 10].copy()

    overall = compute_overall_score(merged_season_df)

    if len(overall) == len(league_df):
        league_df = league_df.copy()
        league_df["OVERALL_SCORE"] = overall.values
    else:
        league_df = merged_season_df.copy()
        league_df["OVERALL_SCORE"] = np.nan

    # Normalize names for robust matching (handles accents, casing, etc.)
    if "PLAYER_NAME" in league_df.columns:
        league_df["NAME_NORM"] = league_df["PLAYER_NAME"].apply(normalize_name)
    else:
        league_df["NAME_NORM"] = ""

    group_norm = set(normalize_name(x) for x in group_players)
    selected = league_df[league_df["NAME_NORM"].isin(group_norm)].copy()

    deni_row = league_df[league_df["PLAYER_ID"] == DENI_ID].copy()
    if not deni_row.empty:
        selected = pd.concat([selected, deni_row], ignore_index=True)

    selected = selected.drop_duplicates(subset=["PLAYER_ID"]).copy()
    selected["IS_DENI"] = selected["PLAYER_ID"] == DENI_ID
    selected["IMG_URL"] = selected["PLAYER_ID"].apply(nba_headshot_url)

    numeric_cols = [
        "PTS", "AST", "REB", "FGA", "FGM", "FG3A", "FG3M",
        "FG_PCT", "FG3_PCT", "TS_PCT",
        "STL", "BLK", "TOV", "MIN", "GP",
        "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_TO",
        "OVERALL_SCORE",
    ]
    for c in numeric_cols:
        if c in selected.columns:
            selected[c] = pd.to_numeric(selected[c], errors="coerce")

    selected = selected.loc[:, ~selected.columns.duplicated()].copy()
    return selected


def make_player_image_scatter(plot_df: pd.DataFrame, x_metric: str) -> go.Figure:
    df = plot_df.copy()

    df = df.dropna(subset=[x_metric, "OVERALL_SCORE"]).copy()
    if df.empty:
        return go.Figure()

    x = df[x_metric].to_numpy(dtype=float)
    y_true = df["OVERALL_SCORE"].to_numpy(dtype=float)

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    y_min = np.nanmin(y_true)
    y_max = np.nanmax(y_true)

    x_range = (x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) != 0 else 1.0
    y_range = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) != 0 else 1.0

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
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGA", "FGM", "FG_PCT",
        "FG3A", "FG3M", "FG3_PCT",
        "TS_PCT",
        "OVERALL_SCORE",
    ]
    hover_cols = [c for c in hover_cols if c in df.columns]
    customdata = df[hover_cols].to_numpy()

    lines = []
    for i, c in enumerate(hover_cols):
        if c == "PLAYER_NAME":
            lines.append(f"<b>%{{customdata[{i}]}}</b>")
        elif c == "OVERALL_SCORE":
            lines.append(f"Overall (true): %{{customdata[{i}]:.3f}}")
        elif c.endswith("_PCT") or c in {"FG_PCT", "FG3_PCT", "TS_PCT"}:
            lines.append(f"{c}: %{{customdata[{i}]:.3f}}")
        else:
            lines.append(f"{c}: %{{customdata[{i}]}}")
    hovertemplate = "<br>".join(lines) + "<extra></extra>"

    fig = go.Figure()

    # Invisible markers (hover hitbox)
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

    # Highlight Deni with a pleasant ring behind the image (not too aggressive)
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
    for xi, yi, url, is_deni in zip(x, y_display, df["IMG_URL"], df["IS_DENI"]):
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

    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title=x_metric, zeroline=False, showgrid=True),
        yaxis=dict(title="Overall Score", zeroline=False, showgrid=True),
        plot_bgcolor="rgba(255,255,255,0.0)",
        paper_bgcolor="rgba(255,255,255,0.0)",
    )

    # Size images by axis ranges
    y_disp_min = np.nanmin(y_display)
    y_disp_max = np.nanmax(y_display)
    y_disp_range = (y_disp_max - y_disp_min) if np.isfinite(y_disp_max - y_disp_min) and (y_disp_max - y_disp_min) != 0 else y_range

    img_sizex = 0.06 * x_range
    img_sizey = 0.10 * y_disp_range

    for i in range(len(fig.layout.images)):
        is_deni = bool(df["IS_DENI"].iloc[i])
        fig.layout.images[i].sizex = img_sizex * (1.35 if is_deni else 1.0)
        fig.layout.images[i].sizey = img_sizey * (1.35 if is_deni else 1.0)

    return fig


def render_player_compare(base_df: pd.DataFrame, adv_df: pd.DataFrame) -> None:
    st.header("Player Comparison – Deni vs Player Groups")
    st.caption(
        "### This page provides a direct comparison of Deni Avdija’s raw performance metrics.\n\n"
        "### It highlights how different statistics contribute to his overall impact, and allows a clearer view of which areas drive his performance the most."
    )

    # Note: CSS wrapper tricks are unreliable in Streamlit because widgets are not actually children
    # of the HTML you inject. Use columns to control widget width (widgets expand to their column).
    deni_seasons = sorted(base_df.loc[base_df["PLAYER_ID"] == DENI_ID, "SEASON"].unique().tolist())
    if len(deni_seasons) == 0:
        st.error("Deni seasons not found in league_all_seasons_base.csv.")
        return

    # Controls row (narrow columns + a spacer so the right side is just page background)
    c_season, c_group, c_metric, c_spacer = st.columns([2, 2, 2, 1])

    with c_season:
        season = st.selectbox(
            "Season",
            deni_seasons,
            index=len(deni_seasons) - 1,
            key="pc_season",
        )

    with c_group:
        group_label = st.selectbox(
            "Compare group",
            list(PLAYER_GROUPS.keys()),
            index=0,
            key="pc_group",
        )

    merged = load_season_data(base_df, adv_df, season)
    plot_df = build_plot_df(merged, group_label=group_label)

    metric_options = [
        "PTS", "REB", "AST",
        "FGA", "FGM", "FG_PCT",
        "FG3A", "FG3M", "FG3_PCT",
        "TS_PCT",
        "STL", "BLK", "TOV",
        "MIN", "GP",
        "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_TO",
    ]
    metric_options = [m for m in metric_options if m in merged.columns]
    if len(metric_options) == 0:
        st.error("No supported metrics found in your season data.")
        return

    with c_metric:
        x_metric = st.selectbox(
            "X axis metric",
            metric_options,
            index=0,
            key="pc_x_metric",
        )

    with c_spacer:
        st.empty()

    st.caption(f"Season: {season} | Group: {group_label} | Players shown: {len(plot_df)}")

    fig = make_player_image_scatter(plot_df, x_metric)
    if len(fig.data) == 0:
        st.warning("No valid points for this season/metric (missing data).")
        return

    st.plotly_chart(fig, width="stretch")

    with st.expander("Show players data table"):
        st.caption("Rank values indicate Deni’s position relative to the other players shown in this table.")

        cols_show = [
            "PLAYER_ID", "PLAYER_NAME", x_metric, "OVERALL_SCORE",
            "PTS", "REB", "AST", "TS_PCT", "FG3_PCT",
            "TOV", "GP", "MIN",
        ]
        cols_show = [c for c in cols_show if c in plot_df.columns]
        cols_show = list(dict.fromkeys(cols_show))

        tbl = plot_df.loc[:, ~plot_df.columns.duplicated()].copy()
        table_df = tbl[cols_show].sort_values("OVERALL_SCORE", ascending=False).copy()

        deni_mask = (
            (pd.to_numeric(table_df["PLAYER_ID"], errors="coerce") == DENI_ID)
            if "PLAYER_ID" in table_df.columns
            else pd.Series(False, index=table_df.index)
        )

        if not deni_mask.any():
            st.dataframe(table_df.drop(columns=["PLAYER_ID"], errors="ignore"), width="stretch")
            return

        deni_idx = table_df.index[deni_mask][0]

        # Metrics to augment (exclude identifiers/text)
        metric_cols = [c for c in table_df.columns if c not in {"PLAYER_ID", "PLAYER_NAME"}]

        # Ranking rules:
        # - For TOV: lower is better
        # - For others: higher is better
        def _ascending_for(col: str) -> bool:
            return col == "TOV"

        def _fmt_val(col: str, v):
            if pd.isna(v):
                return ""
            try:
                x = float(v)
            except Exception:
                return str(v)

            if col == "OVERALL_SCORE":
                return f"{x:.3f}"
            if col.endswith("_PCT"):
                return f"{x:.3f}"
            if col in {"PTS", "REB", "AST", "STL", "BLK", "TOV"}:
                return f"{x:.1f}"
            if col in {"GP"}:
                return f"{int(round(x))}"
            if col in {"MIN"}:
                return f"{x:.1f}"
            return f"{x:.3f}"

        disp = table_df.copy()

        # Format all metric columns as strings for clean HTML rendering
        for c in metric_cols:
            disp[c] = [_fmt_val(c, v) for v in table_df[c].to_list()]

        # Inject rank into Deni cells only
        for c in metric_cols:
            s = pd.to_numeric(table_df[c], errors="coerce")
            ranks = s.rank(method="min", ascending=_ascending_for(c), na_option="bottom")

            deni_rank_val = ranks.loc[deni_idx]
            if pd.isna(deni_rank_val):
                continue

            deni_rank = str(int(deni_rank_val))
            base_val = disp.at[deni_idx, c]
            if base_val == "":
                continue

            disp.at[deni_idx, c] = (
                f'{base_val} '
                f'<span style="color: rgba(0, 100, 50, 0.95); font-size: 11px; font-weight: 700; margin-left: 6px;">{deni_rank}</span>'
            )

        # Bold player name for Deni
        disp["PLAYER_NAME"] = disp["PLAYER_NAME"].astype(str)
        disp.at[deni_idx, "PLAYER_NAME"] = f"<b>{disp.at[deni_idx, 'PLAYER_NAME']}</b>"

        # Drop PLAYER_ID from display
        disp_show = disp.drop(columns=["PLAYER_ID"], errors="ignore")

        # HTML table with highlighted Deni row
        def _row_style(idx):
            if idx == deni_idx:
                return ' style="background-color: rgba(255, 215, 0, 0.18); font-weight: 700;"'
            return ""

        headers = "".join([
            f"<th style='text-align:left; padding:8px; border-bottom:1px solid rgba(0,0,0,0.08); white-space:nowrap;'>{c}</th>"
            for c in disp_show.columns
        ])

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
