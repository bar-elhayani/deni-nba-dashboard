import os
import re
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Constants

CHART_KEY = "career_progression_chart"

# Expected locations
TEAM_LOGO_PATHS = {
    "POR": os.path.join("data", "images", "trailblazers_logo.png"),
    "WAS": os.path.join("data", "images", "wizards_logo.png"),
}

TEAM_BG_TINT = {
    "POR": ("rgba(239,68,68,0.07)", "rgba(0,0,0,0.0)"),   # soft red
    "WAS": ("rgba(59,130,246,0.07)", "rgba(0,0,0,0.0)"),  # soft blue
}


# ----------------------------
# Helpers
# ----------------------------
def _norm_season_to_sort_key(season_val: str) -> int:
    if season_val is None:
        return 10**9
    s = str(season_val).strip()
    m = re.search(r"(19|20)\d{2}", s)
    if not m:
        return 10**9
    return int(m.group(0))


def _find_deni_key(df: pd.DataFrame):
    if "player_id" in df.columns and "player_name" in df.columns:
        deni_rows = df[
            df["player_name"].astype(str).str.lower().str.contains("deni")
            & df["player_name"].astype(str).str.lower().str.contains("avdija")
        ]
        if len(deni_rows) > 0:
            return "player_id", deni_rows.iloc[0]["player_id"]

    if "player_name" in df.columns:
        mask = (
            df["player_name"].astype(str).str.lower().str.contains("deni")
            & df["player_name"].astype(str).str.lower().str.contains("avdija")
        )
        if mask.any():
            return "player_name", df.loc[mask, "player_name"].iloc[0]

    return None, None


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


# ----------------------------
# NEW: standardize column names across sources
# ----------------------------
def _standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # season / player base names
    rename_map = {
        "SEASON": "season",
        "Season": "season",
        "YEAR": "season",
        "year": "season",
        "PLAYER_NAME": "player_name",
        "Player": "player_name",
        "NAME": "player_name",
        "name": "player_name",
        "PLAYER_ID": "player_id",
        "ID": "player_id",
        "Id": "player_id",
        "id": "player_id",
        "PERSON_ID": "player_id",
    }

    # GP: this is a frequent mismatch causing totals to be wrong
    gp_alts = ["GP", "G", "Games", "GAMES", "games_played", "GAMES_PLAYED"]
    for c in gp_alts:
        if c in out.columns and c != "GP":
            rename_map[c] = "GP"
            break

    # MP / Minutes
    mp_alts = ["MP", "MIN", "Minutes", "MINUTES", "MPG", "MINPG", "MIN_PER_GAME"]
    for c in mp_alts:
        if c in out.columns and c != "MP":
            rename_map[c] = "MP"
            break

    # Team abbrev
    team_alts = ["TEAM_ABBREVIATION", "TEAM", "Team", "tm", "TM", "team_abbrev", "team"]
    for c in team_alts:
        if c in out.columns and c != "team_abbrev":
            rename_map[c] = "team_abbrev"
            break

    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "season" in out.columns:
        out["season"] = out["season"].astype(str).str.strip()

    return out


def _build_master_df(base: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    b = _standardize_common_columns(base)
    a = _standardize_common_columns(adv) if adv is not None else None

    if a is not None and ("player_id" in b.columns) and ("player_id" in a.columns):
        join_keys = ["player_id", "season"]
    else:
        join_keys = ["player_name", "season"]

    df = b if a is None else b.merge(a, on=join_keys, how="left", suffixes=("", "_adv"))

    if "season" not in df.columns:
        raise ValueError("career_progression: Could not find a 'season' column in the provided data.")

    df["season_sort"] = df["season"].apply(_norm_season_to_sort_key)
    return df


# ----------------------------
# Derived metrics (robust: always produce PER_GAME + TOTAL consistently)
# ----------------------------
def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # numeric conversion
    for col in ["PTS", "AST", "REB", "STL", "BLK", "TOV", "GP", "MP", "FTM", "FTA"]:
        if col in out.columns:
            out[col] = _safe_float_series(out[col])

    def _looks_like_per_game(col: str) -> bool:
        if col not in out.columns:
            return False

        s = out[col].values.astype(float)
        mx = float(np.nanmax(s)) if len(out) else np.nan
        if not np.isfinite(mx):
            return False

        gp_ok = False
        if "GP" in out.columns:
            gp = out["GP"].values.astype(float)
            gp_med = float(np.nanmedian(gp)) if len(out) else np.nan
            gp_ok = np.isfinite(gp_med) and (1 <= gp_med <= 82)

        # Require GP to look like a season count to classify as per-game.
        if not gp_ok:
            return False

        if col == "PTS":
            return mx <= 60
        if col in {"AST", "REB"}:
            return mx <= 25
        if col in {"STL", "BLK", "TOV"}:
            return mx <= 10
        if col == "MP":
            # per-game minutes usually <= 48 (or low 40s)
            return mx <= 48
        if col in {"FTM", "FTA"}:
            # per-game FT attempts/makes usually within these ranges
            return mx <= 20

        return False

    if "GP" in out.columns:
        gp = out["GP"].replace(0, np.nan)

        # PTS
        if "PTS" in out.columns:
            if _looks_like_per_game("PTS"):
                out["PTS_PER_GAME"] = out["PTS"]
                out["PTS_TOTAL"] = out["PTS"] * gp
            else:
                out["PTS_TOTAL"] = out["PTS"]
                out["PTS_PER_GAME"] = out["PTS"] / gp

        # AST
        if "AST" in out.columns:
            if _looks_like_per_game("AST"):
                out["AST_PER_GAME"] = out["AST"]
                out["AST_TOTAL"] = out["AST"] * gp
            else:
                out["AST_TOTAL"] = out["AST"]
                out["AST_PER_GAME"] = out["AST"] / gp

        # REB
        if "REB" in out.columns:
            if _looks_like_per_game("REB"):
                out["REB_PER_GAME"] = out["REB"]
                out["REB_TOTAL"] = out["REB"] * gp
            else:
                out["REB_TOTAL"] = out["REB"]
                out["REB_PER_GAME"] = out["REB"] / gp

        # Minutes
        if "MP" in out.columns:
            if _looks_like_per_game("MP"):
                out["MP_PER_GAME"] = out["MP"]
                out["MP_TOTAL"] = out["MP"] * gp
            else:
                out["MP_TOTAL"] = out["MP"]
                out["MP_PER_GAME"] = out["MP"] / gp

        # Free throws
        if "FTM" in out.columns:
            if _looks_like_per_game("FTM"):
                out["FTM_PER_GAME"] = out["FTM"]
                out["FTM_TOTAL"] = out["FTM"] * gp
            else:
                out["FTM_TOTAL"] = out["FTM"]
                out["FTM_PER_GAME"] = out["FTM"] / gp

        if "FTA" in out.columns:
            if _looks_like_per_game("FTA"):
                out["FTA_PER_GAME"] = out["FTA"]
                out["FTA_TOTAL"] = out["FTA"] * gp
            else:
                out["FTA_TOTAL"] = out["FTA"]
                out["FTA_PER_GAME"] = out["FTA"] / gp

    # FT%
    if "FT_PCT" not in out.columns:
        if ("FTM_TOTAL" in out.columns) and ("FTA_TOTAL" in out.columns):
            denom = out["FTA_TOTAL"].replace(0, np.nan)
            out["FT_PCT"] = out["FTM_TOTAL"] / denom
        elif ("FTM" in out.columns) and ("FTA" in out.columns):
            denom = out["FTA"].replace(0, np.nan)
            out["FT_PCT"] = out["FTM"] / denom

    # Fouls drawn proxy – always total if possible
    if "FTA_TOTAL" in out.columns:
        out["FOUL_DRAWN_TOTAL"] = out["FTA_TOTAL"]
    elif "FTA" in out.columns:
        out["FOUL_DRAWN_TOTAL"] = out["FTA"]

    # Optional composite metrics (use what exists; keeps design identical)
    needed_eff = ["PTS_PER_GAME", "REB_PER_GAME", "AST_PER_GAME", "STL", "BLK", "TOV"]
    if all(c in out.columns for c in needed_eff):
        out["EFF_PER_GAME"] = (
            out["PTS_PER_GAME"] + out["REB_PER_GAME"] + out["AST_PER_GAME"] + out["STL"] + out["BLK"] - out["TOV"]
        )
        if "GP" in out.columns:
            gp = out["GP"].replace(0, np.nan)
            out["EFF_TOTAL"] = out["EFF_PER_GAME"] * gp

    needed_contr = ["PTS_PER_GAME", "AST_PER_GAME", "REB_PER_GAME"]
    if all(c in out.columns for c in needed_contr):
        out["CONTR_PER_GAME"] = out["PTS_PER_GAME"] + 1.5 * out["AST_PER_GAME"] + 1.2 * out["REB_PER_GAME"]
        if "GP" in out.columns:
            gp = out["GP"].replace(0, np.nan)
            out["CONTR_TOTAL"] = out["CONTR_PER_GAME"] * gp

    return out


def _get_metric_catalog(df: pd.DataFrame):
    preferred_order = [
        # Per game (primary)
        "PTS_PER_GAME", "AST_PER_GAME", "REB_PER_GAME",

        # Totals (preferred over raw PTS/AST/REB)
        "PTS_TOTAL", "AST_TOTAL", "REB_TOTAL",

        # Free throws (prefer totals if exist)
        "FTM_TOTAL", "FTA_TOTAL",
        "FTM", "FTA", "FT_PCT", "FOUL_DRAWN_TOTAL",

        # Other box-score
        "STL", "BLK", "TOV", "GP", "MP", "MP_PER_GAME", "MP_TOTAL",

        # Composite
        "EFF_PER_GAME", "EFF_TOTAL",
        "CONTR_PER_GAME", "CONTR_TOTAL",

        # Raw fallback (only if totals are missing)
        "PTS", "AST", "REB",
    ]

    adv_candidates = ["USG%", "TS%", "eFG%", "PER", "BPM", "VORP", "WS", "WS/48", "ORTG", "DRTG"]
    alt_adv_candidates = ["USG_PCT", "TS_PCT", "EFG_PCT", "PER", "BPM", "VORP", "WS", "WS48", "ORtg", "DRtg"]

    all_candidates = preferred_order + adv_candidates + alt_adv_candidates

    existing = []
    for c in all_candidates:
        if c in df.columns:
            existing.append(c)

    # Deduplicate, preserve order
    seen = set()
    cleaned = []
    for c in existing:
        if c not in seen:
            cleaned.append(c)
            seen.add(c)

    # If TOTAL exists, drop the raw (ambiguous) column to avoid wrong "Totals"
    if "PTS_TOTAL" in df.columns and "PTS" in cleaned:
        cleaned.remove("PTS")
    if "AST_TOTAL" in df.columns and "AST" in cleaned:
        cleaned.remove("AST")
    if "REB_TOTAL" in df.columns and "REB" in cleaned:
        cleaned.remove("REB")

    labels = {
        # Per-game
        "PTS_PER_GAME": "Points Per Game",
        "AST_PER_GAME": "Assists Per Game",
        "REB_PER_GAME": "Rebounds Per Game",

        # Totals
        "PTS_TOTAL": "Points (Total)",
        "AST_TOTAL": "Assists (Total)",
        "REB_TOTAL": "Rebounds (Total)",

        # Minutes
        "MP": "Minutes",
        "MP_PER_GAME": "Minutes Per Game",
        "MP_TOTAL": "Minutes (Total)",

        # Raw (fallback only)
        "PTS": "Points",
        "AST": "Assists",
        "REB": "Rebounds",

        # Other
        "STL": "Steals",
        "BLK": "Blocks",
        "TOV": "Turnovers",
        "GP": "Games Played",

        # Composite
        "EFF_PER_GAME": "Efficiency Per Game (EFF/GP)",
        "EFF_TOTAL": "Efficiency (Total)",
        "CONTR_PER_GAME": "Contribution Per Game",
        "CONTR_TOTAL": "Contribution Score (Total)",

        # Free throws
        "FTM": "Free Throws Made",
        "FTA": "Free Throws Attempted",
        "FTM_TOTAL": "Free Throws Made (Total)",
        "FTA_TOTAL": "Free Throws Attempted (Total)",
        "FT_PCT": "Free Throw %",
        "FOUL_DRAWN_TOTAL": "Fouls Drawn (Proxy) — Total FTA",

        # Advanced
        "USG%": "Usage % (Advanced)",
        "TS%": "True Shooting % (Advanced)",
        "eFG%": "eFG% (Advanced)",

        "USG_PCT": "Usage % (Advanced)",
        "TS_PCT": "True Shooting % (Advanced)",
        "EFG_PCT": "eFG% (Advanced)",

        "WS/48": "Win Shares / 48 (Advanced)",
        "WS48": "Win Shares / 48 (Advanced)",

        "ORtg": "Offensive Rating (Advanced)",
        "DRtg": "Defensive Rating (Advanced)",
        "ORTG": "Offensive Rating (Advanced)",
        "DRTG": "Defensive Rating (Advanced)",

        "PER": "Player Efficiency Rating (Advanced)",
        "BPM": "Box Plus/Minus (Advanced)",
        "VORP": "VORP (Advanced)",
        "WS": "Win Shares (Advanced)",
    }

    return [(col, labels.get(col, col)) for col in cleaned]


def _metric_higher_is_better(metric_col: str) -> bool:
    lower_better = {"TOV", "DRTG", "DRtg"}
    return metric_col not in lower_better


def _is_percent_metric(metric_col: str) -> bool:
    m = str(metric_col)
    if "%" in m:
        return True
    if m in {"TS_PCT", "EFG_PCT", "USG_PCT", "FT_PCT"}:
        return True
    return False


def _format_metric_value(metric_col: str, val: float) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""

    if _is_percent_metric(metric_col):
        if val <= 1.5:
            return f"{val * 100.0:.1f}%"
        return f"{val:.1f}%"

    if metric_col == "GP":
        return str(int(round(val)))

    if abs(val - round(val)) < 1e-9 and abs(val) >= 10:
        return str(int(round(val)))

    return f"{val:.2f}"


def _parse_rank_number(rank_text: str) -> int:
    if not isinstance(rank_text, str):
        return 10**9
    m = re.search(r"Ranked\s+(\d+)\s+out\s+of\s+(\d+)", rank_text)
    if not m:
        return 10**9
    return int(m.group(1))


# ----------------------------
# Team inference + logos
# ----------------------------
def _find_team_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "team_abbrev", "TEAM_ABBREV", "TEAM_ABBREVIATION", "team", "TEAM", "Team",
        "team_name", "TEAM_NAME", "TeamName", "tm", "TM",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_team_code(raw_team: str) -> str | None:
    if raw_team is None:
        return None
    s = str(raw_team).strip().lower()
    if s == "":
        return None

    if s in {"por", "portland", "trail blazers", "portland trail blazers", "portland blazers"}:
        return "POR"
    if s in {"was", "wsh", "washington", "wizards", "washington wizards"}:
        return "WAS"

    if "portland" in s or "blazer" in s:
        return "POR"
    if "washington" in s or "wizard" in s:
        return "WAS"

    return None


def _build_season_team_map(df_deni: pd.DataFrame) -> dict:
    team_col = _find_team_col(df_deni)
    out = {}
    if team_col is None:
        return out

    for _, r in df_deni.iterrows():
        season = str(r.get("season", "")).strip()
        if season == "":
            continue
        code = _normalize_team_code(r.get(team_col, None))
        if code is None:
            continue
        out[season] = code
    return out


@st.cache_data(show_spinner=False)
def _load_logo_as_data_uri(path: str) -> str | None:
    if not path or (not os.path.exists(path)):
        return None
    try:
        with open(path, "rb") as f:
            b = f.read()
        enc = base64.b64encode(b).decode("utf-8")
        return f"data:image/png;base64,{enc}"
    except Exception:
        return None


def _add_team_logos_on_points(fig: go.Figure, df_series: pd.DataFrame, season_team: dict, chosen_metric: str):
    y_vals = pd.to_numeric(df_series["deni_val"], errors="coerce").astype(float).values
    y_vals = y_vals[np.isfinite(y_vals)]
    if len(y_vals) == 0:
        return

    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    y_rng = max(1e-9, (y_max - y_min))

    sizey = y_rng * 0.12
    sizex = 0.75

    for _, r in df_series.iterrows():
        season = str(r.get("season", ""))
        y = r.get("deni_val", np.nan)
        if season == "" or (y is None) or (isinstance(y, float) and np.isnan(y)):
            continue

        team = season_team.get(season, None)
        if team is None:
            continue

        logo_path = TEAM_LOGO_PATHS.get(team, None)
        logo_uri = _load_logo_as_data_uri(logo_path) if logo_path else None
        if not logo_uri:
            continue

        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="x",
                yref="y",
                x=season,
                y=float(y),
                xanchor="center",
                yanchor="middle",
                sizex=sizex,
                sizey=sizey,
                opacity=0.98,
                layer="above",
            )
        )


def _apply_team_tint_background(team_code: str | None):
    if team_code is None:
        return
    if team_code not in TEAM_BG_TINT:
        return

    c1, c2 = TEAM_BG_TINT[team_code]
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: linear-gradient(180deg, {c1} 0%, {c2} 55%);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# CSV rankings integration
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_league_all_seasons_base() -> pd.DataFrame:
    candidates = [
        "league_all_seasons_base.csv",
        os.path.join("data", "league_all_seasons_base.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_deni_season_rankings_csv() -> pd.DataFrame:
    candidates = [
        "deni_season_rankings.csv",
        os.path.join("data", "deni_season_rankings.csv"),
        os.path.join("assets", "deni_season_rankings.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)

            if "SEASON" not in df.columns:
                for alt in ["season", "Season"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "SEASON"})
                        break

            if "METRIC" not in df.columns:
                for alt in ["metric", "Metric"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "METRIC"})
                        break

            if "DENI_VALUE" not in df.columns:
                for alt in ["PLAYER_VALUE", "DENI_VAL", "value"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "DENI_VALUE"})
                        break

            for c in ["SEASON", "METRIC"]:
                if c in df.columns:
                    df[c] = df[c].astype(str)

            for c in ["DENI_VALUE", "ABS_RANK", "TOTAL_PLAYERS", "PERCENTILE"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            keep = [c for c in ["SEASON", "METRIC", "DENI_VALUE", "ABS_RANK", "TOTAL_PLAYERS"] if c in df.columns]
            return df[keep].copy()

    return pd.DataFrame(columns=["SEASON", "METRIC", "DENI_VALUE", "ABS_RANK", "TOTAL_PLAYERS"])


def _metric_to_rankings_csv_code(metric_col: str) -> str | None:
    mapping = {
        "PTS_PER_GAME": "PTS",
        "REB_PER_GAME": "REB",
        "AST_PER_GAME": "AST",

        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",

        "FG_PCT": "FG_PCT",
        "FG%": "FG_PCT",
        "FG3_PCT": "FG3_PCT",
        "FG3%": "FG3_PCT",
        "3P_PCT": "FG3_PCT",
        "3P%": "FG3_PCT",

        "PLUS_MINUS": "PLUS_MINUS",
        "+/-": "PLUS_MINUS",
        "PLUSMINUS": "PLUS_MINUS",
    }
    return mapping.get(str(metric_col), None)


def _lookup_rank_from_csv(rank_df: pd.DataFrame, season: str, metric_code: str):
    if rank_df is None or rank_df.empty:
        return np.nan, np.nan

    m = (
        (rank_df["SEASON"].astype(str) == str(season))
        & (rank_df["METRIC"].astype(str) == str(metric_code))
    )
    if not m.any():
        return np.nan, np.nan

    row = rank_df.loc[m].iloc[0]
    r = row.get("ABS_RANK", np.nan)
    t = row.get("TOTAL_PLAYERS", np.nan)

    try:
        r = float(r)
    except Exception:
        r = np.nan
    try:
        t = float(t)
    except Exception:
        t = np.nan

    return r, t


def _compute_rank_for_player(df_season: pd.DataFrame, deni_key_col: str, deni_key_val, metric_col: str):
    if metric_col not in df_season.columns:
        return np.nan, 0, np.nan, np.nan

    s = _safe_float_series(df_season[metric_col])
    deni_row = df_season[df_season[deni_key_col] == deni_key_val]
    if len(deni_row) == 0:
        return np.nan, 0, np.nan, np.nan

    deni_val = float(pd.to_numeric(pd.Series([deni_row.iloc[0].get(metric_col, np.nan)]), errors="coerce").iloc[0])
    league_avg = float(np.nanmean(s.values)) if len(s) > 0 else np.nan

    higher_is_better = _metric_higher_is_better(metric_col)
    total_players = int(s.notna().sum())
    if total_players <= 0:
        return np.nan, 0, deni_val, league_avg

    ranks = s.rank(ascending=not higher_is_better, method="min")
    deni_idx = df_season.index[df_season[deni_key_col] == deni_key_val]
    if len(deni_idx) == 0:
        return np.nan, total_players, deni_val, league_avg

    deni_rank = float(ranks.loc[deni_idx[0]])
    return deni_rank, total_players, deni_val, league_avg


def _compute_rank_for_player_prefer_csv(
    df_season: pd.DataFrame,
    season: str,
    deni_key_col: str,
    deni_key_val,
    metric_col: str,
    rank_df: pd.DataFrame
):
    deni_rank, total_players, deni_val, league_avg = _compute_rank_for_player(
        df_season, deni_key_col, deni_key_val, metric_col
    )

    code = _metric_to_rankings_csv_code(metric_col)
    if code is None:
        return deni_rank, total_players, deni_val, league_avg

    r_csv, t_csv = _lookup_rank_from_csv(rank_df, season, code)
    if not np.isnan(r_csv) and not np.isnan(t_csv):
        return float(r_csv), float(t_csv), deni_val, league_avg

    return deni_rank, total_players, deni_val, league_avg


# ----------------------------
# Compare-table styling
# ----------------------------
def _soft_cell_bg(hex_color: str, alpha: float = 0.16) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _style_compare_two_seasons(comp_df: pd.DataFrame, season_a: str, season_b: str, metric_col_lookup: dict):
    col_a = f"{season_a} (Deni)"
    col_b = f"{season_b} (Deni)"

    green_bg = _soft_cell_bg("#43aa8b", 0.16)  # winner
    red_bg = _soft_cell_bg("#f94144", 0.14)  # loser

    def _to_num(x):
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        s = s.replace("%", "")
        try:
            return float(s)
        except Exception:
            return np.nan

    def _apply_row(row: pd.Series):
        styles = [""] * len(row)

        label = str(row.get("Metric", ""))
        raw_metric = metric_col_lookup.get(label, None)

        a = _to_num(row.get(col_a, ""))
        b = _to_num(row.get(col_b, ""))

        if raw_metric is None or np.isnan(a) or np.isnan(b):
            return styles

        higher_is_better = _metric_higher_is_better(raw_metric)
        if a == b:
            return styles

        a_better = (a > b) if higher_is_better else (a < b)

        idx_a = row.index.get_loc(col_a)
        idx_b = row.index.get_loc(col_b)

        if a_better:
            styles[idx_a] = f"background-color: {green_bg};"
            styles[idx_b] = f"background-color: {red_bg};"
        else:
            styles[idx_b] = f"background-color: {green_bg};"
            styles[idx_a] = f"background-color: {red_bg};"

        return styles

    styler = comp_df.style.apply(_apply_row, axis=1)
    styler = styler.set_properties(**{
        "border-color": "rgba(0,0,0,0.06)",
        "font-size": "0.95rem",
    })
    return styler


# ----------------------------
# NEW: Rank bar chart for selected season + metric
# ----------------------------
def _render_rank_barchart_for_selected_metric(
    df_season: pd.DataFrame,
    season: str,
    deni_key_col: str,
    deni_key_val,
    metric_col: str,
    metric_label: str,
    rank_df: pd.DataFrame
):
    deni_rank, total_players, deni_val, league_avg_val = _compute_rank_for_player_prefer_csv(
        df_season, season, deni_key_col, deni_key_val, metric_col, rank_df
    )

    deni_text = _format_metric_value(metric_col, deni_val)
    if (not np.isnan(deni_rank)) and (not np.isnan(total_players)) and total_players > 0:
        deni_text = f"{deni_text}<br>Rank {int(deni_rank)} out of {int(total_players)}"

    league_text = _format_metric_value(metric_col, league_avg_val)

    TEAM_orange = "#FB8500"
    BLUE_COLOR = "#023047"
    OUTLINE = "rgba(0,0,0,0.55)"

    x_labels = ["Deni", "League Avg"]
    y_vals = [deni_val, league_avg_val]
    texts = [deni_text, league_text]
    colors = [BLUE_COLOR, TEAM_orange]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_vals,
            text=texts,
            textposition="inside",
            insidetextanchor="middle",
            marker=dict(
                color=colors,
                line=dict(width=1.2, color=OUTLINE),
            ),
            opacity=0.95,
            width=0.38,
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
    )

    y_clean = [v for v in y_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(y_clean) > 0:
        y_max = float(max(y_clean))
        pad = max(1e-9, 0.12 * abs(y_max))
        y_range = [0, y_max + pad] if y_max >= 0 else [y_max - pad, 0]
    else:
        y_range = None

    fig.update_layout(
        title=dict(
            text=f"{metric_label} — Deni vs League Avg",
            x=0.02,
            xanchor="left",
        ),
        height=390,
        margin=dict(l=18, r=18, t=60, b=35),
        showlegend=False,
        font=dict(size=15),
        title_font=dict(size=18),
        bargap=0.55,
        bargroupgap=0.20,
    )

    fig.update_xaxes(
        tickfont=dict(size=16),
        showline=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Value",
        tickfont=dict(size=14),
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        rangemode="tozero",
        range=y_range,
    )

    fig.update_traces(textfont=dict(size=16, color="white"))

    st.plotly_chart(fig, width="stretch")


# ----------------------------
# Main render
# ----------------------------
def render_career_progression(base: pd.DataFrame, adv: pd.DataFrame):
    st.subheader("Career Progression Over Seasons")
    st.caption(
        "### This page focuses on Deni Avdija’s development over time.\n\n"
        "### It shows how his key performance metrics evolve from season to season, allowing us to track trends, improvements, and changes throughout his NBA career."
    )
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
            background: transparent !important;
            box-shadow: none !important;
            border: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stContainer"][data-border="true"]{
            background: rgba(255,255,255,0.96) !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            border-radius: 14px !important;
            padding: 12px 14px !important;
            box-shadow: 0 8px 22px rgba(0,0,0,0.06) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = _build_master_df(base, adv)
    df = _add_derived_metrics(df)

    deni_key_col, deni_key_val = _find_deni_key(df)
    if deni_key_col is None:
        st.error("Could not find Deni Avdija in the provided tables (by player_name).")
        return

    df_deni = df[df[deni_key_col] == deni_key_val].copy()
    if len(df_deni) == 0:
        st.error("Found Deni identifier but no season rows matched.")
        return

    df_deni = df_deni.sort_values("season_sort")
    seasons = df_deni["season"].astype(str).tolist()
    if len(seasons) == 0:
        st.error("No seasons found for Deni.")
        return

    season_team = _build_season_team_map(df_deni)

    metric_options = _get_metric_catalog(df)
    if len(metric_options) == 0:
        st.error("No selectable metrics found in the provided data.")
        return

    metric_labels = {col: label for col, label in metric_options}
    metric_cols = [col for col, _ in metric_options]

    default_metric = "PTS_PER_GAME" if "PTS_PER_GAME" in metric_cols else metric_cols[0]
    c_sel, c_spacer = st.columns([1, 4])
    with c_sel:
        with st.container(border=True):
            chosen_metric = st.selectbox(
                "Choose Y-axis metric",
                metric_cols,
                index=metric_cols.index(default_metric),
                format_func=lambda c: metric_labels.get(c, c),
                key="career_y_metric_select",
            )

    df_plot_all = df[df["season"].astype(str).isin(seasons)].copy()
    df_plot_all["season"] = df_plot_all["season"].astype(str)
    if chosen_metric in df_plot_all.columns:
        df_plot_all[chosen_metric] = _safe_float_series(df_plot_all[chosen_metric])
    else:
        df_plot_all[chosen_metric] = np.nan

    # ----------------------------
    # FIX: League Avg should come from league source + rotation filter (not all players)
    # ----------------------------
    def _rotation_filter(d: pd.DataFrame) -> pd.DataFrame:
        x = d.copy()
        if "GP" in x.columns:
            x = x[_safe_float_series(x["GP"]) >= 15]
        if "MP_PER_GAME" in x.columns:
            x = x[_safe_float_series(x["MP_PER_GAME"]) >= 10]
        elif "MP" in x.columns and "GP" in x.columns:
            mp_pg = _safe_float_series(x["MP"]) / _safe_float_series(x["GP"]).replace(0, np.nan)
            x = x[mp_pg >= 10]
        return x

    league_src = _load_league_all_seasons_base()
    if not league_src.empty:
        league_src = _standardize_common_columns(league_src)
        league_src = _add_derived_metrics(league_src)
        league_src["season"] = league_src["season"].astype(str)
        league_src = league_src[league_src["season"].isin(seasons)]
        league_src = _rotation_filter(league_src)

        if chosen_metric in league_src.columns:
            league_src[chosen_metric] = _safe_float_series(league_src[chosen_metric])
            league_avg = (
                league_src.groupby("season", as_index=False)[chosen_metric]
                .mean()
                .rename(columns={chosen_metric: "league_avg"})
            )
        else:
            tmp = _rotation_filter(df_plot_all)
            league_avg = (
                tmp.groupby("season", as_index=False)[chosen_metric]
                .mean(numeric_only=True)
                .rename(columns={chosen_metric: "league_avg"})
            )
    else:
        tmp = _rotation_filter(df_plot_all)
        league_avg = (
            tmp.groupby("season", as_index=False)[chosen_metric]
            .mean(numeric_only=True)
            .rename(columns={chosen_metric: "league_avg"})
        )

    df_deni_plot = df_deni.copy()
    df_deni_plot["season"] = df_deni_plot["season"].astype(str)
    df_deni_plot[chosen_metric] = _safe_float_series(df_deni_plot[chosen_metric]) if chosen_metric in df_deni_plot.columns else np.nan
    df_deni_plot = df_deni_plot[["season", chosen_metric]].rename(columns={chosen_metric: "deni_val"})

    df_series = pd.DataFrame({"season": seasons})
    df_series = df_series.merge(df_deni_plot, on="season", how="left")
    df_series = df_series.merge(league_avg, on="season", how="left")

    default_season = seasons[-1]
    if "career_selected_season" not in st.session_state:
        st.session_state["career_selected_season"] = default_season

    selected_season = st.session_state["career_selected_season"]

    deni_sizes = []
    deni_line_widths = []
    for s in df_series["season"].astype(str).tolist():
        if s == str(selected_season):
            deni_sizes.append(16)
            deni_line_widths.append(3)
        else:
            deni_sizes.append(12)
            deni_line_widths.append(2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_series["season"],
        y=df_series["league_avg"],
        mode="lines+markers",
        name="League Average",
        opacity=0.85,
        line=dict(width=3, dash="dot", color="#FB8500"),
        marker=dict(size=7, color="#FB8500"),
        hovertemplate="Season: %{x}<br>League Avg: %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_series["season"],
        y=df_series["deni_val"],
        mode="lines+markers",
        name="Deni Avdija",
        line=dict(width=4, color="#023047"),
        marker=dict(size=deni_sizes, color="#023047", line=dict(width=deni_line_widths)),
        hovertemplate="Season: %{x}<br>Deni: %{y:.3f}<extra></extra>",
    ))

    ticktext = []
    for s in seasons:
        team = season_team.get(str(s), None)
        if team is None:
            ticktext.append(str(s))
        else:
            ticktext.append(f"{s}<br>{team}")

    fig.update_layout(
        title=f"Evolution by Season — {metric_labels.get(chosen_metric, chosen_metric)}",
        xaxis_title="Season",
        yaxis_title=metric_labels.get(chosen_metric, chosen_metric),
        margin=dict(l=30, r=30, t=70, b=50),
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=seasons,
        tickmode="array",
        tickvals=seasons,
        ticktext=ticktext,
    )

    annotations = [{
        "x": 0.01,
        "y": 0.98,
        "xref": "paper",
        "yref": "paper",
        "text": "Click a logo to select a season",
        "showarrow": False,
        "opacity": 0.95,
        "font": {"size": 14},
        "xanchor": "left",
        "yanchor": "top",
    }]

    if len(df_series) > 0 and not pd.isna(df_series["league_avg"].iloc[0]):
        annotations.append({
            "x": str(df_series["season"].iloc[0]),
            "y": float(df_series["league_avg"].iloc[0]),
            "xref": "x",
            "yref": "y",
            "text": "League Avg",
            "showarrow": False,
            "opacity": 0.9,
            "font": {"size": 11, "color": "red"},
            "xanchor": "left",
            "yanchor": "bottom",
        })

    if len(df_series) > 0 and not pd.isna(df_series["deni_val"].iloc[0]):
        annotations.append({
            "x": str(df_series["season"].iloc[0]),
            "y": float(df_series["deni_val"].iloc[0]),
            "xref": "x",
            "yref": "y",
            "text": "Deni",
            "showarrow": False,
            "opacity": 0.9,
            "font": {"size": 11, "color": "#023047"},
            "xanchor": "left",
            "yanchor": "top",
        })

    fig.update_layout(annotations=annotations)

    _add_team_logos_on_points(fig, df_series, season_team, chosen_metric)

    st.caption("Click a season point to open the season details below.")
    event = st.plotly_chart(
        fig,
        key=CHART_KEY,
        width="stretch",
        on_select="rerun",
        selection_mode=("points",),
    )

    try:
        if event and event.selection and event.selection.points:
            p = event.selection.points[0]
            if "x" in p:
                st.session_state["career_selected_season"] = str(p["x"])
    except Exception:
        pass

    selected_season = st.session_state["career_selected_season"]
    _apply_team_tint_background(season_team.get(str(selected_season), None))

    # ----------------------------
    # Selected season details
    # ----------------------------
    st.markdown("---")

    rank_metric_order = [
        "PTS_PER_GAME", "AST_PER_GAME", "REB_PER_GAME",
        "FTA", "FT_PCT", "CONTR_PER_GAME", "TS%",
        "USG_PCT",
    ]
    summary_metrics_order = [
        "PTS_PER_GAME", "AST_PER_GAME", "REB_PER_GAME",
        "FTA", "FT_PCT", "CONTR_PER_GAME", "TS%",
        "STL",
        "USG_PCT",
    ]
    rank_metrics = [m for m in rank_metric_order if m in df_plot_all.columns]
    rank_metrics1 = [m for m in summary_metrics_order if m in df_plot_all.columns]

    df_season = df_plot_all[df_plot_all["season"] == str(selected_season)].copy()
    if len(df_season) == 0:
        st.info("No league rows found for this season in the provided data.")
        return

    prev_season = None
    if selected_season in seasons:
        idx = seasons.index(selected_season)
        if idx > 0:
            prev_season = seasons[idx - 1]

    df_prev = None
    if prev_season is not None:
        df_prev = df_plot_all[df_plot_all["season"] == str(prev_season)].copy()

    rank_df = _load_deni_season_rankings_csv()

    rows = []
    best_rank_item = None
    best_improve_item = None
    best_drop_item = None

    for m in rank_metrics1:
        deni_rank, total_players, deni_val, league_avg_val = _compute_rank_for_player_prefer_csv(
            df_season, selected_season, deni_key_col, deni_key_val, m, rank_df
        )

        trend = ""
        delta_places = None
        if prev_season is not None and df_prev is not None and len(df_prev) > 0:
            prev_rank, prev_total, _, _ = _compute_rank_for_player_prefer_csv(
                df_prev, prev_season, deni_key_col, deni_key_val, m, rank_df
            )

            if (not np.isnan(deni_rank)) and (not np.isnan(prev_rank)):
                delta_places = int(round(prev_rank - deni_rank))
                if delta_places > 0:
                    trend = f"▲ +{delta_places}"
                elif delta_places < 0:
                    trend = f"▼ {delta_places}"
                else:
                    trend = "— 0"

        rank_text = ""
        if not np.isnan(deni_rank) and (not np.isnan(total_players)) and total_players > 0:
            rank_text = f"Ranked {int(deni_rank)} out of {int(total_players)}"

        if not np.isnan(deni_rank) and (not np.isnan(total_players)) and total_players > 0:
            if (best_rank_item is None) or (deni_rank < best_rank_item[0]):
                best_rank_item = (deni_rank, metric_labels.get(m, m), rank_text)

        if delta_places is not None:
            if delta_places > 0:
                if (best_improve_item is None) or (delta_places > best_improve_item[0]):
                    best_improve_item = (delta_places, metric_labels.get(m, m))
            if delta_places < 0:
                if (best_drop_item is None) or (delta_places < best_drop_item[0]):
                    best_drop_item = (delta_places, metric_labels.get(m, m))

        rows.append({
            "Metric": metric_labels.get(m, m),
            "Deni Value": _format_metric_value(m, deni_val),
            "League Avg": _format_metric_value(m, league_avg_val),
            "Deni Rank": rank_text,
            "Trend": trend,
        })

    display_df = pd.DataFrame(rows)
    display_df["_rk"] = display_df["Deni Rank"].apply(_parse_rank_number)
    display_df = display_df.sort_values(["_rk", "Metric"]).drop(columns=["_rk"])

    st.subheader(f"Season Summary — {selected_season}")
    c1, c2, c3 = st.columns(3)

    with c1:
        if best_rank_item is None:
            st.metric("Best Rank", "—")
        else:
            st.metric("Best Rank", best_rank_item[1], best_rank_item[2])

    with c2:
        if best_improve_item is None:
            st.metric("Biggest Improvement", "—")
        else:
            st.metric("Biggest Improvement", best_improve_item[1], f"+{best_improve_item[0]} places")

    with c3:
        if best_drop_item is None:
            st.metric("Biggest Drop", "—")
        else:
            st.metric("Biggest Drop", best_drop_item[1], f"{best_drop_item[0]} places")

    st.subheader(f"Season Ranking (Chart) — {selected_season}")

    c_sel2, c_spacer2 = st.columns([1, 4])
    with c_sel2:
        with st.container(border=True):
            chart_metric = st.selectbox(
                "Choose metric for season chart",
                options=rank_metrics,
                index=rank_metrics.index("PTS_PER_GAME") if "PTS_PER_GAME" in rank_metrics else 0,
                format_func=lambda c: metric_labels.get(c, c),
                key="career_season_chart_metric",
            )

    _render_rank_barchart_for_selected_metric(
        df_season=df_season,
        season=str(selected_season),
        deni_key_col=deni_key_col,
        deni_key_val=deni_key_val,
        metric_col=chart_metric,
        metric_label=metric_labels.get(chart_metric, chart_metric),
        rank_df=rank_df,
    )

    with st.expander("Show full season table (all metrics)"):
        try:
            st.dataframe(display_df, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(display_df.reset_index(drop=True), width="stretch")

    # ----------------------------
    # Compare Two Seasons
    # ----------------------------
    st.markdown("---")
    st.subheader("Compare Two Seasons")

    default_pair = []
    if prev_season is not None:
        default_pair = [prev_season, selected_season]
    elif len(seasons) >= 2:
        default_pair = [seasons[0], selected_season]
    else:
        default_pair = [selected_season, selected_season]

    c_ms, c_spacer3 = st.columns([1, 4])
    with c_ms:
        with st.container(border=True):
            picked = st.multiselect(
                "Pick exactly 2 seasons to compare",
                options=seasons,
                default=default_pair,
                max_selections=2,
                key="career_compare_two_seasons",
            )

    if len(picked) != 2:
        st.info("Select exactly 2 seasons to see the comparison table.")
        return

    season_a, season_b = picked[0], picked[1]
    df_a = df_plot_all[df_plot_all["season"] == str(season_a)].copy()
    df_b = df_plot_all[df_plot_all["season"] == str(season_b)].copy()

    if len(df_a) == 0 or len(df_b) == 0:
        st.info("Missing league data for one of the selected seasons.")
        return

    comp_rows = []
    for m in rank_metrics:
        ra, ta, va, _ = _compute_rank_for_player_prefer_csv(df_a, season_a, deni_key_col, deni_key_val, m, rank_df)
        rb, tb, vb, _ = _compute_rank_for_player_prefer_csv(df_b, season_b, deni_key_col, deni_key_val, m, rank_df)

        val_change = np.nan
        if (not np.isnan(va)) and (not np.isnan(vb)):
            val_change = vb - va

        rank_change = np.nan
        if (not np.isnan(ra)) and (not np.isnan(rb)):
            rank_change = int(round(ra - rb))

        comp_rows.append({
            "Metric": metric_labels.get(m, m),
            f"{season_a} (Deni)": _format_metric_value(m, va),
            f"{season_b} (Deni)": _format_metric_value(m, vb),
            "Value differences": "" if np.isnan(val_change) else _format_metric_value(m, val_change),
            "Rank differences": "" if np.isnan(rank_change) else (f"+{int(rank_change)}" if rank_change > 0 else str(int(rank_change))),
        })

    comp_df = pd.DataFrame(comp_rows)

    st.caption("Rank differences: positive means Deni improved (moved up) in league ranking.")
    st.caption("Green = better season for that metric. Red = worse season (soft highlight).")

    metric_label_to_raw = {}
    for raw, lbl in metric_labels.items():
        if lbl not in metric_label_to_raw:
            metric_label_to_raw[lbl] = raw

    styled_comp_df = _style_compare_two_seasons(comp_df, season_a, season_b, metric_label_to_raw)

    try:
        st.dataframe(styled_comp_df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(comp_df.reset_index(drop=True), width="stretch")
