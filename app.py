import streamlit as st
import os
import base64
import pandas as pd
from data_loader import load_csv
from components.shot_map import render_shot_map
from components.player_compare import render_player_compare
from components.usage_vs_defense import render_usage_vs_defense
from components.career_progression import render_career_progression
from components.deni_labron_casspi import render_deni_labron_casspi
import streamlit.components.v1 as components

DENI_ID = 1630166


def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


TEAM_LOGO_PATHS = {
    "POR": os.path.join("data", "images", "trailblazers_logo.png"),
}


def wl_to_emoji(wl: str) -> str:
    """
    WL column: 'W' / 'L'
    """
    s = str(wl).strip().upper()
    if s == "W":
        return "🟢"
    if s == "L":
        return "🔴"
    return "🏀"


def _safe_int(val, default=None):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def get_last_5_games_from_gamelog(gamelog_df: pd.DataFrame, season: str = "2025-26") -> pd.DataFrame:
    """
    Returns last 5 games (most recent) for the given season, based on GAME_DATE.
    """
    if gamelog_df is None or gamelog_df.empty:
        return pd.DataFrame()

    df = gamelog_df.copy()

    if "SEASON" not in df.columns or "GAME_DATE" not in df.columns:
        return pd.DataFrame()

    df = df[df["SEASON"].astype(str).str.strip() == season].copy()
    if df.empty:
        return pd.DataFrame()

    # GAME_DATE is like: "Oct 26, 2025"
    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    df = df.dropna(subset=["GAME_DATE_DT"]).sort_values("GAME_DATE_DT", ascending=False).head(5)

    return df


st.set_page_config(
    page_title="Deni Avdija Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"]{
        background: linear-gradient(180deg, #f7f2e8 0%, #f0e7d7 100%);
      }
      [data-testid="stVerticalBlock"]{
        background: rgba(255,255,255,0.60);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
      }
      [data-testid="stSidebar"] > div:first-child{
        background: rgba(255,255,255,0.85);
      }
      .block-container{
        padding-top: 24px;
        padding-bottom: 24px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Deni Avdija Performance Dashboard")


@st.cache_data(show_spinner=False)
def load_all_data():
    deni_shots = load_csv("deni_shot_chart.csv")
    base = load_csv("league_all_seasons_base.csv")
    adv = load_csv("league_all_seasons_adv.csv")

    lebron_shots = load_csv("lebron_shot_chart.csv")
    casspi_shots = load_csv("casspi_shot_chart.csv")

    deni_lebron_compare = load_csv("deni_lebron_year6_compare.csv")
    deni_casspi_compare = load_csv("deni_casspi_career_compare.csv")

    # NEW: game log (all seasons)
    deni_gamelog = load_csv("deni_avdija_all_seasons_gamelog.csv")

    return deni_shots, base, adv, lebron_shots, casspi_shots, deni_lebron_compare, deni_casspi_compare, deni_gamelog


deni_shots, base, adv, lebron_shots, casspi_shots, deni_lebron_compare, deni_casspi_compare, deni_gamelog = load_all_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "What would you like to see?",
    ["Home", "Career Trends", "Shot Locations", "Offense vs Defense", "Player Comparison", "Player Case Study"],
    index=0
)

if page == "Home":
    st.markdown("## Welcome 🏀")
    st.caption("Use the sidebar to navigate between views.")
    team_abbr = "POR"
    team_logo_b64 = img_to_base64(TEAM_LOGO_PATHS[team_abbr])

    # -----------------------
    # Player Profile Header
    # -----------------------
    components.html(
        f"""
        <div style="
            width: 100%;
            background: linear-gradient(90deg, #c62828, #b71c1c);
            border-radius: 18px;
            padding: 36px 40px;
            margin: 10px 0 26px 0;
            color: white;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
            box-sizing: border-box;
            position: relative;
            overflow: hidden;
        ">

            <!-- Team logo (top-right corner) -->
            <img
                src="data:image/png;base64,{team_logo_b64}"
                style="
                    position: absolute;
                    top: 18px;
                    right: 18px;
                    width: 88px;
                    height: 88px;
                    object-fit: contain;
                    opacity: 0.95;
                    filter: drop-shadow(0 6px 14px rgba(0,0,0,0.25));
                "
            />

            <div style="
                display: grid;
                grid-template-columns: 2.4fr 1fr;
                align-items: center;
                gap: 36px;
                position: relative;
                z-index: 1;
            ">

                <!-- LEFT: TEXT -->
                <div>
                    <div style="font-size:14px; opacity:0.9; font-weight:600;">
                        Portland Trail Blazers | #8 | Forward
                    </div>

                    <div style="font-size:46px; font-weight:800; line-height:1.05; margin-top:8px;">
                        DENI<br>AVDIJA
                    </div>

                    <div style="
                        display: grid;
                        grid-template-columns: repeat(6, 1fr);
                        gap: 16px;
                        margin-top: 26px;
                        font-size: 14px;
                    ">
                        <div><div style="opacity:0.85; font-weight:700;">Height</div><div>6'8&quot; (2.03m)</div></div>
                        <div><div style="opacity:0.85; font-weight:700;">Country</div><div>Israel</div></div>
                        <div><div style="opacity:0.85; font-weight:700;">Age</div><div>25</div></div>
                        <div><div style="opacity:0.85; font-weight:700;">Draft</div><div>2020 • Pick 9</div></div>
                        <div><div style="opacity:0.85; font-weight:700;">Experience</div><div>5 Seasons</div></div>
                        <div><div style="opacity:0.85; font-weight:700;">Position</div><div>Forward</div></div>
                    </div>
                </div>

                <!-- RIGHT: IMAGE (same red background) -->
                <div style="display:flex; justify-content:center; align-items:center;">
                    <img
                        src="https://cdn.nba.com/headshots/nba/latest/1040x760/{DENI_ID}.png"
                        style="
                            max-width: 240px;
                            width: 100%;
                            height: auto;
                            border-radius: 14px;
                            background: transparent;
                            padding: 0;
                            filter: drop-shadow(0 10px 24px rgba(0,0,0,0.25));
                        "
                    />
                </div>

            </div>
        </div>
        """,
        height=390,
    )

    st.markdown(
        """
        <div style="font-size:18px; line-height:1.6;">
        <strong>Deni – Last 5 Games</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='font-size:17px; line-height:1.7;'>",
        unsafe_allow_html=True
    )

    last5_df = get_last_5_games_from_gamelog(deni_gamelog, season="2025-26")

    if last5_df.empty:
        st.markdown(
            "<em>No games found for season 2025-26 in deni_avdija_all_seasons_gamelog.</em><br><br>",
            unsafe_allow_html=True
        )
    else:
        for _, row in last5_df.iterrows():
            emoji = wl_to_emoji(row.get("WL"))
            dt = row.get("GAME_DATE_DT")
            date_str = dt.strftime("%d.%m") if isinstance(dt, pd.Timestamp) else str(row.get("GAME_DATE", "")).strip()

            raw_matchup = str(row.get("MATCHUP", "")).strip()

            # Always show VS
            matchup = (
                raw_matchup
                .replace("@", "vs.")
                .replace("VS.", "vs.")
                .replace("VS", "vs.")
            )
            pts = _safe_int(row.get("PTS"), default=None)

            pts_str = f"{pts} pts" if pts is not None else "— pts"

            st.markdown(
                f"""{emoji} <strong>{date_str}</strong> — 
            <strong>{matchup}</strong>  
            Deni: <strong>{pts_str}</strong><br><br>""",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Shot Locations":
    render_shot_map(deni_shots)

elif page == "Player Comparison":
    render_player_compare(base, adv)

elif page == "Offense vs Defense":
    render_usage_vs_defense(base, adv)

elif page == "Career Trends":
    render_career_progression(base, adv)

elif page == "Player Case Study":
    render_deni_labron_casspi(
        deni_shots=deni_shots,
        lebron_shots=lebron_shots,
        casspi_shots=casspi_shots,
        deni_lebron_compare=deni_lebron_compare,
        deni_casspi_compare=deni_casspi_compare,
    )
