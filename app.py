import streamlit as st
import os
import base64
from data_loader import load_csv
from components.shot_map import render_shot_map
from components.player_compare import render_player_compare
from components.usage_vs_defense import render_usage_vs_defense
from components.career_progression import render_career_progression
from components.deni_labron_casspi import render_deni_labron_casspi
import streamlit.components.v1 as components

DENI_ID = 1630166
# -----------------------------
# Manual last 5 games (update by editing this list)
# -----------------------------
DENI_LAST_5_GAMES = [
    {
        "date": "10.01",
        "home": "Portland Trail Blazers",
        "away": "Houston Rockets",
        "home_score": 111,
        "away_score": 105,
        "deni_pts": 20,
    },
    {
        "date": "08.01",
        "home": "Portland Trail Blazers",
        "away": "Houston Rockets",
        "home_score": 103,
        "away_score": 102,
        "deni_pts": 41,
    },
    {
        "date": "06.01",
        "home": "Portland Trail Blazers",
        "away": "Utah Jazz",
        "home_score": 137,
        "away_score": 117,
        "deni_pts": 33,
    },
    {
        "date": "04.01",
        "home": "San Antonio Spurs",
        "away": "Portland Trail Blazers",
        "home_score": 110,
        "away_score": 115,
        "deni_pts": 29,
    },
    {
        "date": "03.01",
        "home": "New Orleans Pelicans",
        "away": "Portland Trail Blazers",
        "home_score": 109,
        "away_score": 122,
        "deni_pts": 34,
    },
]

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
TEAM_LOGO_PATHS = {
    "POR": os.path.join("data", "images", "trailblazers_logo.png"),
}

def portland_result_emoji(home: str, away: str, home_score: int, away_score: int) -> str:
    """
    Return 🟢 if Portland won, 🔴 if Portland lost, 🏀 if not determinable.
    """
    portland = "Portland Trail Blazers"
    if home == portland:
        return "🟢" if home_score > away_score else "🔴"
    if away == portland:
        return "🟢" if away_score > home_score else "🔴"
    return "🏀"


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

    return deni_shots, base, adv, lebron_shots, casspi_shots, deni_lebron_compare, deni_casspi_compare


deni_shots, base, adv, lebron_shots, casspi_shots, deni_lebron_compare, deni_casspi_compare = load_all_data()

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
                            background: transparent;   /* IMPORTANT: no different tint */
                            padding: 0;               /* IMPORTANT: no extra box */
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

    for g in DENI_LAST_5_GAMES:
        emoji = portland_result_emoji(
            g["home"], g["away"], g["home_score"], g["away_score"]
        )
        st.markdown(
            f"""{emoji} <strong>{g['date']}</strong> — 
    <strong>{g['home']} {g['home_score']} – {g['away_score']} {g['away']}</strong>  
    Deni: <strong>{g['deni_pts']} pts</strong><br><br>""",
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
