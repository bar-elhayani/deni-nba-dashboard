# 1. Imports & Setup
# ==============================
import os
import time
from unittest.case import TestCase
import numpy as np
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, shotchartdetail, leaguedashplayerstats, boxscoresummaryv3
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.library.parameters import SeasonAll, SeasonTypeAllStar
import nba_api.stats.library.http as nba_http


NBA_API_DEFAULT_TIMEOUT = 120  # Increase timeout from 30s to 120s for all API requests

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory at: {DATA_DIR}")
else:
    print(f"Saving to: {DATA_DIR}")

os.makedirs(DATA_DIR, exist_ok=True)
CUSTOM_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com'
}
DENI_NAME = "Deni Avdija"
LEBRON_NAME = "LeBron James"
LEBRON_SEASON = ["2008-09"]
CASSPI_NAME = "Omri Casspi"
CASSPI_SEASON = ["2015-16"]
CURRENT_SEASON = "2025-26"  # Format used by nba_api
DENI_YEAR6_SEASON = "2025-26"  # Deni's Year 6 season
LEBRON_YEAR6_SEASON = "2008-09"  # LeBron's Year 6 season
DENI_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TEST_PLAYERS = [
    "Deni Avdija",
    "LeBron James",
    "Stephen Curry",
    "Nikola Jokić",
    "Luka Dončić",
    "Giannis Antetokounmpo",
    "Kevin Durant",
    "Joel Embiid",
    "Jayson Tatum",
    "Anthony Davis",
    "Lauri Markkanen",
    "Alperen Sengun",
    "Chet Holmgren",
    "Julius Randle",
    "Zion Williamson",
    "Domantas Sabonis",
    "Jalen Williams",
    "Keegan Murray",
    "Jonathan Kuminga",
    "Jabari Smith Jr.",
    "Jerami Grant",
    "Deandre Ayton",
    "DeMar DeRozan",
    "Brandon Ingram",
    "Victor Wembanyama",
    "Cade Cunningham",
    "Anthony Edwards",
    "Shai Gilgeous-Alexander",
    "Tyrese Maxey",
    "Jalen Brunson",
    "James Harden",
    "Donovan Mitchell",
    "Jaylen Brown",
    "Josh Giddey",
    "Devin Booker"
]
KEY_POSITIONS = ["SF", "PF"]
ROLES = ["PG", "SG", "SF", "PF", "C"]




# ==============================
# 2. API Helper Functions
# ==============================

# Helper for caching
def csv_exists(filename):
    return os.path.exists(os.path.join(DATA_DIR, filename))

def set_nba_api_timeout(timeout_in_seconds):
    """
    Monkey-patch nba_api's HTTP timeout globally.
    """
    nba_http.TIMEOUT = timeout_in_seconds

set_nba_api_timeout(NBA_API_DEFAULT_TIMEOUT)


def get_player_id(player_full_name):
    """
    Returns NBA player id for a given full name.
    Handles case insensitivity and raises ValueError if not found.
    """
    plist = players.get_players()
    for p in plist:
        if p["full_name"].lower() == player_full_name.lower():
            return p["id"]
    raise ValueError(f"Player '{player_full_name}' not found in NBA API.")


import os
import time
import pandas as pd
from nba_api.stats.endpoints import playergamelog, shotchartdetail

def fetch_players_season_data(player_ids, player_names, seasons, overwrite=False):
    """
    Fetches data for players.
    For EACH PLAYER:
      1. Fetches/Loads all requested seasons.
      2. Creates a UNIFIED CSV file for that player (all seasons combined).
      3. DELETES the individual season sub-files to keep the folder clean.
    
    Finally, returns concatenated dataframes for all players.
    """
    all_players_games = []
    all_players_shots = []

    # Target output folder (can be changed if needed)

    for pid, name in zip(player_ids, player_names):
        print(f"Processing player: {name}...")

        # Temporary lists for storing only the current player's data
        current_player_games = []
        current_player_shots = []

        # List of temp files to delete for this player
        files_to_delete = []

        # Build a base filename (e.g., deni_avdija)
        sanitized_name = name.replace(' ', '_').lower()

        for season in seasons:
            # --- 1. GAMELOG ---
            gamelog_file = f"{sanitized_name}_{season}_gamelog.csv"
            gamelog_path = os.path.join(DATA_DIR, gamelog_file)
            files_to_delete.append(gamelog_path) # mark for later cleanup

            # Decide whether to re-fetch or load from disk
            if overwrite or not os.path.exists(gamelog_path):
                try:
                    gl = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
                    time.sleep(0.6)
                    df_gl = gl.get_data_frames()[0]
                    df_gl['SEASON'] = season
                    df_gl['PLAYER_NAME'] = name
                    df_gl.to_csv(gamelog_path, index=False)
                except Exception as e:
                    print(f"  Error fetching gamelog {season}: {e}")
                    df_gl = pd.DataFrame()
            else:
                df_gl = pd.read_csv(gamelog_path)

            if not df_gl.empty:
                current_player_games.append(df_gl)

            # --- 2. SHOTCHART ---
            shotchart_file = f"{sanitized_name}_{season}_shots.csv"
            shotchart_path = os.path.join(DATA_DIR, shotchart_file)
            files_to_delete.append(shotchart_path) # mark for later cleanup

            if overwrite or not os.path.exists(shotchart_path):
                try:
                    sc = shotchartdetail.ShotChartDetail(
                        team_id=0, player_id=pid, season_type_all_star="Regular Season",
                        season_nullable=season, context_measure_simple='FGA'
                    )
                    time.sleep(0.6)
                    df_sc = sc.get_data_frames()[0]
                    df_sc['SEASON'] = season
                    df_sc['PLAYER_NAME'] = name
                    df_sc.to_csv(shotchart_path, index=False)
                except Exception as e:
                    print(f"  Error fetching shots {season}: {e}")
                    df_sc = pd.DataFrame()
            else:
                df_sc = pd.read_csv(shotchart_path)

            if not df_sc.empty:
                current_player_shots.append(df_sc)

        # --- Merge + cleanup step (per player) ---

        # 1) Save unified per-player GameLog file
        if current_player_games:
            player_full_gl = pd.concat(current_player_games, ignore_index=True)
            player_gl_filename = f"{sanitized_name}_all_seasons_gamelog.csv"
            player_full_gl.to_csv(os.path.join(DATA_DIR, player_gl_filename), index=False)
            print(f"  Saved unified gamelog: {player_gl_filename}")
            all_players_games.append(player_full_gl)

        # 2) Save unified per-player Shots file
        if current_player_shots:
            player_full_sc = pd.concat(current_player_shots, ignore_index=True)
            player_sc_filename = f"{sanitized_name}_all_seasons_shots.csv"
            player_full_sc.to_csv(os.path.join(DATA_DIR, player_sc_filename), index=False)
            print(f"  Saved unified shots: {player_sc_filename}")
            all_players_shots.append(player_full_sc)

        # 3) Delete per-season temp files for this player
        print(f"  Cleaning up temp files for {name}...")
        for f in files_to_delete:
            if os.path.exists(f):
                os.remove(f)

    # Return final DataFrames aggregated across all requested players (if any)
    final_games = pd.concat(all_players_games, ignore_index=True) if all_players_games else pd.DataFrame()
    final_shots = pd.concat(all_players_shots, ignore_index=True) if all_players_shots else pd.DataFrame()
    
    return final_games, final_shots

def fetch_players_by_role_and_seasons(role, seasons, max_players=20, overwrite=False):
    """
    Finds active NBA players (uses nba_api player static list + info),
    filters by position (best-effort), then fetches their gamelogs and shotcharts.
    Returns two DataFrames for the role+seasons.
    """
    # Get candidate player list
    plist = players.get_players()
    role_players = []
    for p in plist:
        try:
            info_res = commonplayerinfo.CommonPlayerInfo(player_id=p["id"],timeout=NBA_API_DEFAULT_TIMEOUT)
            time.sleep(0.6)
            info = info_res.get_normalized_dict()["CommonPlayerInfo"][0]
            player_position = info.get("POSITION", "")
            if role in player_position:
                # Active only
                if info.get("ROSTERSTATUS", 0) == 1:
                    role_players.append((p["id"], p["full_name"]))
        except Exception as e:
            continue
        if len(role_players) >= max_players:  # To avoid API throttling, sample
            break

    if not role_players:
        print(f"No players found for role {role}")
        return pd.DataFrame(), pd.DataFrame()

    role_ids, role_names = zip(*role_players)
    return fetch_players_season_data(role_ids, role_names, seasons, overwrite=overwrite)


# ==============================
# 3. Data Fetching & Preparation (ETL)
# ==============================

def generate_player_career_stats(player_name: str, seasons: list, output_filename: str, overwrite: bool = False):
    """
    Aggregates a player's stats for all requested seasons to track personal growth and efficiency.

    Output columns:
    - SEASON, CAREER_YEAR, PLAYER_NAME
    - GP, MIN_AVG, PTS_AVG, REB_AVG, AST_AVG, STL_AVG, BLK_AVG
    - PTS_TOT, REB_TOT, AST_TOT
    - FG_PCT, FG3_PCT, FT_PCT, TS_PCT, eFG_PCT
    - AST_TO_TOV, USAGE_INDEX
    """

    # Resolve NBA player_id from full player name
    player_id = get_player_id(player_name)

    # Fetch all game logs for the given seasons (merged into one DataFrame)
    games_df, _ = fetch_players_season_data(
        [player_id],
        [player_name],
        seasons,
        overwrite=overwrite
    )

    career_stats = []

    # Iterate season-by-season and compute aggregates / efficiency metrics
    for idx, season in enumerate(seasons):
        season_df = games_df[games_df["SEASON"] == season].copy()
        if season_df.empty:
            continue

        # Basic totals
        gp = len(season_df)
        total_pts = season_df["PTS"].astype(float).sum()
        total_reb = season_df["REB"].astype(float).sum()
        total_ast = season_df["AST"].astype(float).sum()
        total_stl = season_df["STL"].astype(float).sum()
        total_blk = season_df["BLK"].astype(float).sum()
        total_tov = season_df["TOV"].astype(float).sum()
        total_min = season_df["MIN"].astype(float).sum()

        # Shooting totals for efficiency metrics
        total_fgm = season_df["FGM"].astype(float).sum()
        total_fga = season_df["FGA"].astype(float).sum()
        total_fg3m = season_df["FG3M"].astype(float).sum()
        total_fg3a = season_df["FG3A"].astype(float).sum()
        total_ftm = season_df["FTM"].astype(float).sum()
        total_fta = season_df["FTA"].astype(float).sum()

        # True Shooting % (TS%): PTS / (2 * (FGA + 0.44*FTA))
        ts_divider = 2 * (total_fga + 0.44 * total_fta)
        ts_pct = total_pts / ts_divider if ts_divider > 0 else 0

        # Effective FG % (eFG%): (FGM + 0.5*3PM) / FGA
        efg_pct = (total_fgm + 0.5 * total_fg3m) / total_fga if total_fga > 0 else 0

        # Usage proxy (lightweight approximation): (FGA + 0.44*FTA + TOV) / MIN
        usage_proxy = (total_fga + 0.44 * total_fta + total_tov) / (total_min if total_min > 0 else 1)

        # Final aggregated row for this season
        career_stats.append({
            "SEASON": season,
            "CAREER_YEAR": idx + 1,
            "PLAYER_NAME": player_name,
            "GP": gp,
            "MIN_AVG": round(season_df["MIN"].astype(float).mean(), 1),
            "PTS_AVG": round(season_df["PTS"].astype(float).mean(), 1),
            "REB_AVG": round(season_df["REB"].astype(float).mean(), 1),
            "AST_AVG": round(season_df["AST"].astype(float).mean(), 1),
            "STL_AVG": round(season_df["STL"].astype(float).mean(), 2),
            "BLK_AVG": round(season_df["BLK"].astype(float).mean(), 2),
            "PTS_TOT": int(total_pts),
            "REB_TOT": int(total_reb),
            "AST_TOT": int(total_ast),
            "FG_PCT": round(total_fgm / total_fga, 3) if total_fga > 0 else 0,
            "FG3_PCT": round(total_fg3m / total_fg3a, 3) if total_fg3a > 0 else 0,
            "FT_PCT": round(total_ftm / total_fta, 3) if total_fta > 0 else 0,
            "TS_PCT": round(ts_pct, 3),
            "eFG_PCT": round(efg_pct, 3),
            "AST_TO_TOV": round(total_ast / total_tov, 2) if total_tov > 0 else total_ast,
            "USAGE_INDEX": round(usage_proxy * 100, 1),
        })

    # Build final DataFrame and save
    df = pd.DataFrame(career_stats)
    out_path = os.path.join(DATA_DIR, output_filename)
    df.to_csv(out_path, index=False)

    print(f"✅ Career stats saved: {out_path} | seasons={len(df)} | player={player_name}")
    return df


# -------------------------
# Wrappers (keep filenames stable for dashboard dependencies)
# -------------------------

def generate_deni_career_stats(overwrite=False):
    # Keeps the exact existing filename for Deni
    return generate_player_career_stats(DENI_NAME, DENI_SEASONS, "deni_career_stats.csv", overwrite=overwrite)

def generate_lebron_career_stats(overwrite=False):
    return generate_player_career_stats(LEBRON_NAME, LEBRON_SEASON, "lebron_career_stats.csv", overwrite=overwrite)

def generate_casspi_career_stats(overwrite=False):
    return generate_player_career_stats(CASSPI_NAME, CASSPI_SEASON, "casspi_career_stats.csv", overwrite=overwrite)



def generate_player_season_rankings(
    player_name: str,
    seasons: list,
    output_filename: str,
    overwrite: bool = False,
    value_col_name: str = "PLAYER_VALUE",
    tmp_prefix: str = "league_dash_stats"
):
    """
    Generates a CSV containing the player's rankings and percentiles relative to the league
    for each requested season.

    Implementation:
    - Pulls LeagueDashPlayerStats in PerGame mode (Regular Season).
    - Filters league data to players with GP >= 50% of season max GP (keeps player included regardless).
    - Computes absolute rank and percentile for selected metrics.
    - Saves results to output_filename and cleans up temporary league files.
    """

    rows = []

    # Resolve NBA player_id from full player name
    player_id = get_player_id(player_name)

    # Track temporary files so we can delete them at the end
    files_to_clean = []

    print(f"Generating Season Rankings for {player_name}...")

    for season in seasons:
        # Temporary file used to cache league stats for this season
        fname = f"{tmp_prefix}_{season.replace('-', '')}.csv"
        league_stats_path = os.path.join(DATA_DIR, fname)
        files_to_clean.append(league_stats_path)

        # 1) Fetch or read cached league stats (PerGame)
        if overwrite or not os.path.exists(league_stats_path):
            try:
                lds = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_detailed="PerGame",  # Compare averages, not totals
                    timeout=120
                )
                time.sleep(1.0)  # Basic rate-limit protection
                df = lds.get_data_frames()[0]
                df.to_csv(league_stats_path, index=False)
            except Exception as e:
                print(f"Error fetching league stats for {season}: {e}")
                continue
        else:
            df = pd.read_csv(league_stats_path)

        # 2) Locate the player row (ID first, name fallback)
        player_row = df[df["PLAYER_ID"] == int(player_id)]
        if player_row.empty:
            player_row = df[df["PLAYER_NAME"].astype(str).str.lower() == player_name.lower()]

        if player_row.empty:
            print(f"No data found for {player_name} in season {season}. Skipping.")
            continue

        # 3) Filter dataset by games played (keep player included even if below threshold)
        max_gp_in_season = df["GP"].max()
        min_gp_threshold = max_gp_in_season * 0.50
        df_filtered = df[df["GP"] >= min_gp_threshold].copy()

        # Ensure the player is included in the filtered dataset
        try:
            player_gp = int(player_row.iloc[0]["GP"])
        except Exception:
            player_gp = 0

        if player_gp < min_gp_threshold:
            df_filtered = pd.concat([df_filtered, player_row]).drop_duplicates(subset=["PLAYER_ID"])

        # 4) Calculate rankings & percentiles
        metrics = ["PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]

        for m in metrics:
            if m not in df_filtered.columns:
                continue

            all_vals = df_filtered[m].fillna(0)
            player_val = float(player_row.iloc[0][m])

            # Absolute rank: 1 = best (higher is better)
            abs_rank = int((all_vals > player_val).sum() + 1)

            # Percentile: % of players with a lower value (higher is better)
            percentile = float((all_vals < player_val).mean() * 100)

            rows.append({
                "SEASON": season,
                "METRIC": m,
                value_col_name: round(player_val, 2),
                "ABS_RANK": abs_rank,
                "PERCENTILE": round(percentile, 1),
                "TOTAL_PLAYERS": len(df_filtered)
            })

    # 5) Save results
    final_df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, output_filename)
    final_df.to_csv(out_path, index=False)
    print(f"✅ Successfully generated '{output_filename}'.")

    # 6) Cleanup temporary files
    print("Cleaning up temporary league stats files...")
    for f in files_to_clean:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")

    return final_df


# -------------------------
# Wrappers (keep filenames stable for dashboard dependencies)
# -------------------------

def generate_deni_season_rankings(overwrite=False):
    # Keep Deni's existing filename and existing value column name
    return generate_player_season_rankings(
        player_name=DENI_NAME,
        seasons=DENI_SEASONS,
        output_filename="deni_season_rankings.csv",
        overwrite=overwrite,
        value_col_name="DENI_VALUE"
    )

def generate_lebron_season_rankings(overwrite=False):
    return generate_player_season_rankings(
        player_name=LEBRON_NAME,
        seasons=LEBRON_SEASON,
        output_filename="lebron_season_rankings.csv",
        overwrite=overwrite,
        value_col_name="PLAYER_VALUE"
    )

def generate_casspi_season_rankings(overwrite=False):
    return generate_player_season_rankings(
        player_name=CASSPI_NAME,
        seasons=CASSPI_SEASON,
        output_filename="casspi_season_rankings.csv",
        overwrite=overwrite,
        value_col_name="PLAYER_VALUE"
    )


def generate_player_current_season_comparison(
    player_name: str,
    season: str,
    output_filename: str,
    overwrite: bool = False,
    highlight_col_name: str = "IS_PLAYER"
):
    """
    Fetches season stats for all NBA players and highlights a specific player for comparison.
    Merges PerGame averages + Totals + Advanced (USG_PCT).

    Parameters:
        player_name: Full name (as in nba_api).
        season: Season string (e.g., "2025-26").
        output_filename: CSV name to save inside DATA_DIR.
        overwrite: Kept for API symmetry (not used here for caching yet).
        highlight_col_name: Boolean flag column name for visualization highlighting.

    Returns:
        DataFrame with merged metrics for all players, highlighted player on top.
    """

    # Resolve NBA player_id from full player name
    player_id = get_player_id(player_name)

    try:
        # 1) Fetch PerGame averages
        print(f"Fetching average stats (PerGame) for season={season} ...")
        lds_avg = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            timeout=120
        )
        time.sleep(1.0)
        df_avg = lds_avg.get_data_frames()[0]

        # 2) Fetch Totals
        print(f"Fetching raw totals (Totals) for season={season} ...")
        lds_tot = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals",
            timeout=120
        )
        time.sleep(1.0)
        df_tot = lds_tot.get_data_frames()[0]

        # 3) Fetch Advanced metrics (USG_PCT)
        print(f"Fetching advanced metrics (Advanced) for season={season} ...")
        lds_adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            timeout=120
        )
        time.sleep(1.0)
        df_adv = lds_adv.get_data_frames()[0]

    except Exception as e:
        print(f"Error fetching data from NBA API for season {season}: {e}")
        return pd.DataFrame()

    # Rename totals columns so we can keep both averages and totals
    tot_cols_rename = {
        "PTS": "PTS_TOT",
        "REB": "REB_TOT",
        "AST": "AST_TOT",
        "STL": "STL_TOT",
        "BLK": "BLK_TOT",
        "MIN": "MIN_TOT",
    }

    # Keep only necessary totals columns to avoid clutter
    tot_keep_cols = ["PLAYER_ID"] + [c for c in tot_cols_rename.keys() if c in df_tot.columns]
    df_tot_subset = df_tot[tot_keep_cols].rename(columns=tot_cols_rename)

    # Merge: averages + totals
    df = pd.merge(df_avg, df_tot_subset, on="PLAYER_ID", how="left")

    # Merge: add USG_PCT if available
    if "PLAYER_ID" in df_adv.columns and "USG_PCT" in df_adv.columns:
        df = pd.merge(df, df_adv[["PLAYER_ID", "USG_PCT"]], on="PLAYER_ID", how="left")

    # Final column organization (keep only existing columns)
    final_cols = [
        "PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "MIN_TOT",
        "PTS", "PTS_TOT", "REB", "REB_TOT", "AST", "AST_TOT",
        "STL", "STL_TOT", "BLK", "BLK_TOT",
        "FG_PCT", "FG3_PCT", "USG_PCT",
    ]
    df = df[[c for c in final_cols if c in df.columns]].copy()

    # Add highlight flag column for visualization
    df[highlight_col_name] = df["PLAYER_ID"] == int(player_id)

    # Move highlighted player to the top (no duplicates)
    target_id = int(player_id)
    target_row = df[df["PLAYER_ID"] == target_id]
    other_rows = df[df["PLAYER_ID"] != target_id]
    final_df = pd.concat([target_row, other_rows]).reset_index(drop=True)

    # Save to CSV
    output_path = os.path.join(DATA_DIR, output_filename)
    final_df.to_csv(output_path, index=False)

    print(f"✅ Saved {len(final_df)} players to {output_path} | highlighted={player_name}")
    return final_df


# Wrappers: keep stable filenames for the dashboard
def generate_deni_current_season_comparison(overwrite=False):
    return generate_player_current_season_comparison(
        player_name=DENI_NAME,
        season=CURRENT_SEASON,
        output_filename="deni_current_season_comparison.csv",
        overwrite=overwrite,
        highlight_col_name="IS_DENI",
    )

def generate_lebron_current_season_comparison(overwrite=False):
    return generate_player_current_season_comparison(
        player_name=LEBRON_NAME,
        season=LEBRON_SEASON[0],
        output_filename="lebron_current_season_comparison.csv",
        overwrite=overwrite,
        highlight_col_name="IS_LEBRON",
    )

def generate_casspi_current_season_comparison(overwrite=False):
    return generate_player_current_season_comparison(
        player_name=CASSPI_NAME,
        season=CASSPI_SEASON[0],
        output_filename="casspi_current_season_comparison.csv",
        overwrite=overwrite,
        highlight_col_name="IS_CASSPI",
    )


def generate_player_shot_chart(
    player_name: str,
    seasons: list,
    shot_output_filename: str,
    zone_output_filename: str,
    overwrite: bool = True
):
    """
    Generates two CSV files:
    1) Shot chart (per-shot details)
    2) Zone stats (aggregated by SEASON, PERIOD, SHOT_ZONE_BASIC)

    Parameters:
        player_name: Full name (as in nba_api).
        seasons: List of seasons (e.g., ["2025-26"]).
        shot_output_filename: CSV name for shot-level rows inside DATA_DIR.
        zone_output_filename: CSV name for zone aggregates inside DATA_DIR.
        overwrite: If True, forces re-fetch to avoid stale / missing shot data.

    Returns:
        (shot_df, zone_stats_df)
    """

    # Resolve NBA player_id from full player name
    player_id = get_player_id(player_name)

    # Fetch shot-level data across seasons
    _, all_shots_df = fetch_players_season_data(
        [player_id],
        [player_name],
        seasons,
        overwrite=overwrite
    )

    if all_shots_df.empty:
        print(f"No shot data found for {player_name}.")
        return pd.DataFrame(), pd.DataFrame()

    # 1) Process shot chart data
    cols = [
        "LOC_X", "LOC_Y",
        "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
        "SHOT_MADE_FLAG", "GAME_DATE", "SEASON", "PERIOD", "ACTION_TYPE"
    ]

    # Keep only columns that exist in the response
    valid_cols = [c for c in cols if c in all_shots_df.columns]
    shot_df = all_shots_df[valid_cols].copy()

    # Create a clear label for the result (Made vs Missed)
    if "SHOT_MADE_FLAG" in shot_df.columns:
        shot_df["SHOT_RESULT"] = shot_df["SHOT_MADE_FLAG"].map({1: "Made", 0: "Missed"})

    # Save shot chart CSV
    shot_path = os.path.join(DATA_DIR, shot_output_filename)
    shot_df.to_csv(shot_path, index=False)

    # 2) Process zone statistics
    required = {"SEASON", "PERIOD", "SHOT_ZONE_BASIC", "SHOT_MADE_FLAG"}
    if not required.issubset(set(shot_df.columns)):
        print(f"Missing required columns for zone stats for {player_name}.")
        return shot_df, pd.DataFrame()

    zone_stats = shot_df.groupby(["SEASON", "PERIOD", "SHOT_ZONE_BASIC"]).agg(
        FGA=("SHOT_MADE_FLAG", "count"),
        FGM=("SHOT_MADE_FLAG", "sum"),
    ).reset_index()

    # Calculate FG% with division-by-zero safety
    zone_stats["FG_PCT"] = np.where(
        zone_stats["FGA"] > 0,
        zone_stats["FGM"] / zone_stats["FGA"],
        0
    )

    # Save zone stats CSV
    zone_path = os.path.join(DATA_DIR, zone_output_filename)
    zone_stats.to_csv(zone_path, index=False)

    print(f"✅ Shot chart saved: {shot_path}")
    print(f"✅ Zone stats saved: {zone_path}")
    return shot_df, zone_stats


# Wrappers: keep stable filenames for the dashboard
def generate_deni_shot_chart(overwrite=True):
    return generate_player_shot_chart(
        player_name=DENI_NAME,
        seasons=DENI_SEASONS,
        shot_output_filename="deni_shot_chart.csv",
        zone_output_filename="deni_zone_stats.csv",
        overwrite=overwrite
    )

def generate_lebron_shot_chart(overwrite=True):
    return generate_player_shot_chart(
        player_name=LEBRON_NAME,
        seasons=LEBRON_SEASON,
        shot_output_filename="lebron_shot_chart.csv",
        zone_output_filename="lebron_zone_stats.csv",
        overwrite=overwrite
    )

def generate_casspi_shot_chart(overwrite=True):
    return generate_player_shot_chart(
        player_name=CASSPI_NAME,
        seasons=CASSPI_SEASON,
        shot_output_filename="casspi_shot_chart.csv",
        zone_output_filename="casspi_zone_stats.csv",
        overwrite=overwrite
    )



def generate_off_def_scatter_data(season=None, overwrite=False):     # V
    """
    Prepares scatter dataset with OFF_RATING and DEF_RATING for selected players.

    Metrics Explanation:
    --------------------
    OFF_RATING (Offensive Rating):
        - Points scored per 100 possessions while player is on court
        - Scale: ~100-125, HIGHER = BETTER
        - Elite: 118+, Good: 112-118, Average: 108-112

    DEF_RATING (Defensive Rating):
        - Points allowed per 100 possessions while player is on court
        - Scale: ~100-120, LOWER = BETTER
        - Elite: <108, Good: 108-112, Average: 112-115

    NET_RATING:
        - OFF_RATING minus DEF_RATING
        - Positive = team better with player, HIGHER = BETTER

    PIE (Player Impact Estimate):
        - Overall contribution percentage (points, rebounds, assists, etc.)
        - Scale: 0-1, HIGHER = BETTER
        - Elite: 0.15+, All-Star: 0.12-0.15, Starter: 0.10-0.12
    """
    if season is None:
        season = DENI_YEAR6_SEASON

    output_path = os.path.join(DATA_DIR, "off_def_scatter_data.csv")

    if not overwrite and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if not existing.empty and 'OFF_RATING' in existing.columns:
            if existing['OFF_RATING'].notna().any():
                print("Using existing off_def_scatter_data.csv")
                return existing

    print(f"Fetching advanced stats for season {season}...")

    try:
        time.sleep(1)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            headers=CUSTOM_HEADERS,
            timeout=120
        )

        response = stats.nba_response.get_dict()
        headers = response['resultSets'][0]['headers']
        rows = response['resultSets'][0]['rowSet']
        df = pd.DataFrame(rows, columns=headers)

        print(f"Got {len(df)} players from API")

    except Exception as e:
        print(f"Error fetching stats: {e}")
        return pd.DataFrame()

    # Filter down to the players we care about
    target_players = [CASSPI_NAME] + TEST_PLAYERS
    df_filtered = df[df["PLAYER_NAME"].isin(target_players)].copy()
    print(f"Filtered to {len(df_filtered)} players")

    # Select output columns
    cols = ["PLAYER_ID", "PLAYER_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "PIE"]
    result = df_filtered[cols].copy()

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} players to {output_path}")

    return result




def generate_multi_metric_comparison(overwrite=False):
    """
    Generates data in 3 steps:
    1. Saves a MASTER Base file (All seasons, All players).
    2. Saves a MASTER Advanced file (All seasons, All players).
    3. Generates the specific 'multi_metric_comparison.csv' for the project.
    4. Cleans up individual season files.
    """
    print(f"Generating Multi-Metric Comparison Data...")

    # Collect data from all seasons
    all_seasons_base = []
    all_seasons_adv = []

    # Track temp files for cleanup
    files_to_clean = []

    # Columns we need for the final comparison file
    base_cols_for_metric = [
        "PTS", "REB", "AST", "STL", "BLK", "TOV", 
        "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS", "GP", "MIN"
    ]
    adv_cols_for_metric = [
        "OFF_RATING", "DEF_RATING", "USG_PCT", "TS_PCT", "PIE"
    ]

    for season in DENI_SEASONS:
        # Temp file paths for this season
        base_file = os.path.join(DATA_DIR, f"league_base_{season.replace('-', '')}.csv")
        adv_file = os.path.join(DATA_DIR, f"league_adv_{season.replace('-', '')}.csv")

        # Add to cleanup list
        files_to_clean.extend([base_file, adv_file])
        
        # --- 1. Fetch BASE Stats ---
        if overwrite or not os.path.exists(base_file):
            try:
                base_api = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense="Base", 
                    timeout=120
                )
                time.sleep(1.0)
                df_base = base_api.get_data_frames()[0]
                df_base.to_csv(base_file, index=False)
            except Exception as e:
                print(f"Error fetching Base stats for {season}: {e}")
                df_base = pd.DataFrame()
        else:
            df_base = pd.read_csv(base_file)

        # אם יש נתונים, נוסיף עמודת עונה ונשמור לרשימה הכללית
        if not df_base.empty:
            df_base['SEASON'] = season
            all_seasons_base.append(df_base)

        # --- 2. Fetch ADVANCED Stats ---
        if overwrite or not os.path.exists(adv_file):
            try:
                adv_api = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    season_type_all_star="Regular Season",
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense="Advanced",
                    timeout=120
                )
                time.sleep(1.0)
                df_adv = adv_api.get_data_frames()[0]
                df_adv.to_csv(adv_file, index=False)
            except Exception as e:
                print(f"Error fetching Advanced stats for {season}: {e}")
                df_adv = pd.DataFrame()
        else:
            df_adv = pd.read_csv(adv_file)

        # אם יש נתונים, נוסיף עמודת עונה ונשמור לרשימה הכללית
        if not df_adv.empty:
            df_adv['SEASON'] = season
            all_seasons_adv.append(df_adv)

    # --- Save consolidated master files (separately) ---

    # 1) Save consolidated Base
    if all_seasons_base:
        master_base_df = pd.concat(all_seasons_base, ignore_index=True)
        master_base_path = os.path.join(DATA_DIR, "league_all_seasons_base.csv")
        master_base_df.to_csv(master_base_path, index=False)
        print(f"Saved Master Base File: {master_base_path}")
    else:
        master_base_df = pd.DataFrame()

    # 2) Save consolidated Advanced
    if all_seasons_adv:
        master_adv_df = pd.concat(all_seasons_adv, ignore_index=True)
        master_adv_path = os.path.join(DATA_DIR, "league_all_seasons_adv.csv")
        master_adv_df.to_csv(master_adv_path, index=False)
        print(f"Saved Master Advanced File: {master_adv_path}")
    else:
        master_adv_df = pd.DataFrame()

    # --- Build the final comparison file (multi_metric_comparison) ---
    # We merge in-memory so we can filter by PIE and use metrics from both datasets

    if master_base_df.empty or master_adv_df.empty:
        return pd.DataFrame()

    # Prepare merge: drop duplicate columns from Advanced before merging (keep keys + unique metrics)
    cols_to_use_from_adv = ["PLAYER_ID", "SEASON"] + [c for c in adv_cols_for_metric if c in master_adv_df.columns]
    adv_subset = master_adv_df[cols_to_use_from_adv].copy()

    # In-memory merge
    full_merged_df = pd.merge(master_base_df, adv_subset, on=["PLAYER_ID", "SEASON"], how="inner")

    # Player filtering (same logic as before)
    target_ids = set()
    
    # A. Deni
    try:
        deni_id = get_player_id(CASSPI_NAME)
        if deni_id: target_ids.add(deni_id)
    except: pass
    
    # B. Test Players
    player_list_to_scan = TEST_PLAYERS if 'TEST_PLAYERS' in globals() else []
    for name in player_list_to_scan:
        try:
            pid = get_player_id(name)
            if pid: target_ids.add(pid)
        except: continue

    # C. Top 10 PIE per season
    for season, grp in full_merged_df.groupby("SEASON"):
        if "PIE" in grp.columns:
            top_ids = grp.sort_values("PIE", ascending=False).head(10)["PLAYER_ID"].tolist()
            target_ids.update(top_ids)

    # Final filter
    filtered_df = full_merged_df[full_merged_df["PLAYER_ID"].isin(target_ids)].copy()

    # Convert to long format
    all_records = []
    for _, row in filtered_df.iterrows():
        p_id = row["PLAYER_ID"]
        p_name = row["PLAYER_NAME"]
        seas = row["SEASON"]
        
        available_metrics = [m for m in base_cols_for_metric + adv_cols_for_metric if m in row]
        
        for metric in available_metrics:
            all_records.append({
                "PLAYER_ID": p_id,
                "PLAYER_NAME": p_name,
                "SEASON": seas,
                "METRIC_NAME": metric,
                "METRIC_VALUE": row[metric]
            })

    final_df = pd.DataFrame(all_records)
    output_path = os.path.join(DATA_DIR, "multi_metric_comparison.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Successfully generated 'multi_metric_comparison.csv' with {len(final_df)} rows.")

    # --- Cleanup temp files ---
    print("Cleaning up temporary season files...")
    for f in files_to_clean:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")

    return final_df


def generate_deni_lebron_year6_compare(overwrite=False):
    """
    Detailed comparison between Deni Avdija (Year 6) and LeBron James (Year 6).
    Adjusts for different NBA eras using team share and impact metrics.

    Columns Documentation:
    - PLAYER_NAME: Name of the player.
    - SEASON: The specific 'Year 6' season for each player.
    - GP: Games played in that season.
    - MIN: Average minutes per game.
    - PTS / REB / AST: Average points, rebounds, and assists per game.
    - STL / BLK: Average steals and blocks per game.
    - TS_PCT: True Shooting percentage (Overall shooting efficiency).
    - USG_PCT: Usage percentage (How much the player controls the offense).
    - PTS_RANK / AST_RANK / USG_RANK: League-wide rankings for that season.
    - TEAM_W_PCT: Winning percentage of the player's team.
    - PCT_PTS: The percentage of the team's total points scored by the player while on the floor.
    - PCT_REB: The percentage of the team's total rebounds grabbed by the player while on the floor.
    - PCT_AST: The percentage of the team's total assists created by the player while on the floor.
    - OFF_RATING: Team's offensive efficiency (points per 100 possessions) with the player on court.
    - DEF_RATING: Team's defensive efficiency (points allowed per 100 possessions) with the player on court.
    - PIE (Player Impact Estimate): NBA's metric for overall statistical contribution to the game.
    """
    deni_id = get_player_id(DENI_NAME)
    lebron_id = get_player_id(LEBRON_NAME)

    players_data = []

    for pid, name, season in [(deni_id, DENI_NAME, DENI_YEAR6_SEASON),
                              (lebron_id, LEBRON_NAME, LEBRON_YEAR6_SEASON)]:

        try:
            # 1. Fetch Base League Stats (Averages & Rankings)
            lds = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season", per_mode_detailed='PerGame'
            )
            time.sleep(1.0)
            league_df = lds.get_data_frames()[0]

            # 2. Fetch Advanced League Stats (Usage, PIE, Ratings)
            lds_adv = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season", measure_type_detailed_defense='Advanced'
            )
            time.sleep(1.0)
            adv_df = lds_adv.get_data_frames()[0]

            # 3. Fetch Usage Stats (PCT_PTS, PCT_AST, PCT_REB)
            lds_usage = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season", measure_type_detailed_defense='Usage'
            )
            time.sleep(1.0)
            usage_df = lds_usage.get_data_frames()[0]

        except Exception as e:
            print(f"Error fetching stats for {name} in {season}: {e}")
            continue

        # Extract specific player rows from each dataset
        player_row = league_df[league_df['PLAYER_ID'] == int(pid)].iloc[0]
        adv_row = adv_df[adv_df['PLAYER_ID'] == int(pid)].iloc[0]
        usage_row = usage_df[usage_df['PLAYER_ID'] == int(pid)].iloc[0]

        # Calculate Rankings (How many players were better in that era)
        pts_rank = (league_df['PTS'] > player_row['PTS']).sum() + 1
        ast_rank = (league_df['AST'] > player_row['AST']).sum() + 1
        usg_rank = (adv_df['USG_PCT'] > adv_row['USG_PCT']).sum() + 1

        # 4. Compile Data (Keeping all previous columns + adding new ones)
        players_data.append({
            "PLAYER_NAME": name,
            "SEASON": season,
            "GP": int(player_row['GP']),
            "MIN": round(player_row['MIN'], 1),
            "PTS": round(player_row['PTS'], 1),
            "REB": round(player_row['REB'], 1),
            "AST": round(player_row['AST'], 1),
            "STL": round(player_row['STL'], 2),
            "BLK": round(player_row['BLK'], 2),
            "FG_PCT": round(player_row['FG_PCT'], 3),
            "TS_PCT": round(player_row['PTS'] / (2 * (player_row['FGA'] + 0.44 * player_row['FTA'])), 3) if (player_row['FGA'] + 0.44 * player_row['FTA']) > 0 else 0,
            "USG_PCT": round(adv_row['USG_PCT'] * 100, 1),
            "PTS_RANK": int(pts_rank),
            "AST_RANK": int(ast_rank),
            "USG_RANK": int(usg_rank),
            "TEAM_W_PCT": round(player_row['W_PCT'], 3),
            # New Metrics: Team Share (Era Adjustment)
            "PCT_PTS": round(usage_row['PCT_PTS'] * 100, 1),
            "PCT_REB": round(usage_row['PCT_REB'] * 100, 1),
            "PCT_AST": round(usage_row['PCT_AST'] * 100, 1),
            # New Metrics: Impact
            "OFF_RATING": adv_row['OFF_RATING'],
            "DEF_RATING": adv_row['DEF_RATING'],
            "PIE": round(adv_row['PIE'] * 100, 1)
        })

    df = pd.DataFrame(players_data)

    # Save to CSV
    output_path = os.path.join(DATA_DIR, "deni_lebron_year6_compare.csv")
    df.to_csv(output_path, index=False)

    print(f"Successfully generated comparison for Year 6 (Deni vs LeBron) with {len(df.columns)} columns.")
    return df


def generate_deni_casspi_career_compare(overwrite=False):
    """
    Comparison between Deni Avdija and Omri Casspi, covering both Career Averages
    and specific Peak Seasons (Deni 2025-26 vs Casspi 2015-16).

    Columns Documentation:
    - PLAYER_NAME: Name of the player.
    - DATA_TYPE: Indicates if the row is 'Career' average or 'Peak Season' stats.
    - SEASON_SPAN: The timeframe for the data (e.g., '2009-2019' or '2015-16').
    - GP: Games Played.
    - PTS / REB / AST / STL / BLK: Average stats per game.
    - PTS_TOT / REB_TOT / AST_TOT / STL_TOT / BLK_TOT: Raw cumulative totals.
    - TS_PCT: True Shooting percentage (Efficiency).
    - USG_PCT: Usage percentage (Offensive load).
    - PCT_PTS / PCT_REB / PCT_AST: Percentage of team's stats contributed by player.
    - OFF_RATING / DEF_RATING: Team efficiency with player on court.
    - PIE: NBA's Player Impact Estimate.
    """
    from nba_api.stats.endpoints import playercareerstats

    deni_name = "Deni Avdija"
    casspi_name = "Omri Casspi"

    deni_id = get_player_id(deni_name)
    casspi_id = get_player_id(casspi_name)

    # Configuration for calculation
    configs = [
        {
            "id": deni_id, "name": deni_name, "span": "2020-Present",
            "peak": "2025-26", "all_seasons": ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
        },
        {
            "id": casspi_id, "name": casspi_name, "span": "2009-2019",
            "peak": "2015-16", "all_seasons": ["2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19"]
        }
    ]

    final_results = []

    for player in configs:
        pid = player["id"]
        name = player["name"]

        # --- PART 1: CAREER DATA ---
        print(f"Calculating career stats for {name}...")
        career = playercareerstats.PlayerCareerStats(player_id=pid)
        time.sleep(0.8)
        career_df = career.get_data_frames()[0]
        career_totals = career_df.sum()
        gp_career = int(career_totals['GP'])

        # Career Advanced Aggregation
        career_adv_list = []
        for season in player["all_seasons"]:
            try:
                # Advanced
                adv_lds = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_detailed_defense='Advanced')
                time.sleep(0.7)
                df_adv = adv_lds.get_data_frames()[0]
                p_adv = df_adv[df_adv['PLAYER_ID'] == int(pid)].iloc[0]
                # Usage
                use_lds = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_detailed_defense='Usage')
                time.sleep(0.7)
                df_use = use_lds.get_data_frames()[0]
                p_use = df_use[df_use['PLAYER_ID'] == int(pid)].iloc[0]

                career_adv_list.append({
                    'PIE': p_adv['PIE'], 'USG_PCT': p_adv['USG_PCT'],
                    'OFF_RATING': p_adv['OFF_RATING'], 'DEF_RATING': p_adv['DEF_RATING'],
                    'PCT_PTS': p_use['PCT_PTS'], 'PCT_REB': p_use['PCT_REB'], 'PCT_AST': p_use['PCT_AST']
                })
            except: continue

        career_adv_avg = pd.DataFrame(career_adv_list).mean()
        ts_career = career_totals['PTS'] / (2 * (career_totals['FGA'] + 0.44 * career_totals['FTA']))

        final_results.append({
            "PLAYER_NAME": name, "DATA_TYPE": "Career", "SEASON_SPAN": player["span"],
            "GP": gp_career,
            "PTS": round(career_totals['PTS'] / gp_career, 1), "PTS_TOT": int(career_totals['PTS']),
            "REB": round(career_totals['REB'] / gp_career, 1), "REB_TOT": int(career_totals['REB']),
            "AST": round(career_totals['AST'] / gp_career, 1), "AST_TOT": int(career_totals['AST']),
            "STL": round(career_totals['STL'] / gp_career, 2), "STL_TOT": int(career_totals['STL']),
            "BLK": round(career_totals['BLK'] / gp_career, 2), "BLK_TOT": int(career_totals['BLK']),
            "TS_PCT": round(ts_career, 3), "USG_PCT": round(career_adv_avg['USG_PCT'] * 100, 1),
            "PCT_PTS": round(career_adv_avg['PCT_PTS'] * 100, 1), "PCT_REB": round(career_adv_avg['PCT_REB'] * 100, 1),
            "PCT_AST": round(career_adv_avg['PCT_AST'] * 100, 1),
            "OFF_RATING": round(career_adv_avg['OFF_RATING'], 1), "DEF_RATING": round(career_adv_avg['DEF_RATING'], 1),
            "PIE": round(career_adv_avg['PIE'] * 100, 1)
        })

        # --- PART 2: PEAK SEASON DATA ---
        print(f"Extracting peak season ({player['peak']}) for {name}...")
        try:
            # Peak Standard Stats
            peak_std = leaguedashplayerstats.LeagueDashPlayerStats(season=player["peak"], per_mode_detailed='Totals')
            time.sleep(0.8)
            p_std = peak_std.get_data_frames()[0][peak_std.get_data_frames()[0]['PLAYER_ID'] == int(pid)].iloc[0]

            # Peak Advanced/Usage
            peak_adv_lds = leaguedashplayerstats.LeagueDashPlayerStats(season=player["peak"], measure_type_detailed_defense='Advanced')
            time.sleep(0.8)
            p_adv_peak = peak_adv_lds.get_data_frames()[0][peak_adv_lds.get_data_frames()[0]['PLAYER_ID'] == int(pid)].iloc[0]

            peak_use_lds = leaguedashplayerstats.LeagueDashPlayerStats(season=player["peak"], measure_type_detailed_defense='Usage')
            time.sleep(0.8)
            p_use_peak = peak_use_lds.get_data_frames()[0][peak_use_lds.get_data_frames()[0]['PLAYER_ID'] == int(pid)].iloc[0]

            gp_peak = int(p_std['GP'])
            final_results.append({
                "PLAYER_NAME": name, "DATA_TYPE": "Peak Season", "SEASON_SPAN": player["peak"],
                "GP": gp_peak,
                "PTS": round(p_std['PTS'] / gp_peak, 1), "PTS_TOT": int(p_std['PTS']),
                "REB": round(p_std['REB'] / gp_peak, 1), "REB_TOT": int(p_std['REB']),
                "AST": round(p_std['AST'] / gp_peak, 1), "AST_TOT": int(p_std['AST']),
                "STL": round(p_std['STL'] / gp_peak, 2), "STL_TOT": int(p_std['STL']),
                "BLK": round(p_std['BLK'] / gp_peak, 2), "BLK_TOT": int(p_std['BLK']),
                "TS_PCT": round(p_std['PTS'] / (2 * (p_std['FGA'] + 0.44 * p_std['FTA'])), 3),
                "USG_PCT": round(p_adv_peak['USG_PCT'] * 100, 1),
                "PCT_PTS": round(p_use_peak['PCT_PTS'] * 100, 1), "PCT_REB": round(p_use_peak['PCT_REB'] * 100, 1),
                "PCT_AST": round(p_use_peak['PCT_AST'] * 100, 1),
                "OFF_RATING": round(p_adv_peak['OFF_RATING'], 1), "DEF_RATING": round(p_adv_peak['DEF_RATING'], 1),
                "PIE": round(p_adv_peak['PIE'] * 100, 1)
            })
        except Exception as e:
            print(f"Could not fetch peak season for {name}: {e}")

    df = pd.DataFrame(final_results)
    output_path = os.path.join(DATA_DIR, "deni_casspi_career_compare.csv")
    df.to_csv(output_path, index=False)
    print(f"Success! Final comparison saved to {output_path}")
    return df

# ==============================
# 4. Main Execution Block
# ==============================
if __name__ == "__main__":
    overwrite = True  # Change to True to refresh all data
    import os
    import requests

    def download_image(url: str, save_path: str):
        """
        Downloads an image from the given URL and saves it to save_path.

        This uses a simple HTTP GET request and writes the image bytes to disk.
        """
        try:
            # Optional headers if needed (some servers block requests without User-Agent)
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; image-downloader/1.0)"
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise error for status codes != 200

            with open(save_path, "wb") as f:
                f.write(response.content)

            print(f"✔ Downloaded image to {save_path}")

        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")


    # Local folder to store images
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # List of (URL, output filename)
    image_list = [
        ("https://sport1.maariv.co.il/app/uploads/2020/04/TB.png", "trailblazers_logo.png"),
        ("https://upload.wikimedia.org/wikipedia/he/8/82/Wizards_clipped_rev_1.png", "wizards_logo.png"),
    ]

    for url, filename in image_list:
        save_path = os.path.join(IMAGES_DIR, filename)
        download_image(url, save_path)
    print("1. Career Stats (Deni / LeBron / Casspi) ...")

    # Run the same pipeline step for each player, while keeping stable output filenames
    for label, fn, out_file in [
        ("Deni",   generate_deni_career_stats,   "deni_career_stats.csv"),
        ("LeBron", generate_lebron_career_stats, "lebron_career_stats.csv"),
        ("Casspi", generate_casspi_career_stats, "casspi_career_stats.csv"),
    ]:
        print(f"   - {label} Career Stats ...")
        fn(overwrite)
        print(f"Saved {out_file}")


    print("2. Season Rankings (Deni / LeBron / Casspi) ...")

    # Run the same season-rankings pipeline for each player with stable output filenames
    for label, fn, out_file in [
        ("Deni",   generate_deni_season_rankings,   "deni_season_rankings.csv"),
        ("LeBron", generate_lebron_season_rankings, "lebron_season_rankings.csv"),
        ("Casspi", generate_casspi_season_rankings, "casspi_season_rankings.csv"),
    ]:
        print(f"   - {label} Season Rankings ...")
        fn(overwrite)
        print(f"Saved {out_file}")

    print("3. Current Season Comparison (Deni / LeBron / Casspi) ...")

    # Run the same comparison step for each player with stable output filenames
    for label, fn, out_file in [
        ("Deni",   generate_deni_current_season_comparison,   "deni_current_season_comparison.csv"),
        ("LeBron", generate_lebron_current_season_comparison, "lebron_current_season_comparison.csv"),
        ("Casspi", generate_casspi_current_season_comparison, "casspi_current_season_comparison.csv"),
    ]:
        print(f"   - {label} Current Season Comparison ...")
        fn(overwrite)
        print(f"Saved {out_file}")


    print("4. Shot Chart + Zone Stats (Deni / LeBron / Casspi) ...")

    # Run shot chart generation for each player with stable output filenames
    for label, fn, shot_file, zone_file in [
        ("Deni",   generate_deni_shot_chart,   "deni_shot_chart.csv",   "deni_zone_stats.csv"),
        ("LeBron", generate_lebron_shot_chart, "lebron_shot_chart.csv", "lebron_zone_stats.csv"),
        ("Casspi", generate_casspi_shot_chart, "casspi_shot_chart.csv", "casspi_zone_stats.csv"),
    ]:
        print(f"   - {label} Shot Chart + Zone Stats ...")
        fn(overwrite=True)
        print(f"Saved {shot_file} and {zone_file}")


    print("5. Off/Def Scatter Data ...")
    df5 = generate_off_def_scatter_data(season=DENI_YEAR6_SEASON, overwrite=True)
    print("Saved off_def_scatter_data.csv")

    print("6. Multi-metric Time Series...")
    df6 = generate_multi_metric_comparison(overwrite)
    print("Saved multi_metric_comparison.csv")

    print("7. Deni vs LeBron Year 6 ...")
    df7 = generate_deni_lebron_year6_compare(overwrite)
    print("Saved deni_lebron_year6_compare.csv")

    print("8. Deni vs Casspi ...")
    comparison_df = generate_deni_casspi_career_compare(overwrite=True)
    print("deni_casspi_career_compare.csv")

    print("All pipeline steps complete.")

    # ------------------------------
    # Load outputs for EDA printing
    # ------------------------------
    df1 = pd.read_csv(os.path.join(DATA_DIR, "deni_career_stats.csv"))
    df2 = pd.read_csv(os.path.join(DATA_DIR, "deni_season_rankings.csv"))
    df3 = pd.read_csv(os.path.join(DATA_DIR, "deni_current_season_comparison.csv"))
    df4 = pd.read_csv(os.path.join(DATA_DIR, "deni_shot_chart.csv"))
    df4b = pd.read_csv(os.path.join(DATA_DIR, "deni_zone_stats.csv"))

    # (optional) keep your existing df5/df6/df7, or load too:
    # df5 = pd.read_csv(os.path.join(DATA_DIR, "off_def_scatter_data.csv"))
    # df6 = pd.read_csv(os.path.join(DATA_DIR, "multi_metric_comparison.csv"))
    # df7 = pd.read_csv(os.path.join(DATA_DIR, "deni_lebron_year6_compare.csv"))

    # ==============================
    # 5. EDA: Print basic DataFrame heads
    # ==============================
    print("\n---[ EDA Head Samples ]---")
    print("Deni Career Stats:\n", df1.head(), "\n")
    print("Deni Season Rankings:\n", df2.head(), "\n")
    print("Deni Current Season Comparison:\n", df3.head(), "\n")
    print("Deni Shotchart first rows:\n", df4.head(), "\n")
    print("Deni Zone Stats first rows:\n", df4b.head(), "\n")
    print("Off/Def Scatter Data head:\n", df5.head(), "\n")
    print("Multi-Metric Comparison head:\n", df6.head(), "\n")
    print("Deni vs LeBron Year 6:\n", df7.head(), "\n")


