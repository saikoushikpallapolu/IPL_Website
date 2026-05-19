import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz

def build_auction_feature_matrix():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    master_df = pd.read_csv(processed_dir / "players_master.csv")
    name_to_id = {str(k).strip(): v for k, v in master_df.set_index('cricsheet_name')['canonical_id'].to_dict().items()}
    auc_name_to_id = {str(k).strip(): v for k, v in master_df.set_index('auction_name')['canonical_id'].to_dict().items()}
    all_valid_auction_names = [str(n).strip() for n in master_df['auction_name'].unique()]

    hist_auc = pd.read_csv(raw_dir / "ipl_auctions/historical_auctions.csv.csv", low_memory=False)
    hist_auc.columns = [c.lower().strip() for c in hist_auc.columns]
    
    hist_auc_renamed = hist_auc.rename(columns={
        'player': 'player_name', 
        'amount': 'price_in_crores', 
        'year': 'auction_year',
        'player origin': 'nationality'
    })
    
    auc_2025 = pd.read_csv(processed_dir / "standardized_2025_auction.csv")
    auc_2025.columns = [c.lower().strip() for c in auc_2025.columns]

    combined_auctions = pd.concat([hist_auc_renamed, auc_2025], ignore_index=True)

    def get_league_weight(match_type, league_name):
        m_type = str(match_type).lower().strip()
        l_name = str(league_name).lower().strip()
        if m_type in ['t20i', 'test', 'odi'] or 'ipl' in l_name:
            return 1.00
        elif any(x in l_name for x in ['bbl', 'big bash', 'psl', 'sa20']):
            return 0.85
        elif any(x in l_name for x in ['cpl', 'hundred', 'bpl']):
            return 0.70
        return 0.40

    print("Extracting full multi-format player histories from master file...")
    global_csv = processed_dir / "all_matches.csv"
    valid_wickets = ['bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    
    player_history = {}

    for chunk in pd.read_csv(global_csv, usecols=lambda c: any(x in c for x in ['match_id', 'start_date', 'match_type', 'season', 'striker', 'batter', 'bowler', 'runs_off_bat', 'wides', 'noballs', 'wicket_type']), chunksize=250000, low_memory=False):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        chunk['year'] = pd.to_datetime(chunk['start_date'], errors='coerce').dt.year
        chunk = chunk.dropna(subset=['year', 'match_id']).copy()
        chunk['year'] = chunk['year'].astype(int)
        
        bat_col = 'striker' if 'striker' in chunk.columns else 'batter'
        chunk['b_id'] = chunk[bat_col].astype(str).str.strip().map(name_to_id)
        chunk['w_id'] = chunk['bowler'].astype(str).str.strip().map(name_to_id)
        
        chunk['is_ball_faced'] = np.where(chunk['wides'].isna() | (chunk['wides'] == 0), 1, 0)
        chunk['conceded_runs'] = chunk['runs_off_bat'] + chunk['noballs'].fillna(0) + chunk['wides'].fillna(0)
        chunk['is_legal_ball'] = np.where((chunk['wides'].isna() | (chunk['wides'] == 0)) & (chunk['noballs'].isna() | (chunk['noballs'] == 0)), 1, 0)
        chunk['is_wicket'] = np.where(chunk['wicket_type'].isin(valid_wickets), 1, 0)

        for _, row in chunk.iterrows():
            m_id = row['match_id']
            year = int(row['year'])
            m_type = str(row.get('match_type', 'T20')).lower().strip()
            l_name = str(row.get('season', ''))
            w_coeff = get_league_weight(m_type, l_name)
            
            bid = row['b_id']
            if not pd.isna(bid):
                bid = int(bid)
                if bid not in player_history: player_history[bid] = []
                player_history[bid].append({
                    'type': 'bat', 'year': year, 'match_id': m_id, 'format': m_type,
                    'runs': row['runs_off_bat'], 'balls': row['is_ball_faced'], 'weight': w_coeff
                })
                
            wid = row['w_id']
            if not pd.isna(wid):
                wid = int(wid)
                if wid not in player_history: player_history[wid] = []
                player_history[wid].append({
                    'type': 'bowl', 'year': year, 'match_id': m_id, 'format': m_type,
                    'runs_conceded': row['conceded_runs'], 'balls_bowled': row['is_legal_ball'],
                    'wickets': row['is_wicket'], 'weight': w_coeff
                })

    feature_rows = []
    print("\nCalculating multi-window capabilities and injury safeguards...")

    for idx, row in combined_auctions.iterrows():
        auc_name = str(row['player_name']).strip()
        auc_year = int(float(row['auction_year'])) if not pd.isna(row['auction_year']) else 2025
        sold_price = pd.to_numeric(row['price_in_crores'], errors='coerce')
        
        if pd.isna(sold_price) or sold_price <= 0: continue
        if sold_price > 100000: sold_price = sold_price / 10000000.0

        player_id = auc_name_to_id.get(auc_name)
        if player_id is None or pd.isna(player_id):
            match = process.extractOne(auc_name, all_valid_auction_names, scorer=fuzz.token_sort_ratio)
            if match and match[1] > 70: player_id = auc_name_to_id.get(match[0])

        if player_id is None or pd.isna(player_id): continue

        raw_role = str(row.get('role', 'BAT')).upper()
        player_role = 'BAT'
        if 'BOWLER' in raw_role or 'BOWL' in raw_role: player_role = 'BOWL'
        elif 'ALL-ROUNDER' in raw_role or 'AR' in raw_role: player_role = 'AR'
        elif 'WICKET' in raw_role or 'WK' in raw_role or 'KEEPER' in raw_role: player_role = 'WK'

        p_logs = player_history.get(int(player_id), [])
        
        # Performance trackers
        c_runs, c_balls, c_matches = 0, 0, set()
        c_runs_conceded, c_legal_balls, c_wickets = 0, 0, 0
        r_runs, r_balls, r_matches = 0, 0, set()
        r_runs_conceded, r_legal_balls, r_wickets = 0, 0, 0
        
        intl_matches = set()
        yearly_wickets = {}
        yearly_runs = {}
        last_active_year = 0

        for log in p_logs:
            if log['year'] >= auc_year: continue
            
            fmt = log['format']
            if fmt in ['test', 'odi', 't20i']:
                intl_matches.add(log['match_id'])
            
            if log['year'] > last_active_year:
                last_active_year = log['year']

            # Track peak operational years to catch injured stars
            lyear = log['year']
            if log['type'] == 'bowl':
                yearly_wickets[lyear] = yearly_wickets.get(lyear, 0) + log['wickets']
            if log['type'] == 'bat':
                yearly_runs[lyear] = yearly_runs.get(lyear, 0) + log['runs']

            if fmt in ['t20', 't20i']:
                w = log['weight']
                is_recent = (log['year'] == (auc_year - 1))
                if log['type'] == 'bat':
                    if is_recent:
                        r_runs += log['runs'] * w
                        r_balls += log['balls']
                        r_matches.add(log['match_id'])
                    else:
                        c_runs += log['runs'] * w
                        c_balls += log['balls']
                        c_matches.add(log['match_id'])
                elif log['type'] == 'bowl':
                    if is_recent:
                        r_runs_conceded += log['runs_conceded'] * w
                        r_legal_balls += log['balls_bowled']
                        r_wickets += log['wickets'] * w
                    else:
                        c_runs_conceded += log['runs_conceded'] * w
                        c_legal_balls += log['balls_bowled']
                        c_wickets += log['wickets'] * w

        c_sr = (c_runs / c_balls * 100) if c_balls > 10 else 0
        c_econ = (c_runs_conceded / (c_legal_balls / 6)) if c_legal_balls > 6 else 0
        r_sr = (r_runs / r_balls * 100) if r_balls > 10 else 0
        r_econ = (r_runs_conceded / (r_legal_balls / 6)) if r_legal_balls > 6 else 0

        # 🌟 CRITICAL FIX VARIABLES FOR INJURED SUPERSTARS 🌟
        peak_season_wickets = max(yearly_wickets.values()) if yearly_wickets else 0
        peak_season_runs = max(yearly_runs.values()) if yearly_runs else 0
        injury_gap_years = (auc_year - last_active_year) if last_active_year > 0 else 5

        feature_rows.append({
            "canonical_id": int(player_id),
            "player_name": auc_name,
            "auction_year": auc_year,
            "player_role": player_role,
            "is_overseas": 1 if 'overseas' in str(row['nationality']).lower() else 0,
            
            "career_runs": c_runs, "career_wickets": c_wickets, "career_matches": len(c_matches),
            "career_strike_rate": c_sr, "career_economy": c_econ,
            "recent_runs": r_runs, "recent_wickets": r_wickets, "recent_matches": len(r_matches),
            "recent_strike_rate": r_sr, "recent_economy": r_econ,
            
            # Peak features and injury counters
            "peak_season_wickets": peak_season_wickets,
            "peak_season_runs": peak_season_runs,
            "injury_gap_years": injury_gap_years,
            "international_prestige_score": len(intl_matches),
            
            "sold_price_crore": sold_price
        })

    features_df = pd.DataFrame(feature_rows)

    print("Computing dynamic seasonal peer Z-scores...")
    z_scored_frames = []
    metrics_to_z = ['career_runs', 'career_wickets', 'career_strike_rate', 'career_economy', 
                    'recent_runs', 'recent_wickets', 'recent_strike_rate', 'recent_economy', 
                    'peak_season_wickets', 'peak_season_runs', 'injury_gap_years', 'international_prestige_score']
    
    for (year, role), group in features_df.groupby(['auction_year', 'player_role']):
        group = group.copy()
        for metric in metrics_to_z:
            std = group[metric].std()
            mean = group[metric].mean()
            group[f'z_{metric}'] = 0.0 if pd.isna(std) or std == 0 else (group[metric] - mean) / std
        z_scored_frames.append(group)
        
    features_df = pd.concat(z_scored_frames, ignore_index=True)

    features_df = pd.get_dummies(features_df, columns=['player_role'], prefix='role')
    for role_col in ['role_BAT', 'role_BOWL', 'role_AR', 'role_WK']:
        if role_col not in features_df.columns: features_df[role_col] = 0
        features_df[role_col] = features_df[role_col].astype(int)

    features_df.to_csv(processed_dir / "player_auction_features.csv", index=False)
    print(f"[OK] High-fidelity peak-aware feature matrix saved.")

if __name__ == "__main__":
    build_auction_feature_matrix()