import zipfile
import shutil
from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
from rapidfuzz import process, fuzz

def smart_locate_column(columns, keywords, default_name="column"):
    """
    Scans headers for prioritized keywords. Returns the column name if found.
    """
    for kw in keywords:
        matches = [c for c in columns if kw in c]
        if matches:
            return matches[0]
    return None # Return None instead of crashing if a non-mandatory column is missing

def extract_and_resolve_master_roster():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("--- PHASE 1: Extracting Consolidated Datasets from Zips ---")
    zip_mappings = {
        "ipl_performance/ipl_match_deliveries.csv.zip": "ipl_match_deliveries.csv",
        "domestic_leagues/smat_deliveries.csv.zip": "smat_deliveries.csv",
        "foreign_leagues/t20i_deliveries.csv.zip": "t20i_deliveries.csv",
        "foreign_leagues/bbl_deliveries.csv.zip": "bbl_deliveries.csv",
        "foreign_leagues/psl_deliveries.csv.zip": "psl_deliveries.csv",
        "foreign_leagues/cpl_deliveries.csv.zip": "cpl_deliveries.csv",
        "foreign_leagues/sa20_deliveries.csv.zip": "sa20_deliveries.csv",
        "foreign_leagues/hundred_deliveries.csv.zip": "hundred_deliveries.csv"
    }

    for rel_zip_path, output_name in zip_mappings.items():
        zip_path = raw_dir / rel_zip_path
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                if 'all_matches.csv' in z.namelist():
                    with z.open('all_matches.csv') as source:
                        with open(processed_dir / output_name, 'wb') as target:
                            shutil.copyfileobj(source, target)

    print("\n--- PHASE 1B: Concatenating Consolidated Datasets into all_matches.csv ---")
    all_matches_path = processed_dir / "all_matches.csv"
    dfs = []
    for output_name in zip_mappings.values():
        f_path = processed_dir / output_name
        if f_path.exists():
            print(f"  Reading {output_name} for concatenation...")
            df = pd.read_csv(f_path, low_memory=False)
            
            # Add league metadata dynamically
            league = output_name.replace("_deliveries.csv", "").replace("_match_deliveries.csv", "")
            match_type = "t20i" if league == "t20i" else "t20"
            df['league_name'] = league
            df['match_type'] = match_type
            
            dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(all_matches_path, index=False)
        print(f"  [OK] Created global dataset: {all_matches_path} ({len(combined_df)} rows)")
    else:
        print("  [ERROR] No delivery files found for concatenation!")

    print("\n--- PHASE 2: Unifying 2025 Team Sheets with Adaptive Headers ---")
    combined_2025 = []
    auction_2025_zip = raw_dir / "ipl_auctions/auction_2025.zip"
    
    if auction_2025_zip.exists():
        with zipfile.ZipFile(auction_2025_zip, 'r') as z:
            for file_name in z.namelist():
                if file_name.endswith('.csv') and not file_name.startswith('__MACOSX'):
                    with z.open(file_name) as f:
                        df = pd.read_csv(f)
                        df.columns = [col.lower().strip() for col in df.columns]
                        
                        # Use your exact schema mappings
                        name_col = smart_locate_column(df.columns, ['player', 'name'], "Player Name")
                        price_col = smart_locate_column(df.columns, ['winning bid', 'sold', 'final', 'price', 'amount'], "Sold Price")
                        role_col = smart_locate_column(df.columns, ['role', 'type', 'skill', 'category'], "Player Role")
                        
                        if not name_col or not price_col:
                            print(f"⚠️ Skipping file {file_name} due to missing mandatory columns.")
                            continue
                            
                        print(f"Processing {Path(file_name).name} -> Mapping '{name_col}' and '{price_col}'")
                        
                        for _, row in df.iterrows():
                            p_name = str(row[name_col]).strip()
                            if pd.isna(row[name_col]) or p_name.lower() in ['none', 'total', '']:
                                continue
                                
                            # Detect RCB, RR, or SRH sheets which have 10x inflated pricing
                            file_stem = Path(file_name).name.upper()
                            is_inflated = any(x in file_stem for x in ['RCB', 'RR', 'SRH'])

                            raw_price = pd.to_numeric(row[price_col], errors='coerce')
                            if pd.isna(raw_price) or raw_price <= 0:
                                continue
                                
                            if is_inflated:
                                raw_price = raw_price / 10.0
                                
                            # Convert raw rupees back to Crores dynamically
                            if raw_price > 100000:
                                raw_price = raw_price / 10000000.0

                            # Safe default fallback for role if it doesn't exist in the file
                            role_val = str(row[role_col]).upper() if role_col else 'ALL-ROUNDER'

                            combined_2025.append({
                                "player_name": p_name,
                                "price_in_crores": raw_price,
                                "auction_year": 2025,
                                "role": role_val,
                                "nationality": "Overseas" if any(x in p_name.lower() for x in ['overseas', '(o)', 'foreign']) else "Indian"
                            })
                            
        if combined_2025:
            df_2025 = pd.DataFrame(combined_2025)
            df_2025.to_csv(processed_dir / "standardized_2025_auction.csv", index=False)
            print(f"[OK] Standardized {len(df_2025)} player records from your 2025 grids.")
        else:
            df_2025 = pd.DataFrame(columns=["player_name", "price_in_crores", "auction_year", "role", "nationality"])
            print("⚠️ No records were successfully processed for 2025.")

    print("\n--- PHASE 3: Re-Building Master Identity Matrix ---")
    global_names = set()
    for f_name in zip_mappings.values():
        f_path = processed_dir / f_name
        if f_path.exists():
            for chunk in pd.read_csv(f_path, usecols=lambda c: any(x in c for x in ['batter', 'striker', 'bowler']), chunksize=100000, low_memory=False):
                bat_col = 'striker' if 'striker' in chunk.columns else ('batter' if 'batter' in chunk.columns else None)
                if bat_col: global_names.update(chunk[bat_col].dropna().unique())
                if 'bowler' in chunk.columns: global_names.update(chunk['bowler'].dropna().unique())

    hist_auction_df = pd.read_csv(raw_dir / "ipl_auctions/historical_auctions.csv.csv", low_memory=False)
    hist_auction_df.columns = [c.lower().strip() for c in hist_auction_df.columns]
    auc_name_col = [c for c in hist_auction_df.columns if 'player' in c or 'name' in c][0]
    
    auc_pool_names = set(hist_auction_df[auc_name_col].dropna().unique())
    if not df_2025.empty:
        auc_pool_names = auc_pool_names.union(set(df_2025['player_name'].unique()))
        
    auc_pool_list = list(auc_pool_names)

    def find_initial_match(cric_name, auction_list):
        manual_overrides = {
            "V Kohli": "Virat Kohli", "MS Dhoni": "Mahendra Singh Dhoni", 
            "RG Sharma": "Rohit Sharma", "AB de Villiers": "Abraham de Villiers"
        }
        if cric_name in manual_overrides: return manual_overrides[cric_name]
            
        cric_tokens = cric_name.split()
        if len(cric_tokens) == 0: return cric_name
        
        last_name = cric_tokens[-1].lower()
        first_initial = cric_tokens[0][0].lower()

        candidates = []
        for a_name in auction_list:
            a_tokens = str(a_name).split()
            if len(a_tokens) >= 2:
                if a_tokens[-1].lower() == last_name and a_tokens[0][0].lower() == first_initial:
                    candidates.append(a_name)
                    
        if len(candidates) == 1: return candidates[0]
            
        match = process.extractOne(cric_name, auction_list, scorer=fuzz.token_sort_ratio)
        if match and match[1] > 65: return match[0]
        return cric_name

    master_rows = []
    for i, g_name in enumerate(sorted(global_names)):
        g_name_str = str(g_name).strip()
        assigned_auction_name = find_initial_match(g_name_str, auc_pool_list)

        master_rows.append({
            "canonical_id": 10000 + i,
            "canonical_name": assigned_auction_name,
            "cricsheet_name": g_name_str,
            "auction_name": assigned_auction_name
        })

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(processed_dir / "players_master.csv", index=False)
    print("[OK] Global name alignment index re-mapped successfully without structural crashes.")

if __name__ == "__main__":
    extract_and_resolve_master_roster()