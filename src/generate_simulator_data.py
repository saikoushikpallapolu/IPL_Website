import pandas as pd
import numpy as np
from pathlib import Path

def generate_perfect_2025_assets():
    raw_dir = Path("data/raw/ipl_auctions")
    raw_registry_path = raw_dir / "ipl2025_auctionplayerlist.csv"
    
    print("--- [START] CRITICAL RECONSTRUCTION OF 2025 REGISTRY ---")
    
    if not raw_registry_path.exists():
        print(f"[ERROR] Missing file at: {raw_registry_path}")
        return
        
    # Load as Excel since we verified it's a renamed .xlsx workbook
    try:
        reg_df = pd.read_excel(raw_registry_path, engine='openpyxl')
    except Exception as e:
        print(f"[ERROR] Error loading file as Excel: {e}")
        return

    # Find the header row dynamically
    header_idx = None
    for idx, row in reg_df.iterrows():
        row_vals = [str(v).strip().lower() for v in row.values]
        if any('price rs' in rv for rv in row_vals) or any('2025 set' in rv for rv in row_vals):
            header_idx = idx
            break

    if header_idx is None:
        print("[ERROR] Could not locate header row in Excel file.")
        return

    # Extract headers and data
    reg_df.columns = [str(v).strip() for v in reg_df.iloc[header_idx].values]
    reg_df = reg_df.iloc[header_idx + 1:].copy()
    
    # Standardize column headers to clean lowercase strings
    reg_df.columns = [c.lower().strip() for c in reg_df.columns]
    
    compiled_rows = []
    for idx, row in reg_df.iterrows():
        first_name_col = next((c for c in reg_df.columns if 'first name' in c), None)
        surname_col = next((c for c in reg_df.columns if 'surname' in c), None)
        full_name_col = next((c for c in reg_df.columns if any(x in c for x in ['player', 'name']) and c not in [first_name_col, surname_col]), None)
        
        if first_name_col and pd.notna(row[first_name_col]):
            s_name = str(row[surname_col]).strip() if surname_col and pd.notna(row[surname_col]) else ""
            p_name = f"{str(row[first_name_col]).strip()} {s_name}".strip()
        elif full_name_col and pd.notna(row[full_name_col]):
            p_name = str(row[full_name_col]).strip()
        else:
            continue
            
        p_name = p_name.replace('nan', '').strip()
        if p_name.lower() in ['', 'none', 'total']: continue
        
        # Locate Reserve Price column strings
        price_col = next((c for c in reg_df.columns if any(x in c for x in ['price rs', 'price', 'base', 'reserve', 'cost'])), None)
        raw_price = row[price_col] if price_col else 30.0
        
        # Convert numeric Lakhs into standardized decimal Crore floats
        base_price = pd.to_numeric(raw_price, errors='coerce')
        if not pd.isna(base_price):
            if base_price > 100000:
                base_price = base_price / 10000000.0
            else:
                base_price = base_price / 100.0
        else:
            base_price = 0.30 # Standard uncapped fallback entry fee
            
        # Locate the Set Name text groupings
        set_col = next((c for c in reg_df.columns if any(x in c for x in ['2025 set', 'set'])), None)
        set_name = str(row[set_col]).strip() if set_col and pd.notna(row[set_col]) else "Capped Player Set"
        
        compiled_rows.append({
            "player_name": p_name,
            "reserve_price": np.round(base_price, 2),
            "set_name": set_name
        })
        
    final_pool_df = pd.DataFrame(compiled_rows)
    final_pool_df.to_csv(raw_dir / "registered_auction_pool_2025.csv", index=False)
    print(f"[OK] Mapped {len(final_pool_df)} registered players into target pool format.")

    # 🌟 CORE FIX: Explicitly loading the official 2025 core retention lists 🌟
    print("\nHardcoding official 2025 player rosters & structural baselines...")
    official_retentions = [
        # Mumbai Indians
        {"team_name": "MUMBAI INDIANS", "player_name": "Jasprit Bumrah", "role": "BOWL", "nationality": "Indian", "cost": 18.0},
        {"team_name": "MUMBAI INDIANS", "player_name": "Suryakumar Yadav", "role": "BAT", "nationality": "Indian", "cost": 16.35},
        {"team_name": "MUMBAI INDIANS", "player_name": "Hardik Pandya", "role": "AR", "nationality": "Indian", "cost": 16.35},
        {"team_name": "MUMBAI INDIANS", "player_name": "Rohit Sharma", "role": "BAT", "nationality": "Indian", "cost": 16.30},
        {"team_name": "MUMBAI INDIANS", "player_name": "Tilak Varma", "role": "BAT", "nationality": "Indian", "cost": 8.00},
        
        # Royal Challengers Bengaluru
        {"team_name": "ROYAL CHALLENGERS BENGALURU", "player_name": "Virat Kohli", "role": "BAT", "nationality": "Indian", "cost": 21.0},
        {"team_name": "ROYAL CHALLENGERS BENGALURU", "player_name": "Rajat Patidar", "role": "BAT", "nationality": "Indian", "cost": 11.0},
        {"team_name": "ROYAL CHALLENGERS BENGALURU", "player_name": "Yash Dayal", "role": "BOWL", "nationality": "Indian", "cost": 5.0},
        
        # Chennai Super Kings
        {"team_name": "CHENNAI SUPER KINGS", "player_name": "Ruturaj Gaikwad", "role": "BAT", "nationality": "Indian", "cost": 18.0},
        {"team_name": "CHENNAI SUPER KINGS", "player_name": "Ravindra Jadeja", "role": "AR", "nationality": "Indian", "cost": 18.0},
        {"team_name": "CHENNAI SUPER KINGS", "player_name": "Matheesha Pathirana", "role": "BOWL", "nationality": "Overseas", "cost": 13.0},
        {"team_name": "CHENNAI SUPER KINGS", "player_name": "Shivam Dube", "role": "AR", "nationality": "Indian", "cost": 12.0},
        {"team_name": "CHENNAI SUPER KINGS", "player_name": "MS Dhoni", "role": "WK", "nationality": "Indian", "cost": 4.0},

        # Kolkata Knight Riders
        {"team_name": "KOLKATA KNIGHT RIDERS", "player_name": "Rinku Singh", "role": "BAT", "nationality": "Indian", "cost": 13.0},
        {"team_name": "KOLKATA KNIGHT RIDERS", "player_name": "Varun Chakaravarthy", "role": "BOWL", "nationality": "Indian", "cost": 12.0},
        {"team_name": "KOLKATA KNIGHT RIDERS", "player_name": "Sunil Narine", "role": "AR", "nationality": "Overseas", "cost": 12.0},
        {"team_name": "KOLKATA KNIGHT RIDERS", "player_name": "Andre Russell", "role": "AR", "nationality": "Overseas", "cost": 12.0},
        {"team_name": "KOLKATA KNIGHT RIDERS", "player_name": "Harshit Rana", "role": "BOWL", "nationality": "Indian", "cost": 4.0},
        {"team_name": "KOLKATA KNIGHT RIDERS", "player_name": "Ramandeep Singh", "role": "AR", "nationality": "Indian", "cost": 4.0},

        # Rajasthan Royals
        {"team_name": "RAJASTHAN ROYALS", "player_name": "Sanju Samson", "role": "WK", "nationality": "Indian", "cost": 18.0},
        {"team_name": "RAJASTHAN ROYALS", "player_name": "Yashasvi Jaiswal", "role": "BAT", "nationality": "Indian", "cost": 18.0},
        {"team_name": "RAJASTHAN ROYALS", "player_name": "Riyan Parag", "role": "AR", "nationality": "Indian", "cost": 14.0},
        {"team_name": "RAJASTHAN ROYALS", "player_name": "Dhruv Jurel", "role": "WK", "nationality": "Indian", "cost": 14.0},
        {"team_name": "RAJASTHAN ROYALS", "player_name": "Shimron Hetmyer", "role": "BAT", "nationality": "Overseas", "cost": 11.0},
        {"team_name": "RAJASTHAN ROYALS", "player_name": "Sandeep Sharma", "role": "BOWL", "nationality": "Indian", "cost": 4.0},

        # Sunrisers Hyderabad
        {"team_name": "SUNRISERS HYDERABAD", "player_name": "Heinrich Klaasen", "role": "WK", "nationality": "Overseas", "cost": 23.0},
        {"team_name": "SUNRISERS HYDERABAD", "player_name": "Pat Cummins", "role": "AR", "nationality": "Overseas", "cost": 18.0},
        {"team_name": "SUNRISERS HYDERABAD", "player_name": "Abhishek Sharma", "role": "AR", "nationality": "Indian", "cost": 14.0},
        {"team_name": "SUNRISERS HYDERABAD", "player_name": "Travis Head", "role": "BAT", "nationality": "Overseas", "cost": 14.0},
        {"team_name": "SUNRISERS HYDERABAD", "player_name": "Nitish Kumar Reddy", "role": "AR", "nationality": "Indian", "cost": 6.0},

        # Gujarat Titans
        {"team_name": "GUJARAT TITANS", "player_name": "Rashid Khan", "role": "BOWL", "nationality": "Overseas", "cost": 18.0},
        {"team_name": "GUJARAT TITANS", "player_name": "Shubman Gill", "role": "BAT", "nationality": "Indian", "cost": 16.5},
        {"team_name": "GUJARAT TITANS", "player_name": "Sai Sudharsan", "role": "BAT", "nationality": "Indian", "cost": 8.5},
        {"team_name": "GUJARAT TITANS", "player_name": "Rahul Tewatia", "role": "AR", "nationality": "Indian", "cost": 4.0},
        {"team_name": "GUJARAT TITANS", "player_name": "Shahrukh Khan", "role": "BAT", "nationality": "Indian", "cost": 4.0},

        # Delhi Capitals
        {"team_name": "DELHI CAPITALS", "player_name": "Axar Patel", "role": "AR", "nationality": "Indian", "cost": 16.50},
        {"team_name": "DELHI CAPITALS", "player_name": "Kuldeep Yadav", "role": "BOWL", "nationality": "Indian", "cost": 13.25},
        {"team_name": "DELHI CAPITALS", "player_name": "Tristan Stubbs", "role": "WK", "nationality": "Overseas", "cost": 10.00},
        {"team_name": "DELHI CAPITALS", "player_name": "Abishek Porel", "role": "WK", "nationality": "Indian", "cost": 4.00},

        # Lucknow Super Giants
        {"team_name": "LUCKNOW SUPER GIANTS", "player_name": "Nicholas Pooran", "role": "WK", "nationality": "Overseas", "cost": 21.0},
        {"team_name": "LUCKNOW SUPER GIANTS", "player_name": "Ravi Bishnoi", "role": "BOWL", "nationality": "Indian", "cost": 11.0},
        {"team_name": "LUCKNOW SUPER GIANTS", "player_name": "Mayank Yadav", "role": "BOWL", "nationality": "Indian", "cost": 11.0},
        {"team_name": "LUCKNOW SUPER GIANTS", "player_name": "Mohsin Khan", "role": "BOWL", "nationality": "Indian", "cost": 4.0},
        {"team_name": "LUCKNOW SUPER GIANTS", "player_name": "Ayush Badoni", "role": "BAT", "nationality": "Indian", "cost": 4.0},

        # Punjab Kings
        {"team_name": "PUNJAB KINGS", "player_name": "Shashank Singh", "role": "BAT", "nationality": "Indian", "cost": 5.5},
        {"team_name": "PUNJAB KINGS", "player_name": "Prabhsimran Singh", "role": "WK", "nationality": "Indian", "cost": 4.0}
    ]
    
    squads_df = pd.DataFrame(official_retentions)
    squads_df.to_csv(raw_dir / "squads_2024.csv", index=False)
    print(f"[OK] Generated strict official 'squads_2024.csv' with locked core values.")

if __name__ == "__main__":
    generate_perfect_2025_assets()