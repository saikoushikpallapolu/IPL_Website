import pandas as pd
import numpy as np
import joblib
import sys
import time
import select
from pathlib import Path

class IPL2025IncrementalSimulator:
    def __init__(self):
        self.raw_dir = Path("data/raw/ipl_auctions")
        self.processed_dir = Path("data/processed")
        self.model_path = Path("models/price_model.joblib")
        
        if not (self.raw_dir / "squads_2024.csv").exists() or not (self.raw_dir / "registered_auction_pool_2025.csv").exists():
            print("[ERROR] Simulation assets missing. Run generate_simulator_data.py first.")
            sys.exit(1)
            
        self.squads_df = pd.read_csv(self.raw_dir / "squads_2024.csv")
        self.pool_df = pd.read_csv(self.raw_dir / "registered_auction_pool_2025.csv")
        self.features_df = pd.read_csv(self.processed_dir / "player_auction_features.csv")
        self.model = joblib.load(self.model_path)
        
        self.z_features = [
            'z_career_runs', 'z_career_wickets', 'z_career_strike_rate', 'z_career_economy',
            'z_recent_runs', 'z_recent_wickets', 'z_recent_strike_rate', 'z_recent_economy',
            'z_peak_season_wickets', 'z_peak_season_runs', 'z_injury_gap_years',
            'z_international_prestige_score', 'is_overseas',
            'role_BAT', 'role_BOWL', 'role_AR', 'role_WK'
        ]

        # Starting state initialization configurations
        self.teams = {
            "MUMBAI INDIANS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "CHENNAI SUPER KINGS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "ROYAL CHALLENGERS BENGALURU": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "KOLKATA KNIGHT RIDERS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "RAJASTHAN ROYALS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "SUNRISERS HYDERABAD": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "DELHI CAPITALS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "PUNJAB KINGS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "LUCKNOW SUPER GIANTS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []},
            "GUJARAT TITANS": {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []}
        }
        self.user_team = ""

    def load_pre_auction_balances(self):
        print("\n=======================================================")
        print("    TATA IPL 2025 LIVE INCREMENTAL AUCTION SIMULATOR   ")
        print("=======================================================")
        
        team_list = list(self.teams.keys())
        for i, t in enumerate(team_list, 1):
            print(f"[{i}] {t}")
            
        while True:
            try:
                choice = int(input("\nSelect your franchise index number: "))
                if 1 <= choice <= 10:
                    self.user_team = team_list[choice - 1]
                    break
            except ValueError: pass

        print(f"\n[OK] Loading official financial accounts into live simulation state...")
        
        # Parse through rows to apply direct deductions onto starting purses
        for _, row in self.squads_df.iterrows():
            t_name = row['team_name']
            p_name = row['player_name']
            cost = float(row['cost'])
            is_os = 1 if row['nationality'] == 'Overseas' else 0
            
            self.teams[t_name]['purse'] -= cost
            self.teams[t_name]['slots'] += 1
            if is_os: self.teams[t_name]['overseas'] += 1
            self.teams[t_name]['roster'].append(f"{p_name} (Retained - {cost} Cr)")

        print("\n--- OFFICIAL ROOM STARTING PURSES (POST-RETENTION) ---")
        for t_name, data in self.teams.items():
            print(f"  {t_name:<30} | Wallet: Rs. {data['purse']:>5.2f} Cr | Squad: {data['slots']}/25")
        print("=======================================================\n")

    def get_next_increment(self, current_price):
        if current_price < 2.0: return 0.10
        elif current_price < 5.0: return 0.20
        return 0.50

    def timeout_input(self, prompt, timeout=5):
        # Multi-platform interactive terminal polling framework
        print(prompt, end='', flush=True)
        if sys.platform == "win32":
            # Native Windows msvcrt tracking loop
            import msvcrt
            start_time = time.time()
            input_str = ""
            while time.time() - start_time < timeout:
                if msvcrt.kbhit():
                    ch = msvcrt.getche().decode('utf-8')
                    if ch == '\r' or ch == '\n':
                        print()
                        return input_str.strip()
                    input_str += ch
                time.sleep(0.05)
            print()
            return ""
        else:
            # POSIX UNIX select tracking loop
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if r: return sys.stdin.readline().strip()
            return ""

    def execute_live_auction_loop(self):
        retained_names = [p.split(" (")[0] for t in self.teams.values() for p in t['roster']]
        active_pool = self.pool_df[~self.pool_df['player_name'].isin(retained_names)].copy()
        
        for set_title, set_frame in active_pool.groupby('set_name', sort=False):
            print(f"\n>>> NOW DRAWING FROM REGISTRY: {set_title.upper()} <<<")
            shuffled_set = set_frame.sample(frac=1.0)
            
            for _, player_row in shuffled_set.iterrows():
                p_name = player_row['player_name']
                reserve_price = float(player_row['reserve_price'])
                
                feat_row = self.features_df[(self.features_df['player_name'] == p_name) & (self.features_df['auction_year'] == 2025)]
                if feat_row.empty:
                    base_val = reserve_price; p_role = "BAT"; is_os = 0; prestige = 0.0
                else:
                    feat_row = feat_row.iloc[0]
                    p_role = "BOWL" if feat_row['role_BOWL'] == 1 else ("AR" if feat_row['role_AR'] == 1 else ("WK" if feat_row['role_WK'] == 1 else "BAT"))
                    is_os = int(feat_row['is_overseas'])
                    prestige = float(feat_row['z_international_prestige_score'])
                    
                    pred_input = pd.DataFrame([feat_row[self.z_features]])
                    base_val = np.expm1(self.model.predict(pred_input)[0])
                    base_val = np.maximum(base_val, reserve_price)
                print(f"\n-------------------------------------------------------")
                print(f"PLAYER UP: {p_name} ({p_role}) | {'Overseas' if is_os else 'Indian'}")
                print(f"Opening Reserve Price: Rs. {reserve_price:.2f} Cr")
                print(f"-------------------------------------------------------")
                
                # Calculate absolute ceiling bounds for the 9 computer profiles
                ai_ceilings = {}
                for t_name, data in self.teams.items():
                    if t_name == self.user_team or data['purse'] < reserve_price or data['slots'] >= 25: continue
                    if is_os and data['overseas'] >= 8: continue
                    
                    purse_ratio = data['purse'] / np.mean([x['purse'] for x in self.teams.values()])
                    role_count = sum(1 for p in data['roster'] if f"({p_role})" in p)
                    demand = 1.0 + (1.0 / (role_count + 1))
                    
                    # Apply substantial weight premiums to high prestige assets
                    scarcity = 3.5 if prestige > 0.6 else 1.0
                    
                    ceiling = base_val * purse_ratio * demand * scarcity
                    if ceiling >= reserve_price:
                        ai_ceilings[t_name] = np.round(np.minimum(ceiling, data['purse']), 2)

                current_bid = reserve_price
                current_leader = ""
                
                # 📢 Opening interaction hook checkpoint
                print(f"\n[BID ROOM] Opening floor at base price Rs. {reserve_price:.2f} Cr.")
                user_action = self.timeout_input("Type 'b' to open bid entry or enter to pass (5s window): ", timeout=5)
                
                if user_action.lower() == 'b':
                    current_leader = self.user_team
                    print(f"--> [USER TEAM] places the opening bid of Rs. {current_bid:.2f} Cr!")
                elif not ai_ceilings:
                    print(f"🛑 No franchise matches opening terms. {p_name} goes UNSOLD.")
                    continue
                else:
                    # Select starting AI bidder
                    current_leader = np.random.choice(list(ai_ceilings.keys()))
                    print(f"--> [{current_leader}] places the opening bid of Rs. {current_bid:.2f} Cr!")

                # ⚔️ LIVE SEQUENTIAL STEPPED AUCTION ESCALATION WAR ⚔️
                active_war = True
                while active_war:
                    time.sleep(1.0)
                    inc = self.get_next_increment(current_bid)
                    next_bid = np.round(current_bid + inc, 2)
                    
                    # Filter down active computer contenders capable of matching next increment tiers
                    viable_ai = [t for t, ceil in ai_ceilings.items() if ceil >= next_bid and t != current_leader]
                    
                    if current_leader == self.user_team:
                        if not viable_ai:
                            print(f"\n Hammer falls... Going once... twice...")
                            time.sleep(1.0)
                            self.allocate(self.user_team, p_name, current_bid, p_role, is_os)
                            active_war = False
                        else:
                            next_bidder = np.random.choice(viable_ai)
                            current_bid = next_bid
                            current_leader = next_bidder
                            print(f"  --> [{current_leader}] raises bid to Rs. {current_bid:.2f} Cr!")
                    else:
                        # Current leader is an AI. Prompt user or trigger competing AI counters
                        if self.teams[self.user_team]['purse'] >= next_bid and (self.teams[self.user_team]['overseas'] < 8 or not is_os):
                            print(f"\n Current highest bid: Rs. {current_bid:.2f} Cr held by {current_leader}.")
                            u_choice = self.timeout_input(f"  -> Enter 'b' to counter-bid at Rs. {next_bid:.2f} Cr (5s window) or press Enter: ", timeout=5)
                            
                            if u_choice.lower() == 'b':
                                current_bid = next_bid
                                current_leader = self.user_team
                                print(f"  --> [USER TEAM] counter-bids at Rs. {current_bid:.2f} Cr!")
                                continue
                                
                        # If user passes or times out, let competing AI franchises trade blows
                        if viable_ai:
                            next_bidder = np.random.choice(viable_ai)
                            current_bid = next_bid
                            current_leader = next_bidder
                            print(f"  --> [{current_leader}] counters and raises to Rs. {current_bid:.2f} Cr!")
                        else:
                            print(f"\n Hammer falls... Going once... twice...")
                            time.sleep(1.0)
                            self.allocate(current_leader, p_name, current_bid, p_role, is_os)
                            active_war = False

                cmd = input("\nEnter 'status' to review boards or hit enter to draw next name: ").strip().lower()
                if cmd == 'status': self.print_standings()

    def allocate(self, team, name, price, role, is_os):
        self.teams[team]['purse'] -= price
        self.teams[team]['slots'] += 1
        if is_os: self.teams[team]['overseas'] += 1
        self.teams[team]['roster'].append(f"{name} ({role}) - Rs. {price:.2f} Cr")
        print(f"[SOLD] {name} secured by {team} for Rs. {price:.2f} Crore!")

    def print_standings(self):
        print("\n================ LIVE AUCTION ROOM TRANSACTION RECORD ================")
        for t_name, data in self.teams.items():
            print(f"  {t_name:<28} | Purse: Rs. {data['purse']:>5.2f} Cr | Squad: {data['slots']}/25 (OS: {data['overseas']}/8)")
        print("======================================================================\n")

if __name__ == "__main__":
    sim = IPL2025IncrementalSimulator()
    sim.load_pre_auction_balances()
    sim.execute_live_auction_loop()