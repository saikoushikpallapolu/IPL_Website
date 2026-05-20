import pandas as pd
import numpy as np
import joblib
import sys
import time
from pathlib import Path

# Platform-specific unbuffered keypress reader setup
if sys.platform == "win32":
    import msvcrt
else:
    import select
    import tty
    import termios

class IPL2025DuelTimerSimulator:
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

        self.franchise_dna = {
            "CHENNAI SUPER KINGS": {"bias_metric": "z_international_prestige_score", "bias_weight": 1.35, "target_role": "AR"},
            "KOLKATA KNIGHT RIDERS": {"bias_metric": "z_recent_strike_rate", "bias_weight": 1.30, "target_role": "AR"},
            "ROYAL CHALLENGERS BENGALURU": {"bias_metric": "z_career_runs", "bias_weight": 1.40, "target_role": "BAT"},
            "MUMBAI INDIANS": {"bias_metric": "z_peak_season_wickets", "bias_weight": 1.30, "target_role": "BOWL"},
            "PUNJAB KINGS": {"bias_metric": "z_recent_runs", "bias_weight": 1.25, "target_role": "BAT"},
            "RAJASTHAN ROYALS": {"bias_metric": "z_career_economy", "bias_weight": 1.25, "target_role": "BOWL"},
            "SUNRISERS HYDERABAD": {"bias_metric": "z_peak_season_runs", "bias_weight": 1.35, "target_role": "BAT"},
            "DELHI CAPITALS": {"bias_metric": "z_recent_wickets", "bias_weight": 1.25, "target_role": "WK"},
            "LUCKNOW SUPER GIANTS": {"bias_metric": "z_career_strike_rate", "bias_weight": 1.25, "target_role": "BOWL"},
            "GUJARAT TITANS": {"bias_metric": "z_career_economy", "bias_weight": 1.30, "target_role": "AR"}
        }

        self.teams = {t: {"purse": 120.0, "slots": 0, "overseas": 0, "roster": []} for t in self.franchise_dna.keys()}
        self.user_team = ""
        self.historical_player_team_map = {}

    def load_and_initialize_balances(self):
        print("\n=======================================================")
        print("    TATA IPL 2025 REAL-TIME TIMED ACTION SIMULATOR     ")
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

        print(f"\n[OK] Managing Front Office of: {self.user_team}")
        print("-" * 55)

        for _, row in self.squads_df.iterrows():
            self.historical_player_team_map[str(row['player_name']).strip()] = row['team_name']

        # Manual User Retentions
        user_candidates = [p for p, t in self.historical_player_team_map.items() if t == self.user_team]
        user_retained = []
        
        if user_candidates:
            print(f"\nAvailable players from your team's 2024 roster:")
            for idx, p in enumerate(user_candidates, 1):
                print(f"  ({idx}) {p}")
            print("\nEnter player numbers to retain. Type 'done' to finish.")
            while len(user_retained) < 6:
                ans = input(f"Select player number ({len(user_retained)}/6 retained): ").strip()
                if ans.lower() == 'done': break
                try:
                    p_num = int(ans)
                    if 1 <= p_num <= len(user_candidates):
                        p_name = user_candidates[p_num - 1]
                        if p_name not in user_retained:
                            user_retained.append(p_name)
                            print(f"   [ADDED] {p_name} queued.")
                except ValueError: pass

        slabs = {1: 18.0, 2: 14.0, 3: 11.0, 4: 18.0, 5: 14.0, 6: 4.0}
        for idx, p_name in enumerate(user_retained, 1):
            cost = slabs.get(idx, 12.0)
            m_row = self.squads_df[self.squads_df['player_name'] == p_name]
            is_os = 1 if not m_row.empty and m_row.iloc[0]['nationality'] == 'Overseas' else 0
            self.teams[self.user_team]['purse'] -= cost
            self.teams[self.user_team]['slots'] += 1
            if is_os: self.teams[self.user_team]['overseas'] += 1
            self.teams[self.user_team]['roster'].append(f"{p_name} (Retained - {cost} Cr)")

        # Automated Real-World Retentions for the 9 AI Teams
        for _, row in self.squads_df.iterrows():
            t_name = row['team_name']
            if t_name == self.user_team: continue
            self.teams[t_name]['purse'] -= float(row['cost'])
            self.teams[t_name]['slots'] += 1
            if row['nationality'] == 'Overseas': self.teams[t_name]['overseas'] += 1
            self.teams[t_name]['roster'].append(f"{row['player_name']} (Retained - {row['cost']} Cr)")

        self.print_standings()

    def get_next_increment(self, price):
        if price < 2.0: return 0.10
        elif price < 5.0: return 0.20
        return 0.50

    def get_non_blocking_input(self, timeout_secs=30):
        """Cross-platform non-blocking key scanner with a real-time terminal countdown."""
        start = time.time()
        sys.stdout.flush()
        
        if sys.platform == "win32":
            # Windows Keyboard Polling Logic Loop
            last_printed = -1
            while time.time() - start < timeout_secs:
                rem = int(timeout_secs - (time.time() - start))
                if rem != last_printed:
                    sys.stdout.write(f"\r⏱️  Time remaining: {rem}s | Press 'b' to bid... ")
                    sys.stdout.flush()
                    last_printed = rem
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    if ch == 'b':
                        print("\n[OK] Input caught! Entering floor...")
                        return 'b'
                time.sleep(0.1)
            print("\n⏰ Time Expired!")
            return ''
        else:
            # POSIX Unix Keyboard Polling Logic Loop
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                last_printed = -1
                while time.time() - start < timeout_secs:
                    rem = int(timeout_secs - (time.time() - start))
                    if rem != last_printed:
                        sys.stdout.write(f"\r⏱️  Time remaining: {rem}s | Press 'b' to bid... ")
                        sys.stdout.flush()
                        last_printed = rem
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        ch = sys.stdin.read(1).lower()
                        if ch == 'b':
                            print("\n[OK] Input caught! Entering floor...")
                            return 'b'
            finally:
                termios.tcsetattr(sys.stdin, old_settings)
            print("\n⏰ Time Expired!")
            return ''

    def execute_live_auction_loop(self):
        retained_names = [p.split(" (")[0] for t in self.teams.values() for p in t['roster']]
        active_pool = self.pool_df[~self.pool_df['player_name'].isin(retained_names)].copy()
        
        for set_title, set_frame in active_pool.groupby('set_name', sort=False):
            print(f"\n=======================================================")
            print(f"🔥 ACTIVE SET RUNNING: {set_title.upper()}")
            print("=======================================================")
            
            shuffled_set = set_frame.sample(frac=1.0)
            for _, player_row in shuffled_set.iterrows():
                p_name = player_row['player_name']
                reserve_price = float(player_row['reserve_price'])
                past_owner = self.historical_player_team_map.get(p_name, "None")
                
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

                print(f"\n🏏 OUT OF THE BAG: {p_name} ({p_role}) | {'Overseas' if is_os else 'Indian'}")
                if past_owner != "None": print(f"🏠 2024 Incumbent Past Team: {past_owner}")
                print(f"Opening Reserve Price: Rs. {reserve_price:.2f} Cr")
                print(f"-------------------------------------------------------")
                
                # Pre-calculate active AI ceilings
                ai_ceilings = {}
                for t_name, data in self.teams.items():
                    if t_name == self.user_team or data['purse'] < reserve_price or data['slots'] >= 25: continue
                    if is_os and data['overseas'] >= 8: continue
                    
                    purse_ratio = data['purse'] / np.mean([x['purse'] for x in self.teams.values()])
                    role_count = sum(1 for p in data['roster'] if f"({p_role})" in p)
                    demand = 1.0 + (1.0 / (role_count + 1))
                    dna_profile = self.franchise_dna[t_name]
                    dna_bonus = dna_profile['bias_weight'] if not feat_row.empty and float(feat_row.get(dna_profile['bias_metric'], 0)) > 0.5 else 1.0
                    incumbency = 1.30 if past_owner == t_name else 1.0
                    scarcity = 3.5 if prestige > 0.5 else 1.0
                    
                    ceiling = base_val * purse_ratio * demand * dna_bonus * incumbency * scarcity
                    if ceiling >= reserve_price:
                        ai_ceilings[t_name] = np.round(np.minimum(ceiling, data['purse']), 2)

                current_bid = reserve_price
                current_leader = ""
                current_challenger = ""

                # 🌟 STEP 1: OPENING BID WINDOW (30 SECONDS) 🌟
                print(f"\n[FLOOR INVITE] 30 seconds to open bidding at base Rs. {reserve_price:.2f} Cr.")
                user_keypress = self.get_non_blocking_input(timeout_secs=30)
                
                if user_keypress == 'b':
                    current_leader = self.user_team
                    print(f"\n[BID] USER TEAM opens the floor at Rs. {current_bid:.2f} Cr!")
                    # Select one random interested AI team to form the initial duel
                    viable_ai = [t for t, ceil in ai_ceilings.items() if ceil >= (current_bid + self.get_next_increment(current_bid))]
                    if viable_ai:
                        current_challenger = np.random.choice(viable_ai)
                        print(f"[DUEL START] {current_challenger} raises their paddle to challenge you!")
                else:
                    # User passed opening. Check if any AI wants to open the floor
                    viable_ai = list(ai_ceilings.keys())
                    if len(viable_ai) >= 2:
                        sampled = np.random.choice(viable_ai, size=2, replace=False)
                        current_leader = sampled[0]
                        current_challenger = sampled[1]
                        print(f"\n[BID] {current_leader} opens floor at Rs. {current_bid:.2f} Cr.")
                        print(f"[DUEL START] {current_challenger} steps up to challenge them!")
                    elif len(viable_ai) == 1:
                        current_leader = viable_ai[0]
                        print(f"\n[BID] {current_leader} opens floor unchallenged at Rs. {current_bid:.2f} Cr.")
                    else:
                        print(f"\n🛑 [UNSOLD] No bids inside the 30s window. Player leaves the room.")
                        continue

                # 🌟 STEP 2: STRICT TWO-TEAM LOCKED DUEL LOOP 🌟
                bidding_active = True
                while bidding_active:
                    inc = self.get_next_increment(current_bid)
                    next_bid = np.round(current_bid + inc, 2)
                    
                    if current_challenger != "":
                        # Strict 2-team battle phase
                        time.sleep(0.8) # Pacing break for readable streaming logs
                        if current_challenger == self.user_team:
                            # Prompt the user with a 30s deadline inside the duel
                            print(f"\n[DUEL PROFILE] Current Bid: Rs. {current_bid:.2f} Cr held by {current_leader}.")
                            print(f"👉 Your turn to counter inside the duel. Target: Rs. {next_bid:.2f} Cr.")
                            user_choice = self.get_non_blocking_input(timeout_secs=30)
                            
                            if user_choice == 'b':
                                if self.teams[self.user_team]['purse'] >= next_bid:
                                    current_bid = next_bid
                                    current_leader = self.user_team
                                    # Verify if the opponent AI team can afford to continue the duel
                                    if ai_ceilings.get(current_challenger, 0) >= (current_bid + self.get_next_increment(current_bid)):
                                        # Opponent stays as challenger, roles reverse next turn
                                        pass 
                                    else:
                                        print(f"\n[EXIT] {current_challenger} hits their spending limit and backs out!")
                                        current_challenger = ""
                                else:
                                    print("\n[WARNING] Insufficient funds! You are forced out of the duel.")
                                    self.allocate(current_leader, p_name, current_bid, p_role, is_os)
                                    bidding_active = False
                            else:
                                # User timed out or passed. Duel ends immediately.
                                print(f"\n[EXIT] You backed out or ran out of time! Duel broken.")
                                self.allocate(current_leader, p_name, current_bid, p_role, is_os)
                                bidding_active = False
                        else:
                            # Challenger is an AI team
                            if ai_ceilings[current_challenger] >= next_bid:
                                current_bid = next_bid
                                current_leader, current_challenger = current_challenger, current_leader
                                print(f"  [BID] {current_leader} raises bid to Rs. {current_bid:.2f} Cr!")
                            else:
                                print(f"\n[EXIT] {current_challenger} hits their spending limit and backs out!")
                                current_challenger = ""
                    else:
                        # 🌟 STEP 3: VACANCY ROOM PROMPT (30 SECOND HIJACK WINDOW) 🌟
                        # The duel broke because someone dropped out. Open the floor to the other 8 teams.
                        if current_leader == self.user_team:
                            # User is leading unchallenged. Check if any outside AI wants to enter the room.
                            viable_outside_ai = [t for t, ceil in ai_ceilings.items() if ceil >= next_bid]
                            if viable_outside_ai:
                                current_challenger = np.random.choice(viable_outside_ai)
                                print(f"\n[NEW ENTER] {current_challenger} steps onto the floor to challenge you at Rs. {next_bid:.2f} Cr!")
                                current_bid = next_bid
                                current_leader, current_challenger = current_challenger, current_leader
                            else:
                                print(f"\n[HAMMER] Going once... twice... unchallenged...")
                                self.allocate(self.user_team, p_name, current_bid, p_role, is_os)
                                bidding_active = False
                        else:
                            # An AI team is leading unchallenged. Open a 30s window for you or other AI to hijack.
                            print(f"\n[VACANT FLOOR] Bid is currently stalled at Rs. {current_bid:.2f} Cr held by {current_leader}.")
                            print(f"⚠️  30 seconds on the clock for outside franchises to hijack the auction at Rs. {next_bid:.2f} Cr!")
                            
                            hijack_choice = self.get_non_blocking_input(timeout_secs=30)
                            
                            if hijack_choice == 'b':
                                if self.teams[self.user_team]['purse'] >= next_bid:
                                    current_bid = next_bid
                                    current_challenger = current_leader
                                    current_leader = self.user_team
                                    print(f"\n[HIJACK] USER TEAM takes the floor! New locked duel active at Rs. {current_bid:.2f} Cr.")
                                else:
                                    print("\n[WARNING] Insufficient funds to enter.")
                            else:
                                # User passed or timed out. See if any outside computer team wants to swoop in.
                                viable_outside_ai = [t for t, ceil in ai_ceilings.items() if ceil >= next_bid and t != current_leader]
                                if viable_outside_ai and np.random.rand() > 0.4:
                                    current_challenger = np.random.choice(viable_outside_ai)
                                    current_bid = next_bid
                                    print(f"\n[HIJACK] {current_challenger} swoops in at the last second! Raised to Rs. {current_bid:.2f} Cr.")
                                    current_leader, current_challenger = current_challenger, current_leader
                                else:
                                    # No entries within 30 seconds. Sold to last standing bid.
                                    print(f"\n[HAMMER] Going once... twice... SOLD!")
                                    
                                    # Right to Match (RTM) Check Heuristic
                                    if past_owner == self.user_team and current_leader != self.user_team:
                                        print(f"\n🚨 RTM OPPORTUNITY: {p_name} was part of your 2024 roster baseline!")
                                        rtm_ans = input(f"Activate your Right to Match card to claim him for Rs. {current_bid:.2f} Cr? (y/n): ").strip().lower()
                                        if rtm_ans == 'y':
                                            self.allocate(self.user_team, p_name, current_bid, p_role, is_os)
                                            bidding_active = False
                                            continue
                                            
                                    self.allocate(current_leader, p_name, current_bid, p_role, is_os)
                                    bidding_active = False

                cmd = input("\nEnter 'status' to view leaderboards or press Enter to continue: ").strip().lower()
                if cmd == 'status': self.print_standings()

    def allocate(self, team, name, price, role, is_os):
        self.teams[team]['purse'] -= price
        self.teams[team]['slots'] += 1
        if is_os: self.teams[team]['overseas'] += 1
        self.teams[team]['roster'].append(f"{name} ({role}) - Rs. {price:.2f} Cr")
        print(f"🏆 [SOLD] {name} secured by {team} for Rs. {price:.2f} Crore!\n")

    def print_standings(self):
        print("\n================ LIVE FRANCHISE PURSE STANDINGS ================")
        for t_name, data in self.teams.items():
            print(f"🏢 {t_name:<28} | Purse Left: Rs. {data['purse']:>5.2f} Cr | Squad: {data['slots']}/25 (OS: {data['overseas']}/8)")
        print("==================================================================\n")

if __name__ == "__main__":
    sim = IPL2025DuelTimerSimulator()
    sim.load_and_initialize_balances()
    sim.execute_live_auction_loop()