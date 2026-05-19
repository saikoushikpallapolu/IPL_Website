import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

def train_peer_aware_engine():
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(processed_dir / "player_auction_features.csv")
    df = df[df['sold_price_crore'] > 0].copy()

    # Log transform target scaling to stabilize right-skewed superstar valuations
    df['target_log_price'] = np.log1p(df['sold_price_crore'])

    train_df = df[df['auction_year'] <= 2023]
    test_df = df[df['auction_year'] >= 2024]

    # Locate this block inside src/train_predictor.py and update the array list:
    feature_columns = [
        'z_career_runs', 'z_career_wickets', 'z_career_strike_rate', 'z_career_economy',
        'z_recent_runs', 'z_recent_wickets', 'z_recent_strike_rate', 'z_recent_economy',
        'z_peak_season_wickets', 'z_peak_season_runs', 'z_injury_gap_years',
        'z_international_prestige_score', 'is_overseas',
        'role_BAT', 'role_BOWL', 'role_AR', 'role_WK'
    ]
    
    X_train = train_df[feature_columns]
    y_train = train_df['target_log_price']
    
    X_test = test_df[feature_columns]
    y_test = test_df['target_log_price']

    print(f"Training Peer-Group Aware Valuation Engine on {len(X_train)} historical records...")
    
    model = xgb.XGBRegressor(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    predicted_logs = model.predict(X_test)
    predictions_crore = np.expm1(predicted_logs)
    actual_crore = test_df['sold_price_crore'].values

    # Enforce minimum possible bid floor threshold (20 Lakhs)
    predictions_crore = np.clip(predictions_crore, 0.2, None)

    # Calculate model metrics
    mae = mean_absolute_error(actual_crore, predictions_crore)
    r2 = r2_score(actual_crore, predictions_crore)
    
    print("\n================ PEER-AWARE ADVANCED PERFORMANCE SCORECARD ================")
    print(f"Mean Absolute Error (MAE): Rs. {mae:.2f} Crore")
    print(f"R-squared (R²) Score:     {r2:.4f}")
    print("==========================================================================")
    
    test_results = test_df[['player_name', 'auction_year', 'sold_price_crore']].copy()
    test_results['predicted_market_value'] = np.round(predictions_crore, 2)
    
    # 🌟 CORE UPGRADE: Randomly sample 30 players to check larger data sample diversity 🌟
    print("\nPeer-Aware Market Valuations (Random 30-Player Sample from Holdout Set):")
    sample_size = min(30, len(test_results))
    random_sample = test_results.sample(n=sample_size, random_state=24) # Fixed seed to keep inspection clean
    print(random_sample.to_string(index=False))

    # Serialize trained artifact matrix to disk
    joblib.dump(model, models_dir / "price_model.joblib")
    print(f"\n[OK] Upgraded Peer-Aware Valuation Brain model successfully saved at: models/price_model.joblib")

if __name__ == "__main__":
    train_peer_aware_engine()