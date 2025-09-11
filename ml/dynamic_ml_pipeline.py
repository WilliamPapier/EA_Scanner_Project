import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================
# 1️⃣ Load Scanner Data
# =========================

def load_scanner_data(path="scanner_output_all_tfs.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna()
    return df

# =========================
# 2️⃣ Feature & Label Builder
# =========================

def build_features_labels(df):
    features = [
        'liq_sweep','bos_up','bos_down',
        'fvg_bull','fvg_bear',
        'bull_ob','bear_ob',
        'atr','body','wick_top','wick_bottom','body_to_range'
    ]
    df['label'] = (df['outcome'] == 'win').astype(int)
    return df[features], df['label'], features

# =========================
# 3️⃣ Train New Model
# =========================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Weekly Retrain Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# =========================
# 4️⃣ Save Model with Timestamp
# =========================

def save_model(model, features, folder="models"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"{folder}/scanner_filter_model_{timestamp}.pkl"
    feat_path = f"{folder}/features_{timestamp}.csv"
    
    joblib.dump(model, model_path)
    pd.Series(features).to_csv(feat_path, index=False)
    
    print(f"✅ Model saved: {model_path}")
    return model_path

# =========================
# 5️⃣ Run Weekly Retrain
# =========================

def weekly_retrain(scanner_csv="scanner_output_all_tfs.csv"):
    df = load_scanner_data(scanner_csv)
    X, y, features = build_features_labels(df)
    model = train_model(X, y)
    save_model(model, features)

# Example Run (trigger this weekly)
if __name__ == "__main__":
    weekly_retrain()