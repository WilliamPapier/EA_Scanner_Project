import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# CONFIG
SETUPS_CSV = "historical_setups_logged.csv"
MODEL_OUT = "ml_trading_model.joblib"

# 1. Load historical setups
print("Loading setups from:", SETUPS_CSV)

# Read CSV with semicolon delimiter

try:
	df = pd.read_csv(SETUPS_CSV, delimiter=',', encoding='utf-8-sig')
except Exception as e:
	print(f"Error reading {SETUPS_CSV}: {e}")
	exit(1)

# Print columns for user reference
print("Columns in CSV:", list(df.columns))

# Use correct label column
label_col = None
for col in ["is_Buy", "is_buy", "label", "signal"]:
	if col in df.columns:
		label_col = col
		break
if label_col is None:
	print("No label column found (tried: is_Buy, is_buy, label, signal). Available columns:", list(df.columns))
	exit(1)


# Clean label column: drop rows with missing or invalid values
if df[label_col].dtype == object:
	df[label_col] = df[label_col].str.lower().map({'buy': 1, 'sell': 0, '1': 1, '0': 0})
else:
	df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
df = df[df[label_col].isin([0, 1])].copy()
print(f"Rows with valid labels: {len(df)}")
if len(df) == 0:
	print("No valid rows with label 0 or 1 found. Check your data and scanner output.")
	exit(1)
y = df[label_col].astype(int)

# 2. Feature selection (customize as needed)


# Explicitly use these columns as features if present
possible_features = ["Entry_price", "SL_price", "tp_price", "Profit_Loss", "Confidence"]
feature_cols = [col for col in possible_features if col in df.columns]
if not feature_cols:
	print("No valid feature columns found. Available columns:", list(df.columns))
	exit(1)
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
print("Training GradientBoostingClassifier...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(model, MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
