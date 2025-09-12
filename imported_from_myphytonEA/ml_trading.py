import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import os
import datetime

class MLTradingModel:
	"""
	MLTradingModel supports multi-timeframe analysis. To use, structure your data with columns like:
		open_1m, high_1m, low_1m, close_1m, volume_1m,
		open_5m, high_5m, ..., close_5m, volume_5m,
		open_1d, high_1d, ..., close_1d, volume_1d,
	and optionally add indicators for each timeframe (e.g., ma_5_1m, rsi_14_5m, etc.).
	The model will automatically process all such features.
	"""
	def __init__(self, model_path='ml_model.pkl'):
		self.model_path = model_path
		self.model = None
		self.features = None
		self.last_trained = None

	def load_data(self, csv_path):
		df = pd.read_csv(csv_path)
		# Multi-timeframe feature engineering
		# Use these timeframes: 1d, 4h, 1h, 15m, 5m, 1m
		timeframes = ['1d', '4h', '1h', '15m', '5m', '1m']
		for tf in timeframes:
			close_col = f'close_{tf}'
			if close_col in df.columns:
				df[f'ma_5_{tf}'] = df[close_col].rolling(window=5).mean()
				df[f'ma_20_{tf}'] = df[close_col].rolling(window=20).mean()
				df[f'rsi_14_{tf}'] = self.compute_rsi(df[close_col], 14)
				macd, macd_signal = self.compute_macd(df[close_col])
				df[f'macd_{tf}'] = macd
				df[f'macd_signal_{tf}'] = macd_signal
				df[f'return_{tf}'] = df[close_col].pct_change()
				df[f'volatility_{tf}'] = df[close_col].rolling(window=10).std()
		# Entry logic: entries are made on 5m/1m, whichever timeframe the setup is cleaner.
		# This can be handled in the scanner/EA logic by checking ML signals for both 5m and 1m and choosing the best setup.
		# Backward compatibility: single timeframe columns
		if 'close' in df.columns:
			df['ma_5'] = df['close'].rolling(window=5).mean()
			df['ma_20'] = df['close'].rolling(window=20).mean()
			df['rsi_14'] = self.compute_rsi(df['close'], 14)
			df['macd'], df['macd_signal'] = self.compute_macd(df['close'])
			df['return'] = df['close'].pct_change()
			df['volatility'] = df['close'].rolling(window=10).std()
		# Placeholder for sentiment analysis feature (to be filled with real data)
		df['sentiment'] = 0.0
		df = df.dropna()
		return df

	def compute_rsi(self, series, period=14):
		delta = series.diff()
		gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
		rs = gain / loss
		rsi = 100 - (100 / (1 + rs))
		return rsi

	def compute_macd(self, series, fast=12, slow=26, signal=9):
		exp1 = series.ewm(span=fast, adjust=False).mean()
		exp2 = series.ewm(span=slow, adjust=False).mean()
		macd = exp1 - exp2
		macd_signal = macd.ewm(span=signal, adjust=False).mean()
		return macd, macd_signal

	def prepare_features(self, df, target_col='signal'):
		# Select features and target
		feature_cols = [col for col in df.columns if col not in [target_col, 'date', 'symbol']]
		X = df[feature_cols]
		y = df[target_col]
		self.features = feature_cols
		return X, y

	def train(self, csv_path, target_col='signal', tune_hyperparams=True):
		df = self.load_data(csv_path)
		X, y = self.prepare_features(df, target_col)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		if tune_hyperparams:
			param_grid = {
				'n_estimators': [100, 200],
				'max_depth': [None, 10, 20],
				'min_samples_split': [2, 5]
			}
			grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
			grid.fit(X_train, y_train)
			self.model = grid.best_estimator_
			print(f"Best params: {grid.best_params_}")
		else:
			self.model = RandomForestClassifier(n_estimators=100, random_state=42)
			self.model.fit(X_train, y_train)
		y_pred = self.model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		print(f"Validation Accuracy: {acc:.3f}")
		print(classification_report(y_test, y_pred))
		self.plot_performance(y_test, y_pred)
		self.last_trained = datetime.datetime.now()
		self.save()
		return acc

	def plot_performance(self, y_true, y_pred):
		# Confusion matrix
		cm = confusion_matrix(y_true, y_pred)
		plt.figure(figsize=(4,4))
		plt.imshow(cm, cmap='Blues')
		plt.title('Confusion Matrix')
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.colorbar()
		plt.show()
		# ROC curve (binary only)
		if len(np.unique(y_true)) == 2:
			y_score = self.model.predict_proba(np.array(y_true).reshape(-1,1))[:,1] if hasattr(self.model, 'predict_proba') else None
			if y_score is not None:
				fpr, tpr, _ = roc_curve(y_true, y_score)
				plt.figure()
				plt.plot(fpr, tpr, label='ROC curve')
				plt.plot([0,1],[0,1],'k--')
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('ROC Curve')
				plt.legend()
				plt.show()

	def predict(self, X):
		if self.model is None:
			self.load()
		# Support for incremental learning: if model supports partial_fit, use it
		if hasattr(self.model, 'partial_fit'):
			self.model.partial_fit(X, np.zeros(X.shape[0]))
		return self.model.predict(X)

	def save(self):
		if self.model is not None:
			joblib.dump({'model': self.model, 'features': self.features, 'last_trained': self.last_trained}, self.model_path)

	def load(self):
		if os.path.exists(self.model_path):
			data = joblib.load(self.model_path)
			self.model = data['model']
			self.features = data['features']
			self.last_trained = data.get('last_trained', None)

	def retrain_if_needed(self, csv_path, retrain_interval_days=7, target_col='signal'):
		# Automated retraining if last trained is older than interval
		now = datetime.datetime.now()
		if self.last_trained is None or (now - self.last_trained).days >= retrain_interval_days:
			print("Retraining model...")
			self.train(csv_path, target_col)
		else:
			print("Model is up to date. No retraining needed.")

if __name__ == "__main__":
	# Example usage
	ml = MLTradingModel()
	# Replace 'your_data.csv' with your actual data file
	acc = ml.train('your_data.csv', target_col='signal', tune_hyperparams=True)
	print(f"Model trained. Validation accuracy: {acc:.3f}")
	# Automated retraining example
	ml.retrain_if_needed('your_data.csv', retrain_interval_days=7, target_col='signal')
