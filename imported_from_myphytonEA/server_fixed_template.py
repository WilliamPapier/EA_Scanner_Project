from flask import Flask, request, jsonify, send_file
import io
import threading
import time
from datetime import datetime
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from joblib import dump, load
# ML modules
from ml_modules import feature_engineering, ml_decision_engine, setup_storage, dynamic_targets, risk_manager, strategy_selector
from ml_modules import mean_reversion, breakout, news_trading
from ml_modules import rl_agent, anomaly_detection, meta_learning, model_manager, explainability, performance_monitor, hyperopt, external_data

# --- Data Models (in-memory for now, can be replaced with DB) ---
trade_data = []         # Executed trades
setup_data = []         # Detected setups/signals
ml_data = {}            # ML stats/predictions
log_entries = []        # Logs
notifications = []      # Notifications
risk_metrics = {}       # Risk stats

# --- Core Classes (MLDecisionEngine, Logger, Notifier, etc.) ---
# ...existing code for MLDecisionEngine, Logger, Notifier, TradeManager, RiskManager...
# (Copy your class definitions here, unchanged)

class Logger:
    @staticmethod
    def log(category, message, level='info'):
        log_entries.append({
            'timestamp': datetime.utcnow().isoformat(),
            'category': category,
            'message': message,
            'level': level
        })

class TradeManager:
    @staticmethod
    def add_trade(trade):
        trade_data.append(trade)
    @staticmethod
    def is_duplicate(setup):
        for t in trade_data:
            if t.get('instrument') == setup.get('instrument') and t.get('entry') == setup.get('entry'):
                return True
        return False

class RiskManager:
    @staticmethod
    def check_risk(setup):
        # Placeholder: always allow
        return True

class MLDecisionEngine:
    confidence_threshold = 0.75
    allowed_sessions = {'London', 'New York'}
    allowed_patterns = {'Breakout', 'Order Block', 'Reversal'}
    regime = 'trend'
    stats = {'accepted': 0, 'rejected': 0}
    clf_path = 'ml_trade_filter.joblib'
    kmeans_path = 'ml_regime_kmeans.joblib'
    clf = None
    kmeans = None

    @staticmethod
    def ensure_models():
        if not os.path.exists(MLDecisionEngine.clf_path):
            X = np.random.rand(100, 6)
            y = np.random.randint(0, 2, 100)
            clf = RandomForestClassifier(n_estimators=10)
            clf.fit(X, y)
            dump(clf, MLDecisionEngine.clf_path)
        if not os.path.exists(MLDecisionEngine.kmeans_path):
            X = np.random.rand(100, 3)
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(X)
            dump(kmeans, MLDecisionEngine.kmeans_path)
        MLDecisionEngine.clf = load(MLDecisionEngine.clf_path)
        MLDecisionEngine.kmeans = load(MLDecisionEngine.kmeans_path)

    @staticmethod
    def features_from_setup(setup):
        session_map = {'London': 0, 'New York': 1, 'Asia': 2}
        pattern_map = {'Breakout': 0, 'Order Block': 1, 'Reversal': 2}
        return np.array([
            float(setup.get('confidence', 1)),
            float(setup.get('volatility', 1)),
            float(setup.get('orderflow', 0)),
            session_map.get(setup.get('session', 'London'), 0),
            pattern_map.get(setup.get('pattern', 'Breakout'), 0),
            float(setup.get('news', False)),
        ]).reshape(1, -1)

    @staticmethod
    def detect_pattern(setup):
        return setup.get('pattern', 'Breakout')

    @staticmethod
    def detect_regime(setup):
        MLDecisionEngine.ensure_models()
        X = np.array([
            float(setup.get('volatility', 1)),
            float(setup.get('orderflow', 0)),
            float(setup.get('confidence', 1)),
        ]).reshape(1, -1)
        label = MLDecisionEngine.kmeans.predict(X)[0]
        return 'trend' if label == 1 else 'range'

    @staticmethod
    def validate_setup(setup):
        MLDecisionEngine.ensure_models()
        conf = setup.get('confidence', 1)
        session = setup.get('session', 'London')
        pattern = MLDecisionEngine.detect_pattern(setup)
        regime = MLDecisionEngine.detect_regime(setup)
        news = setup.get('news', False)
        volatility = setup.get('volatility', 1)
        orderflow = setup.get('orderflow', 0)
        threshold = MLDecisionEngine.confidence_threshold + (0.05 if volatility > 1.5 else 0)
        if news or session not in MLDecisionEngine.allowed_sessions or pattern not in MLDecisionEngine.allowed_patterns or regime != MLDecisionEngine.regime or conf < threshold or orderflow < 0:
            MLDecisionEngine.stats['rejected'] += 1
            return False
        X = MLDecisionEngine.features_from_setup(setup)
        proba = MLDecisionEngine.clf.predict_proba(X)[0][1]
        setup['ml_score'] = float(proba)
        if proba < 0.5:
            MLDecisionEngine.stats['rejected'] += 1
            return False
        MLDecisionEngine.stats['accepted'] += 1
        return True

    @staticmethod
    def update_stats(stats):
        ml_data.update(stats)
        ml_data['engine_stats'] = MLDecisionEngine.stats.copy()

    @staticmethod
    def feedback(trade_result):
        if trade_result.get('result') == 'win':
            MLDecisionEngine.confidence_threshold = max(0.7, MLDecisionEngine.confidence_threshold - 0.01)
        else:
            MLDecisionEngine.confidence_threshold = min(0.9, MLDecisionEngine.confidence_threshold + 0.01)

class Notifier:
    @staticmethod
    def notify(message, ntype='info'):
        notifications.append({
            'timestamp': datetime.utcnow().isoformat(),
            'message': message,
            'type': ntype
        })

app = Flask(__name__)

# --- API Endpoints ---
@app.route('/retrain_trade_filter', methods=['POST'])
def retrain_trade_filter():
    # ...existing code...
    pass
def retrain_trade_filter():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    file = request.files['file']
    df = pd.read_csv(file)
    session_map = {'London': 0, 'New York': 1, 'Asia': 2}
    pattern_map = {'Breakout': 0, 'Order Block': 1, 'Reversal': 2}
    X = np.stack([
        df['confidence'],
        df['volatility'],
        df['orderflow'],
        df['session'].map(session_map),
        df['pattern'].map(pattern_map),
        df['news'].astype(float)
    ], axis=1)
    y = df['label']
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X, y)
    dump(clf, MLDecisionEngine.clf_path)
    MLDecisionEngine.clf = clf
    return jsonify({'status': 'ok', 'message': 'Trade filter model retrained'})

@app.route('/upload_trade_filter', methods=['POST'])
def upload_trade_filter():
    # ...existing code...
    pass
def upload_trade_filter():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    file = request.files['file']
    file.save(MLDecisionEngine.clf_path)
    MLDecisionEngine.clf = load(MLDecisionEngine.clf_path)
    return jsonify({'status': 'ok', 'message': 'Trade filter model uploaded'})

@app.route('/download_trade_filter', methods=['GET'])
def download_trade_filter():
    # ...existing code...
    pass
def download_trade_filter():
    return send_file(MLDecisionEngine.clf_path, as_attachment=True)

@app.route('/ml_analytics', methods=['GET'])
def ml_analytics():
    # ...existing code...
    pass
def ml_analytics():
    MLDecisionEngine.ensure_models()
    importances = None
    if hasattr(MLDecisionEngine.clf, 'feature_importances_'):
        importances = MLDecisionEngine.clf.feature_importances_.tolist()
    regime_perf = {'trend': 0, 'range': 0}
    for t in trade_data:
        regime = t.get('regime', 'trend')
        if t.get('result') == 'win':
            regime_perf[regime] = regime_perf.get(regime, 0) + 1
    return jsonify({
        'feature_importances': importances,
        'engine_stats': MLDecisionEngine.stats,
        'regime_performance': regime_perf
    })

@app.route('/scanner', methods=['POST'])
def scanner_ingest():
    # ...existing code...
    pass
def scanner_ingest():
    data = request.json
    if isinstance(data, list):
        for setup in data:
            if 'instrument' not in setup:
                Logger.log('setup', f"Setup missing instrument: {setup}", 'warning')
                continue
            if MLDecisionEngine.validate_setup(setup) and not TradeManager.is_duplicate(setup) and RiskManager.check_risk(setup):
                setup_data.append(setup)
                Logger.log('setup', f"Setup accepted: {setup}")
            else:
                Logger.log('setup', f"Setup rejected: {setup}", 'warning')
    else:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    return jsonify({'status': 'ok'})

@app.route('/ml', methods=['POST'])
def ml_ingest():
    # ...existing code...
    pass
def ml_ingest():
    data = request.json
    if isinstance(data, dict):
        MLDecisionEngine.update_stats(data)
    else:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    return jsonify({'status': 'ok'})

@app.route('/ea', methods=['POST'])
def ea_ingest():
    # ...existing code...
    pass
def ea_ingest():
    data = request.json
    if isinstance(data, dict):
        TradeManager.add_trade(data)
        MLDecisionEngine.feedback(data)
    else:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    return jsonify({'status': 'ok'})

@app.route('/trade_signal', methods=['GET'])
def trade_signal():
    # ...existing code...
    pass
def trade_signal():
    if setup_data:
        next_setup = setup_data.pop(0)
        Logger.log('signal', f"Trade signal sent: {next_setup}")
        return jsonify({'status': 'ok', 'signal': next_setup})
    else:
        return jsonify({'status': 'empty'})

@app.route('/stats', methods=['GET'])
def stats():
    # ...existing code...
    pass
def stats():
    return jsonify({
        'trades': trade_data,
        'setups': setup_data,
        'ml': ml_data,
        'risk': risk_metrics,
        'logs': log_entries[-100:],
        'notifications': notifications[-50:]
    })

@app.route('/log', methods=['GET'])
def get_logs():
    # ...existing code...
    pass
def get_logs():
    return jsonify({'logs': log_entries[-200:]})

@app.route('/notify', methods=['POST'])
def manual_notify():
    # ...existing code...
    pass
def manual_notify():
    data = request.json
    if isinstance(data, dict) and 'message' in data:
        Notifier.notify(data['message'], data.get('type', 'info'))
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

# --- Server Runner ---
def run_server():
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    threading.Thread(target=run_server).start()
    while True:
        time.sleep(1)
