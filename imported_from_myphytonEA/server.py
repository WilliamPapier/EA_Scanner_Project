print("=== UNIQUE TEST: server.py running from:", __file__, flush=True)
print('=== server.py module loaded (outside try) ===', flush=True)
try:
    print('--- server.py: Starting up ---')
    from flask import Flask, request, jsonify, send_file
    import io
    import time
    from datetime import datetime
    import os
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from joblib import dump, load
    from ml_modules import feature_engineering, ml_decision_engine, setup_storage, dynamic_targets, risk_manager, strategy_selector
    print('Imported feature_engineering, ml_decision_engine, setup_storage, dynamic_targets, risk_manager, strategy_selector')
    from ml_modules import mean_reversion, breakout, news_trading
    print('Imported mean_reversion, breakout, news_trading')
    from ml_modules import rl_agent, anomaly_detection, meta_learning
    print('Imported rl_agent, anomaly_detection, meta_learning')

    app = Flask(__name__)
    print('--- Flask app initialized ---')

    print('--- REGISTERING /stats ROUTE AT TOP ---', flush=True)
    @app.route('/stats', methods=['GET'])
    def stats():
        print('=== /stats route called (TOP) ===', flush=True)
        return 'OK', 200
    print('--- /stats ROUTE FUNCTION OBJECT:', stats, flush=True)

    trade_data = []
    setup_data = []
    ml_data = {}
    log_entries = []
    notifications = []
    risk_metrics = {}
    feedback_store = []

    class Logger:
        @staticmethod
        def log(category, message, level='info'):
            log_entries.append({'category': category, 'message': message, 'level': level, 'timestamp': datetime.utcnow().isoformat()})
    print('Logger class defined')

    class TradeManager:
        @staticmethod
        def is_duplicate(setup):
            for s in setup_data:
                if s.get('instrument') == setup.get('instrument') and s.get('timestamp') == setup.get('timestamp'):
                    return True
            return False
        @staticmethod
        def add_trade(trade):
            trade_data.append(trade)
    print('TradeManager class defined')

    import ml_modules.risk_manager as risk_manager_mod
    print('Imported ml_modules.risk_manager as risk_manager_mod')
    if not hasattr(risk_manager_mod.RiskManager, 'check_risk'):
        @staticmethod
        def check_risk(setup):
            return True
        risk_manager_mod.RiskManager.check_risk = check_risk
    print('RiskManager patched if needed')

    import traceback
    @app.errorhandler(Exception)
    def handle_exception(e):
        print('--- Global error handler triggered ---', flush=True)
        print('Exception:', repr(e), flush=True)
        traceback.print_exc()
        return 'Internal Server Error', 500

    @app.route('/explain/shap', methods=['POST'])
    def explain_shap():
        from ml_modules import explainability
        data = request.get_json()
        return jsonify({'status': 'ok', 'explanation': 'shap'}), 200

    @app.route('/explain/lime', methods=['POST'])
    def explain_lime():
        from ml_modules import explainability
        data = request.get_json()
        return jsonify({'status': 'ok', 'explanation': 'lime'}), 200

    @app.route('/scanner', methods=['POST'])
    def scanner():
        data = request.get_json()
        print('--- /scanner route hit ---', flush=True)
        setup_data.append(data)
        return jsonify({'status': 'ok', 'received': data}), 200

    @app.route('/ml', methods=['GET'])
    def ml_stats():
        return jsonify({'ml_stats': ml_data})

    @app.route('/ea', methods=['POST'])
    def ea_feedback():
        data = request.get_json()
        feedback_store.append(data)
        return jsonify({'status': 'ok', 'feedback': data}), 200

    @app.route('/trade_signal', methods=['GET'])
    def trade_signal():
        if setup_data:
            next_setup = setup_data.pop(0)
            if 'instrument' in next_setup:
                next_setup['symbol'] = next_setup['instrument']
            Logger.log('signal', f"Trade signal sent: {next_setup}")
            return jsonify({'status': 'ok', 'signal': next_setup})
        else:
            return jsonify({'status': 'empty'})

    @app.route('/log', methods=['GET'])
    def get_logs():
        return jsonify({'logs': log_entries[-200:]})

    @app.route('/notify', methods=['POST'])
    def manual_notify():
        data = request.json
        if isinstance(data, dict) and 'message' in data:
            notifications.append({'message': data['message'], 'type': data.get('type', 'info')})
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    print('Registered routes:')
    for rule in app.url_map.iter_rules():
        print(rule)
    print('--- END OF TRY BLOCK, BEFORE __main__ ---', flush=True)
    if __name__ == "__main__":
        print('--- __main__ block: Starting Flask server ---')
        app.run(debug=True, use_reloader=False)
except Exception as e:
    import traceback
    print('--- TOP-LEVEL EXCEPTION DURING STARTUP OR REQUEST HANDLING ---')
    print(e)
    traceback.print_exc()
