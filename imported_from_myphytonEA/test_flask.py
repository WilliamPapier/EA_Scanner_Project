from flask import Flask
app = Flask(__name__)

@app.route('/stats')
def stats():
    print("=== /stats route called (MINIMAL) ===", flush=True)
    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)
