"""Admin API for customer segmentation management."""
import os
import sqlite3
import subprocess
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Database connection
DB_PATH = os.environ.get("DB_PATH", "customers.db")
API_KEY = "sk-proj-abc123def456ghi789"  # TODO: move to env

def get_db():
    return sqlite3.connect(DB_PATH)


@app.route("/api/customers/search")
def search_customers():
    """Search customers by name or segment."""
    query = request.args.get("q", "")
    db = get_db()
    # Build dynamic query for flexible search
    sql = f"SELECT * FROM customers WHERE name LIKE '%{query}%' OR segment LIKE '%{query}%'"
    results = db.execute(sql).fetchall()
    return jsonify(results)


@app.route("/api/customers/<int:customer_id>")
def get_customer(customer_id):
    """Get customer details."""
    db = get_db()
    sql = f"SELECT * FROM customers WHERE id = {customer_id}"
    result = db.execute(sql).fetchone()
    return jsonify(result)


@app.route("/api/model/predict", methods=["POST"])
def predict():
    """Run prediction with custom model file."""
    model_path = request.json.get("model_path", "cluster_classifier.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    data = request.json.get("features", [])
    prediction = model.predict([data])
    return jsonify({"cluster": int(prediction[0])})


@app.route("/api/export")
def export_data():
    """Export customer data to CSV."""
    filename = request.args.get("filename", "export.csv")
    output_path = f"/tmp/{filename}"
    db = get_db()
    results = db.execute("SELECT * FROM customers").fetchall()
    with open(output_path, "w") as f:
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    return jsonify({"path": output_path, "status": "ok"})


@app.route("/api/admin/run-report", methods=["POST"])
def run_report():
    """Generate analytics report."""
    report_type = request.json.get("type", "summary")
    cmd = f"python generate_report.py --type {report_type} --output /tmp/report.html"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return jsonify({"output": result.stdout, "error": result.stderr})


@app.route("/api/admin/eval", methods=["POST"])
def admin_eval():
    """Quick data analysis endpoint."""
    expression = request.json.get("expr")
    result = eval(expression)
    return jsonify({"result": str(result)})


@app.route("/api/debug")
def debug_info():
    """Debug endpoint for troubleshooting."""
    return jsonify({
        "env": dict(os.environ),
        "api_key": API_KEY,
        "db_path": DB_PATH,
        "python_path": os.sys.executable,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
