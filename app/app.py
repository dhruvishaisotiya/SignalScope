"""
SignalScope Web App (v2)
========================
Flask app implementing the required predict interface plus:
  Module A - explanation + heat-map
  Module B - generator attribution (heuristic)
  Module D - metadata / provenance evidence
  Module F - deployable drag-and-drop interface

Run:
    pip install -r ../requirements.txt
    python app.py
Then open http://localhost:5006
"""
import os
import sys
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import cv2

from predict import predict_proba, _default_threshold
from explain import render_heatmap_overlay

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "SIGNALSCOPE_MODEL",
    os.path.join(APP_DIR, "..", "model", "signalscope_model.joblib"),
)
UPLOAD_DIR = os.path.join(APP_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    uid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1].lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        ext = ".png"
    # keep the ORIGINAL bytes so Module D can read real EXIF / PNG chunks
    raw_path = os.path.join(UPLOAD_DIR, f"{uid}_input{ext}")
    file.save(raw_path)

    img = cv2.imread(raw_path)
    if img is None:
        return jsonify({"error": "Could not decode image. Please upload a JPG/PNG."}), 400

    try:
        result = predict_proba(raw_path, model_path=MODEL_PATH,
                               threshold=_default_threshold(),
                               with_explanation=True)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    heat_vis = result.pop("_heat_vis", None)
    heatmap_url = None
    if heat_vis is not None:
        heatmap_filename = f"{uid}_heatmap.png"
        render_heatmap_overlay(img, heat_vis, os.path.join(UPLOAD_DIR, heatmap_filename))
        heatmap_url = f"/static/uploads/{heatmap_filename}"

    is_ai = result["label"] == "AI-generated"
    response = {
        "label": "AI Generated" if is_ai else "Not AI Generated",
        "is_ai_generated": bool(is_ai),
        "confidence": result["confidence"],
        "probability_ai_generated": result["p_ai"],
        "threshold": result["threshold"],
        "explanation": result.get("explanation", ""),
        "cue_ranking": result.get("cue_ranking", []),
        "evidence": result["evidence"],
        "generator_attribution": result.get("generator_attribution"),
        "heatmap_url": heatmap_url,
        "input_image_url": f"/static/uploads/{os.path.basename(raw_path)}",
        "disclaimer": result["note"],
    }
    return jsonify(response)


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=False)
