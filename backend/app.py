from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from Classifier import check_image, check_video  # make sure these exist
from database import init_db, save_detection, get_all_detections, get_detection_stats

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize database on startup
init_db()


@app.route("/health", methods=["GET"])
def health():
    """
    Simple health‑check endpoint for the Android app.
    Returns 200 if the Flask server is running.
    """
    return jsonify({"ok": True}), 200


@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    result = check_image(path)  # should return dict with label/score

    # ✅ Save to database
    save_detection(
        filename=filename,
        media_type=result.get("type", "unknown"),
        fake_score=result.get("fake_score", 0),
        suspicion_level=result.get("suspicion_level", "UNKNOWN"),
        source_type="image"
    )

    # Clean up
    try:
        os.remove(path)
    except:
        pass

    return jsonify({"success": True, **result}), 200


@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    result = check_video(path)  # should return dict with label/frame_score

    # ✅ Save to database
    save_detection(
        filename=filename,
        media_type="video",
        fake_score=result.get("fake_score", 0),
        suspicion_level=result.get("suspicion_level", "UNKNOWN"),
        source_type="video"
    )

    # Clean up
    try:
        os.remove(path)
    except:
        pass

    return jsonify({"success": True, **result}), 200


@app.route("/detection-history", methods=["GET"])
def detection_history():
    """Get detection history (last N detections)"""
    limit = request.args.get("limit", 50, type=int)
    detections = get_all_detections(limit=limit)

    return jsonify({
        "success": True,
        "detections": [
            {
                "id": d[0],
                "filename": d[1],
                "type": d[2],
                "fake_score": d[3],
                "suspicion_level": d[4],
                "timestamp": d[5]
            }
            for d in detections
        ]
    }), 200


@app.route("/detection-stats", methods=["GET"])
def detection_stats():
    """Get detection statistics"""
    stats = get_detection_stats()
    return jsonify({
        "success": True,
        **stats
    }), 200


if __name__ == "__main__":
    # this is what we run, not typed into PowerShell directly
    app.run(host="0.0.0.0", port=5000, debug=False)
