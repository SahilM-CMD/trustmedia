from flask import Flask, request, jsonify
from flask_cors import CORS
from Classifier import check_image, check_video
import os
import tempfile

app = Flask(__name__)
CORS(app)

@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]

    # save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = check_image(tmp_path)
    finally:
        os.remove(tmp_path)

    return jsonify(result)

@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = check_video(tmp_path)
    finally:
        os.remove(tmp_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
