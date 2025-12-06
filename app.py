from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import cv2
import os

app = Flask(__name__)

# CẤU HÌNH
WIDTH = 96
HEIGHT = 96
IMG_SIZE = (WIDTH, HEIGHT)
VOCAB_SIZE = 800

# LOAD MODELS
base_path = "model/"

try:
    svm_model = joblib.load(os.path.join(base_path, "svm_garbage_model.pkl"))
    pca_model = joblib.load(os.path.join(base_path, "pca_garbage_model.pkl"))
    le_model = joblib.load(os.path.join(base_path, "le_garbage_model.pkl"))
    kmeans_model = joblib.load(os.path.join(base_path, "kmeans_garbage_model.pkl"))
    scaler_model = joblib.load(os.path.join(base_path, "scaler_garbage_model.pkl"))
except Exception:
    pass

orb = cv2.ORB_create(nfeatures=500)


def extract_features_bow(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    bow_hist = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if des is not None:
        preds = kmeans_model.predict(des.astype(float))
        unique, counts = np.unique(preds, return_counts=True)
        for u, c in zip(unique, counts):
            if u < VOCAB_SIZE:
                bow_hist[u] = c

    bow_hist = cv2.normalize(bow_hist.reshape(1, -1), None, norm_type=cv2.NORM_L2).flatten()

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    color_feat = np.concatenate([h_hist, s_hist, v_hist])
    color_feat = cv2.normalize(color_feat.reshape(1, -1), None, norm_type=cv2.NORM_L2).flatten()

    return np.concatenate([bow_hist, color_feat])


def preprocess_image(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Error")

    feat = extract_features_bow(img)
    feat = feat.reshape(1, -1)
    feat_scaled = scaler_model.transform(feat)

    feat_pca = pca_model.transform(feat_scaled)
    return feat_pca


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        X_final = preprocess_image(file)
        y_pred = svm_model.predict(X_final)
        class_name = le_model.inverse_transform(y_pred)[0]

        return jsonify({"prediction": class_name})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run()
