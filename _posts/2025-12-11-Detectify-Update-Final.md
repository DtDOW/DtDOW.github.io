---
title: Detectify Update Final
date: 2025-12-11
categories: [project, closed]
tags: [project, closed, technology, detectify]
---
## Detectify Final Version 

This is the final version including front-end, back-end, AI/ML model. We have used RandomForest for training our model ( it gave us an over all acuracy of ~ 75%, with heighst upto ~97% (we trained on multiple datasets for comparisions)). 

So why did we use random forest ? We used random forest caue we tried running CNN on T4 GPU ( free one on google colab ) and it was quite slow. Thus we tried something that was less heavy and still give us good accuracy. Plus this also proves that classical ML model can make reliable assumptions if trained well. I HAVENT ATTATCHED THE PCA FILES, YOU WILL HAVE TO CREATE THEM YOURSELF.

Here is a link to a working version : ( NOTE : the version may perform poor overtime given the advancement in AI image genration) 

https://detectify-1tby.onrender.com

![Results](https://i.ibb.co/Gfb69gBg/Screenshot-2025-12-11-at-13-01-05.png)

## Detectify ML Model : 
```python 
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


df = pd.read_csv('/content/drive/MyDrive/Python/Dataset/video_features3.csv')

X = df.drop(columns=["video_id", "label"])
y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1023, stratify=y
)


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
print("=== Random Forest Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#joblib.dump(pca, "pca_50.joblib")
#joblib.dump(rf, "rf_pca50_model.joblib")

print("Saved PCA and model successfully!")
```
## Detectify Flask : 

```python 
import os
import uuid
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from models import db, User
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from detect import predict_file

# CONFIG
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".gif", ".jpeg", ".jpg", ".png"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "change-this-secret"

# INIT DB
db.init_app(app)
with app.app_context():
    db.create_all()

# LOGIN MANAGER
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def allowed_file(filename):
    if not filename:
        return False
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


# ---- ROUTES ----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_and_predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    f = request.files["file"]

    if f.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(f.filename):
        return jsonify({"success": False, "error": "File type not allowed"}), 400

    filename = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    f.save(save_path)

    try:
        label, prob = predict_file(save_path)

        # SAVE HISTORY IF LOGGED IN
        if current_user.is_authenticated:
            record = Prediction(
                user_id=current_user.id,
                filename=filename,
                label=label,
                confidence=prob
            )
            db.session.add(record)
            db.session.commit()

        return jsonify({
            "success": True,
            "label": label,
            "confidence": round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/history")
@login_required
def history():
    logs = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", logs=logs)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").lower()
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

        login_user(user)
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email").lower()
        password = request.form.get("password")

        if User.query.filter_by(email=email).first():
            flash("User already exists!", "error")
            return redirect(url_for("signup"))

        hashed_pw = generate_password_hash(password)
        user = User(email=email, password=hashed_pw)
        db.session.add(user)
        db.session.commit()

        flash("Signup successful — login now", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/ping")
def ping():
    return "pong", 200


if __name__ == "__main__":
    app.run(debug=True)

``` 
## Detectify Model : 

```python 

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<User {self.email}>"
```

## Detectify Detect : 
(Links to trained Model, Contains Joblib File)

```python
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib

BASE_DIR = os.path.dirname(__file__)

PCA_PATH = os.path.join(BASE_DIR, "models", "pca_50Best.joblib")
RF_PATH = os.path.join(BASE_DIR, "models", "rf_pca50_modelBest.joblib")

# Load Models
print("Loading PCA and RF models...")
_pca = joblib.load(PCA_PATH)
_rf = joblib.load(RF_PATH)
print("Models loaded.")

facem = mp.solutions.face_mesh

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def extract_landmark_features(image):
    """Extract facial landmarks from a single image."""
    H, W = image.shape[:2]
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    with facem.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False
    ) as fm:
        out = fm.process(rgb)

        if not out.multi_face_landmarks:
            return None

        pts = np.array([[p.x * W, p.y * H] for p in out.multi_face_landmarks[0].landmark])
        mean = pts.mean(axis=0)
        std = pts.std(axis=0) + 1e-6
        norm = (pts - mean) / std

        return norm.flatten()


def extract_video_features(path):
    """Extract frame-level facial landmarks."""
    video = cv.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError("Unable to open video")

    feats = []
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    stride = max(total_frames // 30, 1)  # sample ~30 frames

    idx = 0
    with facem.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False
    ) as fm:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if idx % stride != 0:
                idx += 1
                continue

            H, W = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            out = fm.process(rgb)

            if out.multi_face_landmarks:
                pts = np.array([[p.x * W, p.y * H] for p in out.multi_face_landmarks[0].landmark])
                mean = pts.mean(axis=0)
                std = pts.std(axis=0) + 1e-6
                feats.append(((pts - mean) / std).flatten())

            idx += 1

    video.release()
    if not feats:
        return None

    feats = np.stack(feats)
    return feats


def predict_file(path):
    """Auto-detect if image or video and run prediction."""
    ext = os.path.splitext(path)[1].lower()

    if ext in IMAGE_EXTS:
        img = cv.imread(path)
        feat = extract_landmark_features(img)
        if feat is None:
            raise RuntimeError("No face detected.")
        feats = np.array([feat])

    else:
        feats = extract_video_features(path)
        if feats is None:
            raise RuntimeError("No face detected in video.")

    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    features = np.concatenate([mean, std]).reshape(1, -1)

    x = _pca.transform(features)
    pred = _rf.predict(x)[0]
    prob = _rf.predict_proba(x)[0][pred]

    label = "REAL" if pred == 1 else "DEEPFAKE"
    return label, float(prob)

```

## Index HTML : 

```python
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Detectify — Deepfake Detector</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg-grad-start: #0f274c;
      --bg-grad-end: #1b3a74;
      --card-bg: rgba(255,255,255,0.04);
      --card-glow: rgba(19,37,84,0.6);
      --accent-1: #5eead4;
      --accent-2: #7c3aed;
      --muted: rgba(255,255,255,0.75);
    }

    html,body{
      height:100%;
      margin:0;
      font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      color: white;
      -webkit-font-smoothing:antialiased;
      -moz-osx-font-smoothing:grayscale;
    }

    /* full-screen header with background image and gradient overlay */
    .page-bg{
      min-height:100vh;
      background-image:
        linear-gradient(180deg, rgba(6,18,49,0.75) 0%, rgba(8,18,44,0.88) 60%),
        url("/mnt/data/WhatsApp Image 2025-11-20 at 16.57.49.jpeg");
      background-size: cover;
      background-position: center;
      display:flex;
      flex-direction:column;
    }

    header{
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:28px 48px;
      border-bottom: 1px solid rgba(255,255,255,0.03);
      backdrop-filter: blur(4px);
    }

    .brand {
      display:flex;
      gap:16px;
      align-items:center;
    }
    .logo {
      width:44px;
      height:44px;
      border-radius:8px;
      background: linear-gradient(135deg,var(--accent-1),var(--accent-2));
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:800;
      color:#04202a;
      box-shadow: 0 6px 18px rgba(12,18,32,0.35);
    }
    .brand h1{
      margin:0;
      font-size:20px;
      letter-spacing:0.2px;
      font-weight:800;
    }
    .brand p{
      margin:0 0 0 6px;
      font-size:12px;
      color:rgba(255,255,255,0.6);
    }

    .signed-in {
      text-align:right;
      font-size:13px;
      color:rgba(255,255,255,0.85);
    }
    .signed-in small { display:block; color:rgba(255,255,255,0.5); font-size:11px; }

    /* center container */
    .main {
      width:100%;
      max-width:920px;
      margin: 60px auto;
      padding: 28px;
      display:flex;
      flex-direction:column;
      align-items:center;
      gap:36px;
    }

    .upload-card {
      width:100%;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.015));
      border-radius:18px;
      padding: 44px 36px;
      box-shadow: 0 18px 40px rgba(3,12,35,0.45);
      border: 1px solid rgba(255,255,255,0.03);
      backdrop-filter: blur(8px);
    }

    .upload-card h2{
      margin:0 0 6px 0;
      text-align:center;
      font-size:24px;
      letter-spacing:0.2px;
    }
    .upload-card p.lead{
      margin:0 0 24px 0;
      text-align:center;
      color:rgba(255,255,255,0.65);
      font-size:14px;
    }

    .dropzone {
      margin: 0 auto;
      width: 78%;
      min-height: 240px;
      border-radius:12px;
      border: 2px dashed rgba(255,255,255,0.12);
      display:flex;
      align-items:center;
      justify-content:center;
      flex-direction:column;
      color: rgba(255,255,255,0.6);
      position:relative;
      padding:18px;
      transition: all 180ms ease;
      background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
    }
    .dropzone.dragover {
      border-color: rgba(124,58,237,0.9);
      box-shadow: 0 10px 30px rgba(124,58,237,0.12), 0 2px 6px rgba(0,0,0,0.35) inset;
      transform: translateY(-4px);
    }
    .dropzone .hint { font-size:15px; margin-bottom:18px; }

    .btn-choose {
      display:inline-block;
      padding:10px 18px;
      border-radius:10px;
      cursor:pointer;
      font-weight:700;
      background: linear-gradient(90deg, #4f46e5, #06b6d4);
      border: none;
      color: white;
      box-shadow: 0 8px 20px rgba(8,16,40,0.45);
    }

    .small-note { margin-top:10px; font-size:13px; color:rgba(255,255,255,0.5); }

    /* bottom info card */
    .info-card {
      width:100%;
      border-radius:14px;
      padding:28px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      box-shadow: 0 12px 30px rgba(3,12,35,0.45);
      border:1px solid rgba(255,255,255,0.03);
    }
    .info-card h3 { margin:0 0 10px 0; font-size:20px; text-align:center; }
    .info-card p { margin:0; color:rgba(255,255,255,0.75); line-height:1.6; text-align:center; max-width:820px; margin-left:auto; margin-right:auto; }

    /* progress */
    #progressBar { width:78%; margin:12px auto 0 auto; height:14px; background: rgba(255,255,255,0.06); border-radius:8px; overflow:hidden; display:none; }
    #progress { height:100%; width:0%; transition: width 0.15s linear; background: linear-gradient(90deg,#60a5fa,#7c3aed); }

    /* responsive tweaks */
    @media (max-width:840px){
      .main { margin: 28px 18px; }
      .dropzone { width:100%; min-height:200px; }
      #progressBar { width:100%; }
      header { padding:18px; }
    }
  </style>
</head>
<body>
  <div class="page-bg">
    <header>
      <div class="brand">
        <div class="logo">π</div>
        <div>
          <h1>Deepfake Detector</h1>
          <p>AI-powered tool for detecting manipulated videos.</p>
        </div>
      </div>

      <div class="signed-in">
        {% if current_user.is_authenticated %}
          Signed in as
          <small>{{ current_user.email }}</small>
          <div style="margin-top:6px;"><a href="{{ url_for('logout') }}" style="color:rgba(255,255,255,0.8);font-size:12px;">Logout</a></div>
        {% else %}
          <a href="{{ url_for('login') }}" style="color:white;margin-right:8px;">Login</a>
          <a href="{{ url_for('signup') }}" style="color:white;">Signup</a>
        {% endif %}
      </div>
    </header>

    <main class="main" role="main">
      <section class="upload-card card">
        <h2>Upload Your Video</h2>
        <p class="lead">Select or drop a video file below (.mp4, .mov, .avi, .mkv).</p>

        <div id="dropzone" class="dropzone">
          <div class="hint">Drag & drop your video here</div>
          <div>or</div>
          <button id="chooseBtn" class="btn-choose" type="button">Choose File</button>
          <div class="small-note">Recommended: short clips up to 200MB</div>
          <input id="fileInput" name="file" type="file" accept="video/*,image/*" style="display:none" />
        </div>

        <div id="progressBar">
          <div id="progress"></div>
        </div>

        <div id="status" style="text-align:center; margin-top:12px; color:rgba(255,255,255,0.85)"></div>
        <div id="result" style="text-align:center; margin-top:12px; font-weight:700;"></div>
      </section>

      <section class="info-card">
        <h3>What is a Deepfake?</h3>
        <p>Deepfakes are AI-generated videos or images that make it look like a person said or did something they never actually did. They are created using deep learning models such as GANs and can be very convincing — this tool gives a quick automated check as a first step when you suspect manipulation.</p>
      </section>
    </main>
  </div>

  <script src="{{ url_for('static', filename='upload.js') }}"></script>
</body>
</html>
```

## Base HTML : 

```python 
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{{ title if title else "Detectify" }}</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">

  <style>
    body {
      margin:0;
      font-family:"Inter", system-ui;
      background:linear-gradient(180deg, #0f274c, #1b3a74);
      color:white;
      min-height:100vh;
    }

    header{
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:24px 40px;
      border-bottom:1px solid rgba(255,255,255,0.08);
      backdrop-filter:blur(6px);
    }

    .brand {display:flex;gap:14px;align-items:center;}
    .logo {
      width:40px;height:40px;border-radius:8px;
      background:linear-gradient(135deg,#5eead4,#7c3aed);
      font-weight:800;color:#04202a;
      display:flex;justify-content:center;align-items:center;
    }
    .brand-title {font-size:20px;margin:0;font-weight:800;}
    .brand-sub {margin:0;font-size:12px;opacity:0.75;}

    a { color:#5eead4; text-decoration:none; }
    .nav-links a {margin-left:12px;font-size:14px;}

    /* flash messages */
    .flash-box{
      max-width:900px;
      margin:20px auto 0 auto;
      padding:12px 18px;
      background:rgba(255,255,255,0.09);
      border-left:4px solid #5eead4;
      border-radius:8px;
      font-size:14px;
    }

    /* content area wrapper */
    .content-wrapper {
      max-width:1100px;
      margin:30px auto;
      padding:0 16px;
    }
  </style>

  {% block head %}{% endblock %}
</head>

<body>
  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="logo">π</div>
      <div>
        <div class="brand-title">Detectify</div>
        <div class="brand-sub">AI-powered deepfake analysis</div>
      </div>
    </div>

    <div class="nav-links">
      {% if current_user.is_authenticated %}
        Signed in as <strong>{{ current_user.email }}</strong>
        <a href="{{ url_for('logout') }}">Logout</a>
      {% else %}
        <a href="{{ url_for('login') }}">Login</a>
        <a href="{{ url_for('signup') }}">Signup</a>
      {% endif %}
    </div>
  </header>

  <!-- FLASH MESSAGES -->
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
      {% for category, msg in messages %}
        <div class="flash-box">
          <strong>{{ category.title() }}:</strong> {{ msg }}
        </div>
      {% endfor %}
    {% endif %}
  {% endwith %}

  <!-- MAIN CONTENT -->
  <div class="content-wrapper">
    {% block content %}{% endblock %}
  </div>

  {% block scripts %}{% endblock %}
</body>
</html>
```
## SignUp HTML : 

```python 
{% extends "base.html" %}

{% block head %}
<style>
.card{
  width:100%;
  max-width:420px;
  padding:32px;
  border-radius:16px;
  background:rgba(255,255,255,0.05);
  box-shadow:0 10px 30px rgba(0,0,0,0.3);
  backdrop-filter:blur(8px);
  margin: 40px auto;
}
h2{margin-top:0;text-align:center;font-weight:700;}
input{
  width:100%;
  padding:12px;
  margin:8px 0 16px 0;
  border-radius:10px;
  border:none;
  outline:none;
  background:rgba(255,255,255,0.12);
  color:white;
}
button{
  width:100%;
  padding:12px;
  border-radius:10px;
  border:none;
  cursor:pointer;
  font-weight:700;
  color:white;
  background:linear-gradient(90deg,#4f46e5,#06b6d4);
  box-shadow:0 6px 20px rgba(0,0,0,0.35);
}
a{color:#5eead4;text-decoration:none;}
.small{text-align:center;margin-top:14px;color:#ccc;font-size:14px;}
</style>
{% endblock %}

{% block content %}
  <div class="card">
    <h2>Create Account</h2>

    <form method="POST" action="{{ url_for('signup') }}">
      <label>Email</label>
      <input type="email" name="email" required placeholder="Enter email">

      <label>Password</label>
      <input type="password" name="password" required placeholder="Choose password">

      <button type="submit">Sign Up</button>
    </form>

    <p class="small">Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
  </div>
{% endblock %}
```

## Login HTML : 

```python 
{% extends "base.html" %}

{% block head %}
<style>
.page-bg {
    min-height: 100vh;
    background: linear-gradient(180deg, #0f274c, #1b3a74);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 0;
}

.card{
  width:100%;
  max-width:420px;
  padding:32px;
  border-radius:16px;
  background:rgba(255,255,255,0.05);
  box-shadow:0 10px 30px rgba(0,0,0,0.3);
  backdrop-filter:blur(8px);
}
h2{margin-top:0;text-align:center;font-weight:700;}
input{
  width:100%;
  padding:12px;
  margin:8px 0 16px 0;
  border-radius:10px;
  border:none;
  outline:none;
  background:rgba(255,255,255,0.12);
  color:white;
}
button{
  width:100%;
  padding:12px;
  border-radius:10px;
  border:none;
  cursor:pointer;
  font-weight:700;
  color:white;
  background:linear-gradient(90deg,#4f46e5,#06b6d4);
  box-shadow:0 6px 20px rgba(0,0,0,0.35);
}
a{color:#5eead4;text-decoration:none;}
.small{text-align:center;margin-top:14px;color:#ccc;font-size:14px;}
</style>
{% endblock %}

{% block content %}

<div class="page-bg">
  <div class="card">
    <h2>Log In</h2>

    <form method="POST" action="{{ url_for('login') }}">
      <label>Email</label>
      <input type="email" name="email" required placeholder="Enter email">

      <label>Password</label>
      <input type="password" name="password" required placeholder="Enter password">

      <button type="submit">Log In</button>
    </form>

    <p class="small">Don't have an account? <a href="{{ url_for('signup') }}">Sign up</a></p>
  </div>
</div>

{% endblock %}
```

