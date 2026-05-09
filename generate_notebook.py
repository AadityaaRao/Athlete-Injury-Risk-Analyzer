import json, textwrap

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": []
}

def md(text):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(keepends=True)})

def code(text):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(keepends=True), "execution_count": None, "outputs": []})

# ====================  TITLE  ====================
md("""
    # 🏃 Analyzing an Athlete's Movements to Optimize Performance and Reduce Injury Risk

    **ML Term Paper — Complete Implementation Notebook**

    This notebook implements an end-to-end pipeline that:
    1. Takes a single photo of an athlete as input
    2. Uses Google's **MediaPipe PoseLandmarker** (pretrained) to detect body landmarks
    3. Extracts **biomechanical features** (joint angles, symmetry indices)
    4. Trains **Random Forest** and **XGBoost** classifiers on synthetic data
    5. Predicts **injury risk** (Low / High) with a confidence score
    6. Launches an interactive **Gradio web app** for easy testing

    > **No dataset download required.** Everything runs in Google Colab.
""")

# ====================  STEP 1  ====================
md("""
    ---
    ## Step 1 — Install Required Libraries & Verify Versions

    We install all the Python packages we need. After installing, we import every library and
    print its version so we can confirm everything is set up correctly.

    | Library | Purpose |
    |---|---|
    | `mediapipe` | Pretrained pose estimation (by Google) |
    | `gradio` | Build a web UI for our model |
    | `xgboost` | Gradient-boosted tree classifier |
    | `scikit-learn` | Random Forest, metrics, train/test split |
    | `opencv-python` | Image reading and drawing |
    | `numpy` | Numerical arrays |
    | `matplotlib` / `seaborn` | Plotting |
""")

code("""
    # Install all required packages in one go
    # The '!' prefix runs shell commands inside Colab
    !pip install -q mediapipe gradio xgboost scikit-learn opencv-python-headless numpy matplotlib seaborn
""")

code("""
    # ── Import every library we will use ──

    import mediapipe as mp                # Google's pose estimation toolkit
    import cv2                            # OpenCV — read / draw on images
    import numpy as np                    # Fast numerical arrays
    import matplotlib.pyplot as plt       # Plotting library
    import seaborn as sns                 # Pretty statistical plots
    import gradio as gr                   # Web UI builder
    import xgboost as xgb                 # XGBoost classifier
    import joblib                         # Save / load trained models to disk
    import math                           # math.degrees, math.acos, etc.
    import os                             # File-system utilities
    import urllib.request                  # Download files from the internet
    import warnings                       # Suppress noisy warnings

    from sklearn.ensemble import RandomForestClassifier   # Random Forest model
    from sklearn.model_selection import train_test_split   # Split data 80/20
    from sklearn.metrics import (                          # Evaluation metrics
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        classification_report
    )

    # New MediaPipe Tasks API imports
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    warnings.filterwarnings('ignore')  # Hide non-critical warnings for cleaner output

    # ── Print versions to confirm everything installed correctly ──
    import sklearn
    print('Library versions:')
    print(f'  mediapipe   : {mp.__version__}')
    print(f'  opencv      : {cv2.__version__}')
    print(f'  numpy       : {np.__version__}')
    print(f'  matplotlib  : {plt.matplotlib.__version__}')
    print(f'  seaborn     : {sns.__version__}')
    print(f'  gradio      : {gr.__version__}')
    print(f'  xgboost     : {xgb.__version__}')
    print(f'  scikit-learn: {sklearn.__version__}')
    print('\\n✅ All libraries imported successfully!')
""")

# ====================  STEP 2  ====================
md("""
    ---
    ## Step 2 — Pose Estimation Using Pretrained MediaPipe

    MediaPipe PoseLandmarker is a **pretrained deep-learning model** by Google that detects
    **33 body landmarks** (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.) from a
    single image.

    **Important:** The legacy `mp.solutions.pose` API has been removed. We use the current
    **MediaPipe Tasks API** (`mediapipe.tasks.python.vision.PoseLandmarker`).

    We first download the model file (`pose_landmarker_full.task`), then write a function
    `run_pose_estimation(image_input)` that:
    1. Reads an image (from a file path or a numpy array)
    2. Runs PoseLandmarker to find body landmarks
    3. Draws the skeleton on the image
    4. Returns the annotated image **and** the raw landmarks

    If no person is detected the function returns a friendly message instead of crashing.
""")

code("""
    # ── Download the PoseLandmarker model file from Google ──
    # This is a ~30 MB pretrained model bundle that MediaPipe needs.
    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
    MODEL_PATH = '/tmp/pose_landmarker_full.task'

    if not os.path.exists(MODEL_PATH):
        print('Downloading PoseLandmarker model (~30 MB)...')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f'✅ Model saved to {MODEL_PATH}')
    else:
        print(f'✅ Model already exists at {MODEL_PATH}')
""")

code("""
    # ── Landmark indices (same numbering as the legacy API) ──
    # MediaPipe Pose defines 33 body landmarks, each with an integer index.
    # We store the indices we need as named constants for readability.
    # Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model

    class LM:
        \"\"\"Landmark index constants (matches MediaPipe Pose spec).\"\"\"
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_PINKY = 17
        RIGHT_PINKY = 18
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_THUMB = 21
        RIGHT_THUMB = 22
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32

    # ── Pose connections for skeleton drawing ──
    # Each tuple is (start_landmark_index, end_landmark_index).
    # These define which landmarks to connect with lines when drawing.
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),   # Left face
        (0,4),(4,5),(5,6),(6,8),   # Right face
        (9,10),                     # Mouth
        (11,12),                    # Shoulders
        (11,13),(13,15),            # Left arm
        (12,14),(14,16),            # Right arm
        (11,23),(12,24),            # Torso sides
        (23,24),                    # Hips
        (23,25),(25,27),            # Left leg
        (24,26),(26,28),            # Right leg
        (27,29),(29,31),(31,27),    # Left foot
        (28,30),(30,32),(32,28),    # Right foot
        (15,17),(15,19),(15,21),    # Left hand
        (16,18),(16,20),(16,22),    # Right hand
    ]

    print('✅ Landmark constants and skeleton connections defined.')
""")

code("""
    def _draw_skeleton(image_rgb, landmarks):
        \"\"\"
        Draw pose landmarks and skeleton connections on an RGB image.

        Parameters
        ----------
        image_rgb : numpy array (H, W, 3) in RGB
        landmarks : list of mediapipe NormalizedLandmark objects (33 items)

        Returns
        -------
        annotated : numpy array (H, W, 3) in RGB with skeleton drawn
        \"\"\"
        annotated = image_rgb.copy()
        h, w, _ = annotated.shape  # Image height and width (needed to convert normalized coords to pixels)

        # Draw connection lines first (so dots appear on top)
        for (start_idx, end_idx) in POSE_CONNECTIONS:
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            # Convert normalized (0-1) coordinates to pixel coordinates
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Draw landmark dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), radius=4, color=(255, 0, 0), thickness=-1)  # Filled red circle

        return annotated


    def run_pose_estimation(image_input):
        \"\"\"
        Run MediaPipe PoseLandmarker on an image.

        Parameters
        ----------
        image_input : str or numpy.ndarray
            Either a file path (string) or a numpy array in RGB (from Gradio).

        Returns
        -------
        annotated_image : numpy.ndarray or None
            The image with the skeleton drawn on it (RGB), or None if no person found.
        landmarks : list or None
            List of 33 NormalizedLandmark objects, or None if no person found.
        message : str
            A status message (success or error).
        \"\"\"

        # ── 1. Build a MediaPipe Image object ──
        if isinstance(image_input, str):
            # image_input is a file path
            if not os.path.exists(image_input):
                return None, None, '❌ Could not find the image file. Check the path.'
            # Use OpenCV to read so we easily unify format
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                return None, None, '❌ Could not read the image file.'
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        else:
            # image_input is a numpy array (e.g. from Gradio)
            if image_input is None:
                return None, None, '❌ No image provided.'
            # Ensure uint8
            img_array = np.asarray(image_input, dtype=np.uint8)
            
            # MediaPipe Pose requires exactly 3 channels (RGB)
            # If the image is grayscale or has an alpha channel, convert it
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
            # Put into MediaPipe Image container
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        # ── 2. Create the PoseLandmarker and run detection ──
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,             # We don't need segmentation
            num_poses=1                                   # Detect at most 1 person
        )

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)          # Run the model

        # ── 3. Check if any person was detected ──
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None, '⚠️ No person detected in the image. Try a clearer photo.'

        # Get the first (and only) detected person's landmarks
        landmarks = result.pose_landmarks[0]  # List of 33 NormalizedLandmark

        # ── 4. Draw skeleton on the image ──
        # Convert mp.Image to a numpy array (RGB)
        image_rgb = mp_image.numpy_view().copy()  # .copy() so we can draw on it
        annotated_rgb = _draw_skeleton(image_rgb, landmarks)

        return annotated_rgb, landmarks, '✅ Pose detected successfully!'


    print('✅ run_pose_estimation() defined.')
""")

md("""
    ### Quick test — download a sample athlete image and run pose estimation

    We download a public-domain image of an athlete and pass it through our function
    to make sure everything works.
""")

code("""
    # ── Helper function to download images with proper headers ──
    # Some websites block downloads that don't include a User-Agent header.
    def _download_image(url, save_path):
        \"\"\"Download an image from a URL to a local file path.\"\"\"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())

    # Download a sample image of an athlete (basketball player) for testing
    # Using a stable GitHub Raw URL from OpenCV's official sample data repository
    sample_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png'
    sample_path = '/tmp/sample_athlete.png'

    _download_image(sample_url, sample_path)
    print(f'Downloaded sample image to {sample_path}')

    # Run pose estimation on it
    annotated_img, landmarks, msg = run_pose_estimation(sample_path)
    print(msg)

    # Display the result
    if annotated_img is not None:
        plt.figure(figsize=(8, 10))
        plt.imshow(annotated_img)
        plt.title('Pose Estimation Result', fontsize=14)
        plt.axis('off')        # Hide the x/y axes — we just want to see the image
        plt.tight_layout()
        plt.show()
    else:
        print('No pose detected — try another image.')
""")

# ====================  STEP 3  ====================
md(r"""
    ---
    ## Step 3 — Biomechanical Feature Extraction

    From the 33 body landmarks we compute **biomechanical features** that are meaningful
    for injury-risk analysis:

    ### Joint Angles (8 total)
    Each angle is computed at a joint using three landmarks (the joint itself and two
    connected bones). We use the **3-point vector dot-product** formula:

    $$\theta = \arccos\!\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}|\,|\vec{BC}|}\right)$$

    | Angle | Three landmarks used |
    |---|---|
    | Left Knee | Left Hip → **Left Knee** → Left Ankle |
    | Right Knee | Right Hip → **Right Knee** → Right Ankle |
    | Left Hip | Left Shoulder → **Left Hip** → Left Knee |
    | Right Hip | Right Shoulder → **Right Hip** → Right Knee |
    | Left Elbow | Left Shoulder → **Left Elbow** → Left Wrist |
    | Right Elbow | Right Shoulder → **Right Elbow** → Right Wrist |
    | Left Shoulder | Left Elbow → **Left Shoulder** → Left Hip |
    | Right Shoulder | Right Elbow → **Right Shoulder** → Right Hip |

    ### Symmetry Indices (4 total)
    $$\text{SI} = \frac{|L - R|}{(L + R) / 2 + 10^{-6}}$$

    A value near 0 means the left and right sides are symmetric (good);
    a high value means asymmetry (potential injury risk).
""")

code("""
    # Each entry: (index_A, index_B_center, index_C)  — angle is measured AT B
    JOINT_ANGLE_DEFS = {
        'left_knee_angle':      (LM.LEFT_HIP,       LM.LEFT_KNEE,      LM.LEFT_ANKLE),
        'right_knee_angle':     (LM.RIGHT_HIP,      LM.RIGHT_KNEE,     LM.RIGHT_ANKLE),
        'left_hip_angle':       (LM.LEFT_SHOULDER,   LM.LEFT_HIP,       LM.LEFT_KNEE),
        'right_hip_angle':      (LM.RIGHT_SHOULDER,  LM.RIGHT_HIP,      LM.RIGHT_KNEE),
        'left_elbow_angle':     (LM.LEFT_SHOULDER,   LM.LEFT_ELBOW,     LM.LEFT_WRIST),
        'right_elbow_angle':    (LM.RIGHT_SHOULDER,  LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
        'left_shoulder_angle':  (LM.LEFT_ELBOW,      LM.LEFT_SHOULDER,  LM.LEFT_HIP),
        'right_shoulder_angle': (LM.RIGHT_ELBOW,     LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    }

    # Bilateral pairs for symmetry index calculation
    SYMMETRY_PAIRS = [
        ('left_knee_angle',     'right_knee_angle',     'knee_symmetry'),
        ('left_hip_angle',      'right_hip_angle',      'hip_symmetry'),
        ('left_elbow_angle',    'right_elbow_angle',     'elbow_symmetry'),
        ('left_shoulder_angle', 'right_shoulder_angle',  'shoulder_symmetry'),
    ]


    def _calc_angle(a, b, c):
        \"\"\"
        Calculate the angle at point B given three 3-D points A, B, C.
        Each point is a NormalizedLandmark with .x, .y, .z attributes.
        Returns the angle in degrees.
        \"\"\"
        # Convert each landmark to a numpy array [x, y, z]
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])

        # Vectors from B to A, and from B to C
        ba = a - b
        bc = c - b

        # Dot product divided by product of magnitudes gives cos(angle)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)

        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert radians → degrees
        angle_deg = math.degrees(math.acos(cos_angle))
        return angle_deg


    def extract_features(landmarks):
        \"\"\"
        Extract biomechanical features from PoseLandmarker landmarks.

        Parameters
        ----------
        landmarks : list of NormalizedLandmark (33 items)

        Returns
        -------
        feature_array : np.ndarray   — flat array of all feature values (for the ML model)
        feature_dict  : dict         — readable {feature_name: value} (for display)
        \"\"\"
        feature_dict = {}  # Human-readable feature names → values

        # ── Compute 8 joint angles ──
        for name, (idx_a, idx_b, idx_c) in JOINT_ANGLE_DEFS.items():
            angle = _calc_angle(landmarks[idx_a], landmarks[idx_b], landmarks[idx_c])
            feature_dict[name] = round(angle, 2)

        # ── Compute 4 symmetry indices ──
        for left_name, right_name, si_name in SYMMETRY_PAIRS:
            left_val  = feature_dict[left_name]
            right_val = feature_dict[right_name]
            # SI = |L - R| / ((L + R) / 2 + epsilon)  — epsilon prevents division by zero
            si = abs(left_val - right_val) / ((left_val + right_val) / 2 + 1e-6)
            feature_dict[si_name] = round(si, 4)

        # ── Build a flat numpy array (consistent order every time) ──
        feature_names = list(feature_dict.keys())
        feature_array = np.array([feature_dict[k] for k in feature_names], dtype=np.float32)

        return feature_array, feature_dict


    # Print the feature names so the student can see what gets extracted
    dummy_names = list(JOINT_ANGLE_DEFS.keys()) + [p[2] for p in SYMMETRY_PAIRS]
    print('Features that will be extracted (12 total):')
    for i, name in enumerate(dummy_names, 1):
        print(f'  {i:2d}. {name}')
    print('\\n✅ extract_features() defined.')
""")

md("### Quick test — extract features from the sample image")

code("""
    # Run feature extraction on the landmarks we detected earlier
    if landmarks is not None:
        feat_array, feat_dict = extract_features(landmarks)
        print('Extracted features:')
        for name, val in feat_dict.items():
            unit = '°' if 'angle' in name else ''
            print(f'  {name:30s} = {val:8.2f}{unit}')
        print(f'\\nFeature array shape: {feat_array.shape}')
    else:
        print('Skipping — no landmarks available from previous step.')
""")

# ====================  STEP 4  ====================
md("""
    ---
    ## Step 4 — Rule-Based Risk Label Generator

    Since we are generating **synthetic** training data (Step 5), we need a way to label
    each sample as **High Risk (1)** or **Low Risk (0)**.

    We use biomechanics rules derived from sports-science literature:

    | Condition | Biomechanical Reason | Risk |
    |---|---|---|
    | Either knee angle < 70° | Deep squat / extreme flexion — ACL / meniscus stress | High Risk |
    | Any bilateral symmetry index > 0.35 | Severe left-right imbalance — compensatory injury risk | High Risk |
    | Either hip angle < 55° | Extreme forward fold — lumbar spine overload | High Risk |
    | Either elbow angle < 25° | Extreme elbow compression — joint stress | High Risk |
    | Either shoulder angle > 165° | Overhead position — shoulder impingement risk | High Risk |
    | None of the above | Posture within safe athletic ranges | Low Risk |
""")

code("""
    def generate_risk_label(features_dict):
        \"\"\"
        Apply biomechanical rules to decide High Risk (1) or Low Risk (0).

        Parameters
        ----------
        features_dict : dict
            Dictionary with feature names as keys and values in degrees / ratios.

        Returns
        -------
        int : 1 for High Risk, 0 for Low Risk.
        \"\"\"

        # ── Rule 1: Extreme knee flexion (deep squat territory) ──
        # Angles below 70° put enormous stress on the ACL and meniscus.
        # Normal standing/walking has knee angles of 160°–180°, which is perfectly safe.
        if features_dict['left_knee_angle'] < 70 or features_dict['right_knee_angle'] < 70:
            return 1

        # ── Rule 2: Severe bilateral asymmetry ──
        # A symmetry index above 0.35 means one side differs from the other by > 35%,
        # which indicates compensatory movement patterns that increase injury risk.
        for _, _, si_name in SYMMETRY_PAIRS:
            if features_dict[si_name] > 0.35:
                return 1

        # ── Rule 3: Extreme hip flexion ──
        # Hip angles below 55° indicate an extreme forward fold that overloads the lumbar spine.
        if features_dict['left_hip_angle'] < 55 or features_dict['right_hip_angle'] < 55:
            return 1

        # ── Rule 4: Extreme elbow compression ──
        # Very tight elbow angles (< 25°) put stress on the joint capsule.
        if features_dict['left_elbow_angle'] < 25 or features_dict['right_elbow_angle'] < 25:
            return 1

        # ── Rule 5: Overhead shoulder position (impingement risk) ──
        # Shoulder angles above 165° mean the arm is nearly overhead,
        # which can cause rotator cuff impingement during repetitive movements.
        if features_dict['left_shoulder_angle'] > 165 or features_dict['right_shoulder_angle'] > 165:
            return 1

        # ── If none of the danger rules triggered → Low Risk ──
        return 0


    print('✅ generate_risk_label() defined.')
""")

# ====================  STEP 5  ====================
md("""
    ---
    ## Step 5 — Synthetic Training Data Generation

    Real labelled injury-risk datasets require manual downloads and extensive preprocessing.
    Instead, we **simulate 3 000 realistic samples** using random distributions that mirror
    actual athletic ranges.

    Each sample has the same 12 features our `extract_features()` function produces.
    We label each sample with `generate_risk_label()`.

    **Key design choice:** After labeling, we add **Gaussian noise** (±8°) to the joint angles
    to simulate the measurement error that MediaPipe introduces on real images. This means
    the ML models must learn to handle noisy inputs — this is the core reason we use ML
    instead of just hard-coding the rules.
""")

code("""
    # ── Configuration ──
    NUM_SAMPLES = 3000                 # Total number of synthetic athletes
    RANDOM_SEED = 42                   # For reproducibility
    np.random.seed(RANDOM_SEED)

    # ── Feature name list (must match extract_features output order) ──
    FEATURE_NAMES = [
        'left_knee_angle', 'right_knee_angle',
        'left_hip_angle',  'right_hip_angle',
        'left_elbow_angle','right_elbow_angle',
        'left_shoulder_angle','right_shoulder_angle',
        'knee_symmetry',   'hip_symmetry',
        'elbow_symmetry',  'shoulder_symmetry'
    ]

    data_rows = []   # Will hold one dict per sample (clean, for labeling)

    for _ in range(NUM_SAMPLES):
        row = {}
        # Joint angles — sampled from uniform distributions within athletic ranges
        row['left_knee_angle']      = np.random.uniform(60, 180)
        row['right_knee_angle']     = np.random.uniform(60, 180)
        row['left_hip_angle']       = np.random.uniform(50, 160)
        row['right_hip_angle']      = np.random.uniform(50, 160)
        row['left_elbow_angle']     = np.random.uniform(10, 170)
        row['right_elbow_angle']    = np.random.uniform(10, 170)
        row['left_shoulder_angle']  = np.random.uniform(20, 180)
        row['right_shoulder_angle'] = np.random.uniform(20, 180)

        # Symmetry indices — absolute value of a normal distribution (always >= 0)
        row['knee_symmetry']     = abs(np.random.normal(0, 0.12))
        row['hip_symmetry']      = abs(np.random.normal(0, 0.12))
        row['elbow_symmetry']    = abs(np.random.normal(0, 0.12))
        row['shoulder_symmetry'] = abs(np.random.normal(0, 0.12))

        data_rows.append(row)

    # ── Label each sample using the CLEAN features ──
    y = np.array([generate_risk_label(row) for row in data_rows], dtype=np.int32)

    # ── Now add Gaussian noise to simulate real-world measurement error ──
    # MediaPipe's pose estimation introduces ~5-10° of noise on joint angles.
    # By adding this noise AFTER labeling, the ML model must learn to generalize
    # from noisy inputs — this is why ML outperforms a simple if/else rule check.
    NOISE_STD_ANGLES = 8.0     # Standard deviation of noise in degrees
    NOISE_STD_SYMMETRY = 0.05  # Standard deviation of noise for symmetry indices

    X_clean = np.array([[row[f] for f in FEATURE_NAMES] for row in data_rows], dtype=np.float32)

    # Add noise to angle columns (first 8 features)
    angle_noise = np.random.normal(0, NOISE_STD_ANGLES, size=(NUM_SAMPLES, 8))
    # Add noise to symmetry columns (last 4 features)
    sym_noise = np.random.normal(0, NOISE_STD_SYMMETRY, size=(NUM_SAMPLES, 4))

    X = X_clean.copy()
    X[:, :8] += angle_noise.astype(np.float32)   # Add angle noise
    X[:, 8:] += sym_noise.astype(np.float32)      # Add symmetry noise
    X[:, 8:] = np.abs(X[:, 8:])                   # Symmetry indices must stay non-negative

    # ── Show class distribution ──
    unique, counts = np.unique(y, return_counts=True)
    print('Class distribution:')
    for cls, cnt in zip(unique, counts):
        label = 'Low Risk' if cls == 0 else 'High Risk'
        print(f'  {label} (y={cls}): {cnt} samples  ({cnt/len(y)*100:.1f}%)')

    # ── Train / Test split (80% train, 20% test) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y               # Keep class proportions the same in both splits
    )

    print(f'\\nTraining set : {X_train.shape[0]} samples')
    print(f'Test set     : {X_test.shape[0]} samples')
    print(f'Features     : {X_train.shape[1]}')
    print('\\n✅ Synthetic data generated and split.')
""")

# ====================  STEP 6  ====================
md("""
    ---
    ## Step 6 — Train & Compare ML Models

    We train two classifiers on the synthetic data and compare them:

    1. **Random Forest** — an ensemble of decision trees that vote on the final prediction
    2. **XGBoost** — gradient-boosted decision trees that learn sequentially from each other's errors

    We evaluate both with: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**.
    The better model (by ROC-AUC) is saved as `best_model.pkl`.
""")

code("""
    # ══════════════════════════════════════════════════
    # Train Random Forest
    # ══════════════════════════════════════════════════
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Build 100 decision trees
        random_state=42
    )
    rf_model.fit(X_train, y_train)        # Train the model
    rf_preds = rf_model.predict(X_test)   # Predict on the test set
    rf_probs = rf_model.predict_proba(X_test)[:, 1]  # Probability of class 1 (High Risk)

    # ══════════════════════════════════════════════════
    # Train XGBoost
    # ══════════════════════════════════════════════════
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,     # Step size for gradient boosting
        random_state=42,
        eval_metric='logloss'  # Metric used internally during training
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # ══════════════════════════════════════════════════
    # Evaluate both models
    # ══════════════════════════════════════════════════
    def evaluate_model(name, y_true, y_pred, y_prob):
        \"\"\"Print evaluation metrics and return them as a dict.\"\"\"
        metrics = {
            'Accuracy':  accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall':    recall_score(y_true, y_pred, zero_division=0),
            'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
            'ROC-AUC':   roc_auc_score(y_true, y_prob),
        }
        print(f'\\n── {name} ──')
        for k, v in metrics.items():
            print(f'  {k:12s}: {v:.4f}')
        return metrics

    rf_metrics  = evaluate_model('Random Forest', y_test, rf_preds, rf_probs)
    xgb_metrics = evaluate_model('XGBoost',       y_test, xgb_preds, xgb_probs)

    # ══════════════════════════════════════════════════
    # Pick the winner by ROC-AUC and save it
    # ══════════════════════════════════════════════════
    if xgb_metrics['ROC-AUC'] >= rf_metrics['ROC-AUC']:
        best_model = xgb_model
        best_name  = 'XGBoost'
    else:
        best_model = rf_model
        best_name  = 'Random Forest'

    # Save the winning model to disk
    joblib.dump(best_model, 'best_model.pkl')
    print(f'\\n🏆 Winner: {best_name} (ROC-AUC = '
          f'{max(rf_metrics["ROC-AUC"], xgb_metrics["ROC-AUC"]):.4f})')
    print('   Saved as best_model.pkl')
""")

md("### Confusion Matrices")

code("""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, y_pred) in zip(axes, [('Random Forest', rf_preds), ('XGBoost', xgb_preds)]):
        cm = confusion_matrix(y_test, y_pred)  # 2×2 matrix of TP/TN/FP/FN
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            ax=ax
        )
        ax.set_title(f'{name} — Confusion Matrix', fontsize=13)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()
""")

md("### Feature Importance Charts")

code("""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, model) in zip(axes, [('Random Forest', rf_model), ('XGBoost', xgb_model)]):
        importances = model.feature_importances_          # Array of importance scores
        sorted_idx  = np.argsort(importances)             # Sort ascending for horizontal bar chart
        ax.barh(
            [FEATURE_NAMES[i] for i in sorted_idx],       # Feature names on y-axis
            importances[sorted_idx],                       # Importance values on x-axis
            color='steelblue'
        )
        ax.set_title(f'{name} — Feature Importance', fontsize=13)
        ax.set_xlabel('Importance Score')

    plt.tight_layout()
    plt.show()
""")

# ====================  STEP 7  ====================
md("""
    ---
    ## Step 7 — Full Prediction Pipeline

    We now combine everything into a single function `predict_injury_risk(image)` that:
    1. Runs pose estimation
    2. Extracts biomechanical features
    3. Loads the saved model and predicts injury risk
    4. Returns the annotated image, a feature summary, and the risk verdict
""")

code("""
    def predict_injury_risk(image):
        \"\"\"
        End-to-end pipeline: image → pose → features → prediction.

        Parameters
        ----------
        image : numpy.ndarray
            An RGB image (e.g. from Gradio upload).

        Returns
        -------
        annotated_image  : numpy.ndarray or None
        feature_text     : str   — Pretty-printed feature summary
        risk_text        : str   — Risk label + confidence
        suggestions_text : str   — Actionable advice to reduce risk
        \"\"\"

        # ── 1. Pose estimation ──
        annotated_img, lm, msg = run_pose_estimation(image)

        if lm is None:
            # No person detected — return a friendly message
            return None, '⚠️ No features extracted.', msg

        # ── 2. Feature extraction ──
        feat_array, feat_dict = extract_features(lm)

        # ── 3. Load the saved model ──
        model = joblib.load('best_model.pkl')

        # ── 4. Predict ──
        feat_2d = feat_array.reshape(1, -1)         # Model expects 2D: (1 sample, 12 features)
        prediction  = model.predict(feat_2d)[0]     # 0 or 1
        probability = model.predict_proba(feat_2d)[0]  # [prob_class_0, prob_class_1]
        confidence  = probability[prediction] * 100    # Confidence % for the chosen class

        # ── 5. Build pretty text outputs ──
        lines = []
        for name, val in feat_dict.items():
            unit = '°' if 'angle' in name else ''
            lines.append(f'{name:30s}  {val:8.2f}{unit}')
        feature_text = '\\n'.join(lines)

        # Risk verdict & Suggestions
        suggestions = []
        if prediction == 1:
            risk_text = f'🔴 HIGH RISK  (confidence: {confidence:.1f}%)'
            
            # Generate specific advice based on the biomechanical rules
            if feat_dict['left_knee_angle'] < 70 or feat_dict['right_knee_angle'] < 70:
                suggestions.append("🦵 Deep Knee Flexion: Avoid 'bouncing' at the bottom of the movement. Reduce load if experiencing knee pain.")
            if any(feat_dict[k] > 0.35 for k in ['knee_symmetry', 'hip_symmetry', 'elbow_symmetry', 'shoulder_symmetry']):
                suggestions.append("⚖️ Severe Asymmetry: Focus on unilateral (single-side) exercises to correct muscle imbalances and prevent compensations.")
            if feat_dict['left_hip_angle'] < 55 or feat_dict['right_hip_angle'] < 55:
                suggestions.append("🔙 Extreme Forward Fold: Brace your core and maintain a neutral spine to avoid lumbar overload.")
            if feat_dict['left_elbow_angle'] < 25 or feat_dict['right_elbow_angle'] < 25:
                suggestions.append("💪 Extreme Elbow Compression: Avoid holding heavy resistance in fully maxed-out elbow flexion.")
            if feat_dict['left_shoulder_angle'] > 165 or feat_dict['right_shoulder_angle'] > 165:
                suggestions.append("✋ Extreme Overhead Reach: Ensure proper rotator cuff mobility warmup. Limit repetitive overhead loading.")
            
            if not suggestions:
                suggestions.append("⚠️ General high risk detected. Consult a sports therapist to evaluate your form.")
        else:
            risk_text = f'🟢 LOW RISK   (confidence: {confidence:.1f}%)'
            suggestions.append("✅ Posture is within safe athletic ranges. Maintain good form and ensure proper recovery.")

        suggestions_text = '\\n\\n'.join(suggestions)

        return annotated_img, feature_text, risk_text, suggestions_text


    print('✅ predict_injury_risk() defined.')
""")

md("### Quick test — run the full pipeline on the sample image")

code("""
    # Test the full pipeline using the sample athlete image
    test_img = cv2.imread(sample_path)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

    result_img, result_features, result_risk, result_suggestions = predict_injury_risk(test_img_rgb)

    if result_img is not None:
        plt.figure(figsize=(8, 10))
        plt.imshow(result_img)
        plt.title('Full Pipeline Test', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print('\\n── Extracted Features ──')
    print(result_features)
    print('\\n── Risk Assessment ──')
    print(result_risk)
    print('\\n── Suggestions ──')
    print(result_suggestions)
""")

# ====================  STEP 8  ====================
md("""
    ---
    ## Step 8 — Gradio Web App 🚀

    Finally, we build an interactive web interface using **Gradio**.

    - Upload any photo of an athlete
    - See the pose skeleton overlaid on the image
    - Read the extracted biomechanical features
    - Get a **Low Risk / High Risk** prediction with confidence

    The `share=True` flag generates a **public link** so you can share the app with
    anyone — even from Colab.
""")

code("""
    # Download example images so the Gradio demo has built-in samples to try
    example_dir = '/tmp/examples'
    os.makedirs(example_dir, exist_ok=True)

    example_urls = [
        ('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png',
         'basketball_player_1.png'),
        ('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball2.png',
         'basketball_player_2.png')
    ]

    example_paths = []
    for url, fname in example_urls:
        path = os.path.join(example_dir, fname)
        try:
            _download_image(url, path)  # Use our helper with proper User-Agent header
            example_paths.append([path])    # Gradio expects [[path1], [path2], ...]
            print(f'  ✅ Downloaded {fname}')
        except Exception as e:
            print(f'  ⚠️ Could not download {fname}: {e}')

    print(f'\\n{len(example_paths)} example images ready.')
""")

code("""
    # ══════════════════════════════════════════════════
    # Build and launch the Gradio app
    # ══════════════════════════════════════════════════

    demo = gr.Interface(
        fn=predict_injury_risk,                  # The function to call when user uploads an image
        inputs=gr.Image(type='numpy', label='Upload Athlete Photo'),  # Input widget
        outputs=[
            gr.Image(type='numpy', label='Pose Skeleton Overlay'),     # Output 1: annotated image
            gr.Textbox(label='Biomechanical Features', lines=14),      # Output 2: features text
            gr.Textbox(label='Injury Risk Prediction', lines=2),       # Output 3: risk label
            gr.Textbox(label='Prevention Suggestions', lines=4),       # Output 4: suggestions
        ],
        title='🏃 Athlete Injury Risk Analyzer',
        description=(
            'Upload a photo of an athlete. The system uses MediaPipe Pose to detect '
            'body landmarks, extracts biomechanical features (joint angles & symmetry), '
            'and predicts injury risk using a trained ML model. It also provides actionable '
            'advice based on the specific pose mechanics.'
        ),
        examples=example_paths if example_paths else None,
        allow_flagging='never',
    )

    # Launch with a public link (share=True) so anyone can access it
    demo.launch(share=True, debug=True)
""")

# ==================== WRITE NOTEBOOK ====================
output_path = r"D:\ML TERM PAPER\Athlete_Injury_Risk_Analyzer.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Wrote {len(nb['cells'])} cells to {output_path}")
