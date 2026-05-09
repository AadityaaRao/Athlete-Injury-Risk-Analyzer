# Athlete Injury Risk Analyzer - Project Overview

**High-Level Goal:** The project takes a single image of an athlete, uses computer vision to detect their posture, calculates biomechanical features (like joint angles), and uses Machine Learning to predict if the athlete is at high or low risk of injury.

---

## 1. Core Technologies Used
Before explaining *how* it works, you can briefly mention *what* you built it with:

* **MediaPipe:** A powerful AI model by Google used for Pose Estimation (detecting the body’s skeleton from an image).
* **Scikit-Learn & XGBoost:** The Machine Learning libraries used to train the classifiers.
* **Gradio:** A UI framework used to take all your code and turn it into a working, interactive web app.
* **NumPy & OpenCV:** Used for mathematical calculations and drawing the skeleton on the images.

---

## 2. The Step-by-Step Pipeline (How it works under the hood)
This is the heart of your project, explained in 5 logical phases:

### Phase 1: Pose Estimation (Computer Vision)
The system starts by taking in a single 2D photo of an athlete. It passes this image through **MediaPipe's PoseLandmarker**. 
* **What it does:** It maps out the human body in 3D space by pinpointing exactly **33 body landmarks** (like the nose, left shoulder, right knee, left ankle, etc.). 
* **Why it's important:** We can't do any mathematical analysis until we know exactly where the athlete's joints are located in the photo.

### Phase 2: Biomechanical Feature Extraction (Math & Physics)
Just having dots on a screen isn't enough for Machine Learning. We wrote custom Python algorithms to calculate **12 crucial biomechanical features** from those 33 landmarks:
* **8 Joint Angles:** We use vector math (the 3-point dot product formula, calculating inverse cosine) to find the exact angle of the left/right knees, hips, elbows, and shoulders. 
* **4 Bilateral Symmetry Indices:** We compare the left side to the right side to output a "symmetry score".
* *Why?* Extreme knee flexion or severe left-to-right imbalance are the primary indicators of structural stress and injury risk in sports science.

### Phase 3: Synthetic Data Generation (The "Clever" Part)
Normally, ML models need thousands of real-world examples to learn. Training data containing exact joint angles and injury risk labels is incredibly hard to find. To solve this, you generated **3,000 synthetic athlete samples**:
1. **Simulation:** You simulated random, mathematically realistic joint angles for 3,000 "virtual" athletes.
2. **Labeling:** You created sports-science rules to label them as `High Risk (1)` or `Low Risk (0)`. (e.g., if a knee angle is bent too deeply < 70 degrees, it is classified as High Risk).
3. **Adding Noise:** To make the AI useful in the real world, you purposefully injected "Gaussian Noise" (random statistical errors) into this data. Why? Because camera angles and MediaPipe predictions are never perfect. Introducing noise forces the final Machine Learning model to calculate probabilities and handle imperfect real-world photos rather than just relying on rigid if/then rules!

### Phase 4: Machine Learning Classification
With the 3,000 samples, the code splits the data (80% for training, 20% for testing) and trains two popular Machine Learning models:
1. **Random Forest:** An ensemble of decision trees voting on the outcome.
2. **XGBoost:** A highly optimized gradient-boosting model that learns from its own mistakes sequentially.

**Evaluation:** The system evaluates both models strictly using standard metrics (Accuracy, Precision, Recall, F1-Score, and primarily ROC-AUC). It mathematically chooses the "winner" (based on the highest ROC-AUC score) and saves it to disk (`best_model.pkl`) to be used in the final application.

### Phase 5: The Final Software Output & UI
Finally, everything is wrapped into a simple Gradio Web App. When a user uploads a photo, the system does everything above in real-time, outputting four things:
1. **Visual:** The uploaded image perfectly overlaid with a drawn skeletal map.
2. **Data:** A text readout of the athlete's exact calculated joint angles and symmetry scores.
3. **Prediction:** A `Low Risk` or `High Risk` verdict with the Machine Learning confidence percentage.
4. **Actionable Advice:** If a high-risk mechanic is detected (like an extreme forward fold), the app will output specific text-based advice (e.g., *"Brace your core and maintain a neutral spine to avoid lumbar overload"*).

---

## 3. How to answer common Lecturer Questions:

**Lecturer: "Why did you use Machine Learning if you already had rule-based labels for the data?"**
*Your Answer:* "The rule-based labels are perfect in a vacuum, but in the real world, camera distortions and MediaPipe pose tracking introduce errors. A rigid 'if-then' rule system would fail or crash easily. By injecting simulated noise into my dataset and training Random Forest/XGBoost on it, the Machine Learning model learns to generalize and predict probabilities confidently even when the real-world input image is slightly messy or imperfect."

**Lecturer: "How did you measure if your ML model was actually good?"**
*Your Answer:* "I didn't just use Accuracy, because Accuracy can be misleading if the data is imbalanced. I evaluated the models using Precision, Recall, F1-Score, and primarily the **ROC-AUC score**, which perfectly grades how well the model distinguishes between the High Risk and Low Risk classes. The system automatically plots a Confusion Matrix and dynamically picks the best model based on the ROC-AUC score."

**Lecturer: "What is the practical application of this project?"**
*Your Answer:* "It serves as an accessible, rapid-screening tool for coaches and physios. Without needing expensive motion-capture suits, anyone with a smartphone camera can test an athlete's posture. The system doesn't just act as an alarm bell; by integrating the 'Prevention Suggestions', it immediately gives the user actionable biomechanical feedback to fix their form."
