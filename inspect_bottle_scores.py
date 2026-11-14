
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from ml_engine.tf_inference import analyze_image, load_thresholds

test_dir = Path("datasets/MVTecAD/bottle/test")
thresholds = load_thresholds("bottle")
current_threshold = thresholds["image_threshold"]

normal_scores = []
anomaly_scores = []

for defect_dir in sorted(test_dir.iterdir()):
    if not defect_dir.is_dir():
        continue

    for image_path in sorted(defect_dir.glob("*.png")):
        result = analyze_image(image_path, "bottle")
        score = result["anomaly_score"]

        if defect_dir.name == "good":
            normal_scores.append(score)
        else:
            anomaly_scores.append(score)

normal_scores = np.array(normal_scores)
anomaly_scores = np.array(anomaly_scores)

y_true = np.concatenate([
    np.zeros(len(normal_scores)),
    np.ones(len(anomaly_scores))
])

y_scores = np.concatenate([
    normal_scores,
    anomaly_scores
])

print("\nNORMAL IMAGE SCORES")
print("Count:", len(normal_scores))
print("Minimum:", normal_scores.min())
print("Maximum:", normal_scores.max())
print("Mean:", normal_scores.mean())

print("\nANOMALOUS IMAGE SCORES")
print("Count:", len(anomaly_scores))
print("Minimum:", anomaly_scores.min())
print("Maximum:", anomaly_scores.max())
print("Mean:", anomaly_scores.mean())

print("\nCURRENT THRESHOLD:", current_threshold)
print("Normal images above threshold:",
      np.sum(normal_scores > current_threshold))
print("Anomalous images below threshold:",
      np.sum(anomaly_scores <= current_threshold))

print("\nAUROC:", roc_auc_score(y_true, y_scores))

fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

youden_index = tpr - fpr
best_index = np.argmax(youden_index)

print("\nDIAGNOSTIC ROC OPERATING POINT")
print("Youden threshold:", roc_thresholds[best_index])
print("TPR:", tpr[best_index])
print("FPR:", fpr[best_index])

print("\nNote: This ROC threshold is diagnostic only.")
print("Do not use it as a production threshold.")
