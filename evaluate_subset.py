import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import os

# ------------------------------
# Configuration
# ------------------------------
test_dir = "dataset_subset/test"   # ✅ updated path
batch_size = 32
model_path = "models/asl_subset_mobilenet.h5"      # ✅ correct TensorFlow model path
class_indices_path = "models/class_indices.json"
img_size = (160, 160)  # ✅ matches training script

# ------------------------------
# Data generator (matches training preprocessing)
# ------------------------------
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ------------------------------
# Load Test Dataset
# ------------------------------
test_ds = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Important for evaluation
)

# ------------------------------
# Load Model and Class Indices
# ------------------------------
print("Loading model...")
model = tf.keras.models.load_model(model_path)

# Load class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Create inverse mapping for class names
inv_class_indices = {v: k for k, v in class_indices.items()}
class_names = [inv_class_indices[i] for i in range(len(inv_class_indices))]

print(f"Model loaded. Classes: {class_names}")
print(f"Found {test_ds.samples} test images in {len(test_ds.class_indices)} classes")

# ------------------------------
# Evaluation
# ------------------------------
print("Evaluating...")

# Get predictions and true labels
y_pred = []
y_true = []

# Make sure we get all samples
steps = np.ceil(test_ds.samples / test_ds.batch_size)

for i in range(int(steps)):
    batch_x, batch_y = next(test_ds)
    preds = model.predict(batch_x, verbose=0)
    
    # Convert predictions to class indices
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(batch_y, axis=1)
    
    y_pred.extend(pred_classes)
    y_true.extend(true_classes)

# Trim to exact sample count (in case last batch was padded)
y_pred = y_pred[:test_ds.samples]
y_true = y_true[:test_ds.samples]

# ------------------------------
# Results
# ------------------------------
accuracy = np.mean(np.array(y_pred) == np.array(y_true)) * 100

print(f"\n✅ Evaluation on {test_dir}:")
print(f"Total samples: {len(y_true)}")
print(f"Accuracy: {accuracy:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print("Classes:", class_names)
print(cm)
