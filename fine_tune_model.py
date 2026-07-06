import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# ============ YOUR DATA PATHS ============
DATA_ROOT = r"S:\Python\TrustMediaBackend\data"
CASIA2_DIR = os.path.join(DATA_ROOT, "CASIA2")
FFPP_DIR = os.path.join(DATA_ROOT, "ffpp_processed")
TEST_MATERIAL = r"S:\Python\TrustMediaBackend\DeepfakeDetector\Testing_Material"
MODEL_SAVE_PATH = r"S:\Python\TrustMediaBackend\DeepfakeDetector\models\fine_tuned_model.h5"

os.makedirs(r"S:\Python\TrustMediaBackend\DeepfakeDetector\models", exist_ok=True)

print(f"CASIA2 exists: {os.path.exists(CASIA2_DIR)}")
print(f"FFPP exists: {os.path.exists(FFPP_DIR)}")
print(f"Testing_Material exists: {os.path.exists(TEST_MATERIAL)}")


# ============ LOAD IMAGES FROM DATASETS ============
def load_images_from_path(folder_path, label, max_images=500):
    """Load images from a folder"""
    images = []
    labels = []
    count = 0

    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} not found")
        return images, labels

    # Recursively find all images
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if count >= max_images:
                    return images, labels

                file_path = os.path.join(root, file)
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        file_path,
                        target_size=(224, 224)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(label)
                    count += 1

                    if count % 50 == 0:
                        print(f"  Loaded {count} images from {os.path.basename(folder_path)}")
                except Exception as e:
                    print(f"  Error loading {file_path}: {str(e)[:50]}")

    return images, labels


# ============ LOAD DATA ============
print("\n=== LOADING TRAINING DATA ===")
print("Loading CASIA2 (fake faces)...")
casia_images, casia_labels = load_images_from_path(CASIA2_DIR, label=1, max_images=300)

print(f"Loading FFPP (fake videos/frames)...")
ffpp_images, ffpp_labels = load_images_from_path(FFPP_DIR, label=1, max_images=300)

# Combine fake data
fake_images = casia_images + ffpp_images
fake_labels = casia_labels + ffpp_labels

print(f"\nTotal fake images collected: {len(fake_images)}")

# For real images, load from Test_Material
print(f"\nLoading real images from Testing_Material...")
real_images, real_labels = load_images_from_path(TEST_MATERIAL, label=0, max_images=150)

print(f"Total real images collected: {len(real_images)}")

# Combine all
all_images = real_images + fake_images
all_labels = real_labels + fake_labels

if len(all_images) == 0:
    print("\nERROR: No images loaded!")
    print(f"Check if these paths exist:")
    print(f"  {CASIA2_DIR}")
    print(f"  {FFPP_DIR}")
    print(f"  {TEST_MATERIAL}")
    exit(1)

X = np.array(all_images)
y = np.array(all_labels)

print(f"\n=== DATASET SUMMARY ===")
print(f"Total images: {len(X)}")
print(f"Real images: {np.sum(y == 0)}")
print(f"Fake images: {np.sum(y == 1)}")
print(f"Image shape: {X[0].shape}")

# ============ SPLIT DATA ============
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print(f"  Train real: {np.sum(y_train == 0)}, fake: {np.sum(y_train == 1)}")
print(f"  Val real: {np.sum(y_val == 0)}, fake: {np.sum(y_val == 1)}")

# ============ BUILD MODEL ============
print("\n=== BUILDING MODEL ===")
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze early layers, unfreeze later ones for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(model.summary())

# ============ TRAIN MODEL ============
print("\n=== TRAINING MODEL ===")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
)

# ============ SAVE MODEL ============
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

# ============ PLOT RESULTS ============
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.title('Model AUC')
plt.grid(True)

plt.tight_layout()
plt.savefig(r"S:\Python\TrustMediaBackend\DeepfakeDetector\training_history.png", dpi=100)
print("✓ Training history saved to training_history.png")
plt.show()

# ============ EVALUATE ============
print("\n=== VALIDATION RESULTS ===")
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

# Test predictions
print("\n=== SAMPLE PREDICTIONS ===")
test_predictions = model.predict(X_val[:10])
for i, pred in enumerate(test_predictions[:10]):
    actual = "FAKE" if y_val[i] == 1 else "REAL"
    predicted = "FAKE" if pred[0] > 0.5 else "REAL"
    confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
    print(f"Image {i}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.2%}")
