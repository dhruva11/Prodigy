import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import zipfile
import sys
import re
from sklearn.preprocessing import LabelEncoder

# Define paths and parameters
dataset_zip = 'leapGestRecog.zip'
extract_dir = 'leapGestRecog/'
image_size = (64, 64)
batch_size = 32
num_epochs = 20
max_images_per_class = 1000  # Limit per class

# Unzip dataset only if directory doesn't exist
def unzip_files(zip_path, extract_to):
    try:
        if os.path.exists(extract_to):
            print(f"Directory {extract_to} already exists, skipping extraction.")
            return extract_to
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file {zip_path} not found.")
        if not zip_path.endswith('.zip'):
            raise ValueError(f"File {zip_path} is not a zip file.")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
        return extract_to
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid zip file: {zip_path}")
    except Exception as e:
        raise Exception(f"Error unzipping {zip_path}: {str(e)}")

# Unzip dataset
try:
    print("Checking for unzipped dataset...")
    extract_path = unzip_files(dataset_zip, extract_dir)
except Exception as e:
    print(f"Failed to unzip dataset: {str(e)}")
    sys.exit(1)

# Check directory
if not os.path.exists(extract_path):
    print(f"Extracted directory {extract_path} not found.")
    sys.exit(1)

# Extract gesture label from subfolder name
def extract_label(subfolder):
    # Extract gesture name after numeric prefix (e.g., '01_palm' -> 'palm')
    match = re.match(r'^\d{2}_([a-zA-Z_]+)$', subfolder)
    if match:
        return match.group(1).lower()
    return None

# Load and preprocess images from nested folders
def load_images(directory):
    images = []
    labels = []
    filenames = []
    class_counts = {}
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path) or not folder.isdigit():
            continue
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            label = extract_label(subfolder)
            if label is None:
                print(f"Warning: Could not extract label from {subfolder}, skipping.")
                continue
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] >= max_images_per_class:
                continue
            for filename in os.listdir(subfolder_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Failed to load {img_path}")
                    continue
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
                filenames.append(os.path.join(folder, subfolder, filename))
                class_counts[label] += 1
    print(f"Loaded {len(images)} images from {directory}")
    print(f"Class counts: {class_counts}")
    return np.array(images), np.array(labels), filenames

# Load all images
try:
    print("Loading images...")
    X_full, y_full, filenames = load_images(extract_path)
    if len(X_full) == 0:
        print("No valid images loaded.")
        sys.exit(1)
except Exception as e:
    print(f"Error loading images: {str(e)}")
    sys.exit(1)

# Encode labels
label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(y_full)
num_classes = len(label_encoder.classes_)
print(f"Number of gesture classes: {num_classes}")

# Reshape images for CNN (add channel dimension)
X_full = X_full.reshape(-1, image_size[0], image_size[1], 1)

# Normalize pixel values
X_full = X_full / 255.0

# Split into train and test sets
X_train_full, X_test, y_train_full, y_test, train_filenames, test_filenames = train_test_split(
    X_full, y_full, filenames, test_size=0.2, random_state=42, stratify=y_full)

# Split training data into train and validation
X_train, X_val, y_train, y_val, train_filenames, val_filenames = train_test_split(
    X_train_full, y_train_full, train_filenames, test_size=0.2, random_state=42, stratify=y_train_full)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
train_datagen.fit(X_train)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=num_epochs,
    callbacks=[early_stopping]
)

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"\nValidation Accuracy: {val_accuracy:.2f}")

# Predict on validation set
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_classes, target_names=label_encoder.classes_))
cm = confusion_matrix(y_val, y_val_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Visualize validation predictions
def visualize_predictions(images, true_labels, pred_labels, filenames, class_names, num_samples=5):
    plt.figure(figsize=(15, 4))
    indices = np.random.choice(len(images), num_samples, replace=False)
    for i, idx in enumerate(indices):
        img = images[idx].reshape(image_size)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nVisualizing validation predictions...")
visualize_predictions(X_val, y_val, y_val_pred_classes, val_filenames, label_encoder.classes_)

# Visualize misclassifications
def visualize_misclassified_images(images, true_labels, pred_labels, filenames, class_names, num_samples=5):
    misclassified = np.where(true_labels != pred_labels)[0]
    if len(misclassified) == 0:
        print("No misclassifications found!")
        return
    num_samples = min(num_samples, len(misclassified))
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(np.random.choice(misclassified, num_samples, replace=False)):
        img = images[idx].reshape(image_size)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nVisualizing validation misclassified images...")
visualize_misclassified_images(X_val, y_val, y_val_pred_classes, val_filenames, label_encoder.classes_)

# Save the model
model.save('hand_gesture_classifier.h5')
print("\nModel saved to hand_gesture_classifier.h5")

# Predict on test set
try:
    print("\nGenerating test predictions...")
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    test_predictions = label_encoder.inverse_transform(y_test_pred_classes)
    submission = pd.DataFrame({'id': range(1, len(test_predictions) + 1), 'label': test_predictions})
    submission.to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to test_predictions.csv")

    # Visualize test predictions
    def visualize_test_predictions(images, pred_labels, filenames, class_names, num_samples=5):
        plt.figure(figsize=(15, 4))
        indices = np.random.choice(len(images), num_samples, replace=False)
        for i, idx in enumerate(indices):
            img = images[idx].reshape(image_size)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Pred: {class_names[pred_labels[idx]]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    print("\nVisualizing sample test predictions...")
    visualize_test_predictions(X_test, y_test_pred_classes, test_filenames, label_encoder.classes_)

except Exception as e:
    print(f"Error processing test set: {str(e)}")
    sys.exit(1)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# End program
print("\nAll tasks completed successfully. Exiting program.")
sys.exit(0)