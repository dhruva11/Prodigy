import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import zipfile
from skimage.feature import hog
from skimage.transform import rotate
import sys

# Define paths and parameters
train_zip = 'train.zip'
test_zip = 'test.zip'
extract_dir = 'dataset/'
train_dir = os.path.join(extract_dir, 'train/')
test_dir = os.path.join(extract_dir, 'test/')
image_size = (64, 64)
max_images_per_class = 1000
hog_pixels_per_cell = (4, 4)  # Smaller cells for finer details
hog_cells_per_block = (2, 2)

# Unzip files only if directories don't exist
def unzip_files(zip_path, extract_to, target_dir):
    try:
        if os.path.exists(target_dir):
            print(f"{target_dir} already exists, skipping extraction.")
            return target_dir
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file {zip_path} not found.")
        if not zip_path.endswith('.zip'):
            raise ValueError(f"File {zip_path} is not a zip file.")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
        return os.path.join(extract_to, os.path.basename(zip_path).replace('.zip', ''))
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid zip file: {zip_path}")
    except Exception as e:
        raise Exception(f"Error unzipping {zip_path}: {str(e)}")

# Unzip files if necessary
try:
    print("Checking for unzipped datasets...")
    unzip_files(train_zip, extract_dir, train_dir)
    unzip_files(test_zip, extract_dir, test_dir)
except Exception as e:
    raise Exception(f"Failed to unzip files: {str(e)}")

# Check directories
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory {train_dir} not found.")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory {test_dir} not found.")

# Extract HOG features with augmentation
def extract_hog_features(img, augment=False):
    img = cv2.resize(img, image_size)
    if augment:
        # Random rotation up to 15 degrees
        angle = np.random.uniform(-15, 15)
        img = rotate(img, angle, mode='edge')
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img)
    fd = hog(img, pixels_per_cell=hog_pixels_per_cell, cells_per_block=hog_cells_per_block,
             block_norm='L2-Hys', visualize=False)
    return fd

# Load training images
def load_train_images():
    images = []
    labels = []
    filenames = []
    cat_count = 0
    dog_count = 0
    for filename in os.listdir(train_dir):
        if filename.startswith('cat') and cat_count < max_images_per_class:
            label = 0
            cat_count += 1
        elif filename.startswith('dog') and dog_count < max_images_per_class:
            label = 1
            dog_count += 1
        else:
            continue
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        fd = extract_hog_features(img, augment=True)
        images.append(fd)
        labels.append(label)
        filenames.append(filename)
    print(f"Loaded {cat_count} cat images and {dog_count} dog images")
    return np.array(images), np.array(labels), filenames

# Load test images
def load_test_images():
    images = []
    ids = []
    filenames = []
    for filename in sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0])):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        fd = extract_hog_features(img)
        images.append(fd)
        ids.append(int(filename.split('.')[0]))
        filenames.append(filename)
    return np.array(images), ids, filenames

# Load training data
try:
    print("Loading training data...")
    X_train_full, y_train_full, train_filenames = load_train_images()
    if len(X_train_full) == 0:
        raise ValueError("No valid training images loaded.")
    print(f"Loaded {len(X_train_full)} training images")
except Exception as e:
    print(f"Error loading training data: {e}")
    sys.exit(1)

# Normalize features
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)

# Split training data
X_train, X_val, y_train, y_val, train_filenames_train, train_filenames_val = train_test_split(
    X_train_full, y_train_full, train_filenames, test_size=0.2, random_state=42)

# Hyperparameter tuning
print("Performing grid search for SVM...")
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm = SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Train final model
svm = grid_search.best_estimator_
svm.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Cat', 'Dog']))

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualize validation predictions
def visualize_validation_predictions(X_val, y_val, y_val_pred, filenames, num_samples=5):
    plt.figure(figsize=(15, 5))
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    for i, idx in enumerate(indices):
        img_path = os.path.join(train_dir, filenames[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {'Cat' if y_val[idx] == 0 else 'Dog'}\nPred: {'Cat' if y_val_pred[idx] == 0 else 'Dog'}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nVisualizing sample validation predictions...")
visualize_validation_predictions(X_val, y_val, y_val_pred, train_filenames_val)

# Visualize misclassifications
def visualize_misclassified_images(X_val, y_val, y_val_pred, filenames, max_samples=5):
    misclassified_indices = np.where(y_val != y_val_pred)[0]
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    num_samples = min(max_samples, len(misclassified_indices))
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(np.random.choice(misclassified_indices, num_samples, replace=False)):
        img_path = os.path.join(train_dir, filenames[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {'Cat' if y_val[idx] == 0 else 'Dog'}\nPred: {'Cat' if y_val_pred[idx] == 0 else 'Dog'}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nVisualizing misclassified validation images...")
visualize_misclassified_images(X_val, y_val, y_val_pred, train_filenames_val)

# Save the model
with open('svm_cat_dog_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
print("\nModel saved to svm_cat_dog_model.pkl")

# Predict on test set
try:
    print("\nLoading test data...")
    X_test, test_ids, test_filenames = load_test_images()
    if len(X_test) == 0:
        raise ValueError("No valid test images loaded.")
    X_test = scaler.transform(X_test)
    test_predictions = svm.predict(X_test)

    # Save test predictions
    submission = pd.DataFrame({'id': test_ids, 'label': test_predictions})
    submission.to_csv('submission.csv', index=False)
    print("Test predictions saved to submission.csv")

    # Visualize test predictions
    def visualize_test_predictions(filenames, predictions, num_samples=5):
        plt.figure(figsize=(15, 5))
        indices = np.random.choice(len(filenames), num_samples, replace=False)
        for i, idx in enumerate(indices):
            img_path = os.path.join(test_dir, filenames[idx])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Pred: {'Cat' if predictions[idx] == 0 else 'Dog'}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    print("\nVisualizing sample test predictions...")
    visualize_test_predictions(test_filenames, test_predictions)

except Exception as e:
    print(f"Error processing test data: {e}")
    sys.exit(1)

# End program
print("\nAll tasks completed successfully. Exiting program.")
sys.exit(0)