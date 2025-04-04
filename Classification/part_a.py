import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Global parameters for LBP
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    feature = hog(
        image,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        channel_axis=-1
    )
    return feature

def extract_lbp_features(image, n_points=N_POINTS, radius=RADIUS, method=METHOD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def load_images_and_labels(dataset_path):
    features = []
    labels = []
    total_files = 0
    skipped_files = 0
    classes = ['with_mask', 'without_mask']

    for label in classes:
        class_path = os.path.join(dataset_path, label)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_files += 1
                img_path = os.path.join(class_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Unable to read image at {img_path}. Skipping.")
                    skipped_files += 1
                    continue

                image = cv2.resize(image, (128, 128))

                hog_features = extract_hog_features(image)
                lbp_features = extract_lbp_features(image)
                combined_features = np.concatenate([hog_features, lbp_features])
                features.append(combined_features)
                labels.append(label)

    print(f"Total image files found: {total_files}")
    print(f"Total image files successfully read: {total_files - skipped_files}")
    print(f"Total image files skipped: {skipped_files}")
    
    return np.array(features), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=['with_mask', 'without_mask'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['with_mask', 'without_mask'],
                yticklabels=['with_mask', 'without_mask'])
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    dataset_dir = '/workspace/dataset'
    X, y = load_images_and_labels(dataset_dir)
    print("Feature matrix shape:", X.shape)
    print("Labels array shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Number of training samples:", X_train.shape[0])
    print("Number of testing samples:", X_test.shape[0])

    svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print("\nSVM Accuracy: {:.2f}%".format(svm_accuracy * 100))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,),
                                   max_iter=300,
                                   random_state=42)
    mlp_classifier.fit(X_train, y_train)
    y_pred_mlp = mlp_classifier.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
    print("\nMLP Accuracy: {:.2f}%".format(mlp_accuracy * 100))
    print("MLP Classification Report:\n", classification_report(y_test, y_pred_mlp))

    plot_confusion_matrix(y_test, y_pred_svm, 'SVM Confusion Matrix')
    plot_confusion_matrix(y_test, y_pred_mlp, 'MLP Confusion Matrix')

if __name__ == "__main__":
    main()