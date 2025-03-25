import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, ConfusionMatrixDisplay

# Define image dimensions
img_height, img_width = 128, 128

def create_cnn_model(learning_rate=1e-3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Path to the saved model weights (.h5 file)
weights_path = '/workspace/Classification/best_model.h5'
# Directory containing test images (with subfolders "with_mask" and "without_mask")
test_dir = '/workspace/dataset'

# Create the model and load the weights
model = create_cnn_model()
model.load_weights(weights_path)
print("Model weights loaded.")

# Create a test data generator that rescales images
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Ensure order for predictions aligns with ground truth labels
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions on the test dataset
pred_probs = model.predict(test_generator)
# Convert probabilities to binary class predictions using threshold 0.5
pred_classes = (pred_probs > 0.5).astype(int).reshape(-1)

# Ground truth labels from the generator
true_classes = test_generator.classes

# Generate a classification report
report = classification_report(true_classes, pred_classes, target_names=list(test_generator.class_indices.keys()))
print("Classification Report:")
print(report)

# Compute Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
print("Confusion Matrix:")
print(cm)
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Compute AUC Score
auc_score = roc_auc_score(true_classes, pred_probs)
print(f"AUC Score: {auc_score:.4f}")