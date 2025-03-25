"""
Test Script for U-Net Segmentation Evaluation

This script loads a U-Net model architecture, loads saved weights,
runs predictions on images in a face crop directory, and compares them 
with ground truth masks using various evaluation metrics.
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128
FACE_CROP_DIR = '/workspace/MSFD/1/face_crop'
GROUND_TRUTH_DIR = '/workspace/MSFD/1/face_crop_segmentation'
OUTPUT_DIR = '/workspace/MSFD/2/unet_test_results'
WEIGHTS_PATH = '/workspace/MSFD/2/models/unet_weights_20250325_164438.h5'  # Update if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_unet_model():
    """Build the U-Net model architecture."""
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder (Contracting Path)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bridge
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = Dropout(0.2)(c4)
    
    # Decoder (Expansive Path)
    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-3), 
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])
    
    return model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize
    img = img.astype('float32') / 255.0
    return img

def compute_metrics(segmented, ground_truth):
    # Ensure both masks are the same size
    if segmented.shape != ground_truth.shape:
        segmented = cv2.resize(segmented, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    segmented = segmented.astype(bool)
    ground_truth = ground_truth.astype(bool)
    
    intersection = np.logical_and(segmented, ground_truth).sum()
    union = np.logical_or(segmented, ground_truth).sum()
    iou = intersection / union if union != 0 else 0
    
    dice = (2 * intersection) / (segmented.sum() + ground_truth.sum()) if (segmented.sum() + ground_truth.sum()) != 0 else 0
    precision = intersection / segmented.sum() if segmented.sum() != 0 else 0
    recall = intersection / ground_truth.sum() if ground_truth.sum() != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return iou, dice, precision, recall, f1_score

model = build_unet_model()
model.load_weights(WEIGHTS_PATH)
print("U-Net model loaded with weights.")

image_files = sorted([f for f in os.listdir(FACE_CROP_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
metrics_list = []

for image_name in image_files:
    image_path = os.path.join(FACE_CROP_DIR, image_name)
    gt_path = os.path.join(GROUND_TRUTH_DIR, image_name)
    
    img = preprocess_image(image_path)
    if img is None:
        continue
    
    # Predict segmentation mask; add batch dimension then remove it later
    pred_mask = model.predict(np.expand_dims(img, axis=0))[0, :, :, 0]
    # Convert to binary mask using threshold 0.5
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, image_name), pred_mask)
    
    if not os.path.exists(gt_path):
        print(f"Ground truth for {image_name} not found. Skipping metrics.")
        continue
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Error reading ground truth mask for {image_name}.")
        continue
    gt_mask = (gt_mask > 0).astype(np.uint8) * 255
    
    iou, dice, precision, recall, f1_score = compute_metrics(pred_mask, gt_mask)
    metrics_list.append([image_name, iou, dice, precision, recall, f1_score])
    
    print(f"{image_name} - IoU: {iou:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

metrics_df = pd.DataFrame(metrics_list, columns=["Image", "IoU", "Dice", "Precision", "Recall", "F1-Score"])
metrics_csv_path = os.path.join(OUTPUT_DIR, "unet_evaluation_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

avg_iou = metrics_df["IoU"].mean()
avg_dice = metrics_df["Dice"].mean()
avg_precision = metrics_df["Precision"].mean()
avg_recall = metrics_df["Recall"].mean()
avg_f1 = metrics_df["F1-Score"].mean()

print("\n--- Average Metrics ---")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice: {avg_dice:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1-Score: {avg_f1:.4f}")

print(f"\nProcessing complete. Evaluation metrics saved to {metrics_csv_path}")
