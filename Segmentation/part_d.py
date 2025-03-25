#!/usr/bin/env python3
"""
U-Net Mask Segmentation Implementation
- Train a U-Net model for precise segmentation of mask regions
- Compare with traditional segmentation methods
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
import datetime

# GPU configuration for local machine (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )
        print("GPU is set to utilize a full 2GB of memory.")
    except Exception as e:
        print("Error in GPU configuration:", e)
else:
    print("GPU not detected")

# Set paths to dataset directories
DATASET_DIR = '/workspace/MSFD'
DATASET_CSV = '/workspace/MSFD/1/dataset.csv'
FACE_CROP_DIR = '/workspace/MSFD/1/face_crop'
SEGMENTATION_DIR = '/workspace/MSFD/1/face_crop_segmentation'

# Create output directory for results
OUTPUT_DIR = '/workspace/MSFD/2/output'
MODEL_DIR = '/workspace/MSFD/2/models'
RESULTS_DIR = '/workspace/MSFD/2/results'
PLOTS_DIR = '/workspace/MSFD/2/plots'

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set parameters
IMG_SIZE = 128  # Resize images to this size for U-Net
BATCH_SIZE = 32 # Batch size for training
EPOCHS = 5     # Number of epochs

# Pre-load segmentation filenames for faster lookup
all_segmentation_files = os.listdir(SEGMENTATION_DIR)
segmentation_dict = {file: file for file in all_segmentation_files}  # Dictionary for fast lookup

def create_dataset():
    """Create train/test datasets from the MSFD dataset"""
    # Find face crop images that have corresponding segmentation ground truth
    face_crops = glob.glob(os.path.join(FACE_CROP_DIR, '*.jpg'))
    valid_samples = []

    print(f"Found {len(face_crops)} face crop images.")

    # Filter for samples that have corresponding segmentation masks
    for idx, face_path in enumerate(face_crops, start=1):
        face_id = os.path.basename(face_path).split('.')[0]
        # Use segmentation_dict for faster lookup instead of glob.glob()
        segmentation_paths = [file for file in segmentation_dict if file.startswith(face_id) and file.endswith('.jpg')]

        if segmentation_paths:
            valid_samples.append((face_path, os.path.join(SEGMENTATION_DIR, segmentation_paths[0])))

        # Log progress every 100 images or on the last image
        if idx % 100 == 0 or idx == len(face_crops):
            print(f"Processing image {idx}/{len(face_crops)}: {face_id}")

    print(f"Using {len(valid_samples)} valid face-mask pairs.")

    # Split into train and test sets
    train_samples, val_samples = train_test_split(valid_samples, test_size=0.2, random_state=42)

    print(f"Training set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")

    return train_samples, val_samples

def preprocess_image(image_path, mask_path):
    """Preprocess image and mask for training"""
    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize to the target size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    # Normalize the image (0-1 range)
    image = image / 255.0

    # Binarize the mask
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # Expand mask dimensions for model input
    mask = np.expand_dims(mask, axis=-1)

    return image, mask

def data_generator(samples, batch_size=BATCH_SIZE, augment=False):
    """Generate batches of data for training"""
    num_samples = len(samples)
    while True:
        # Shuffle samples each epoch
        np.random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            masks = []

            for image_path, mask_path in batch_samples:
                image, mask = preprocess_image(image_path, mask_path)

                # Data augmentation (if enabled)
                if augment:
                    # Random horizontal flip
                    if np.random.random() > 0.5:
                        image = np.fliplr(image)
                        mask = np.fliplr(mask)

                    # Random rotation (±15 degrees)
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
                    image = cv2.warpAffine(image, M, (IMG_SIZE, IMG_SIZE))
                    mask = cv2.warpAffine(mask.astype(np.float32), M, (IMG_SIZE, IMG_SIZE))
                    mask = np.round(mask).astype(np.uint8)

                images.append(image)
                masks.append(mask)

            yield np.array(images), np.array(masks)

def build_unet_model():
    """Build the U-Net model architecture"""
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

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    # Create model
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])

    model.summary()  # Print model summary

    return model

def calculate_metrics(true_mask, pred_mask, threshold=0.5):
    """Calculate IoU and Dice Score for a prediction"""
    # Threshold the prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = true_mask.astype(np.uint8)

    # Calculate IoU
    intersection = np.logical_and(true_binary, pred_binary).sum()
    union = np.logical_or(true_binary, pred_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    # Calculate Dice Score
    dice = (2 * intersection) / (true_binary.sum() + pred_binary.sum()) if (true_binary.sum() + pred_binary.sum()) > 0 else 0.0

    return iou, dice

def visualize_results(model, validation_samples, num_samples=10):
    """Visualize predictions compared to ground truth"""
    # Select a few samples randomly
    if len(validation_samples) > num_samples:
        indices = np.random.choice(len(validation_samples), num_samples, replace=False)
        test_samples = [validation_samples[i] for i in indices]
    else:
        test_samples = validation_samples

    results = []

    # Create a figure to display results
    for i, (image_path, mask_path) in enumerate(test_samples):
        # Preprocess image
        image, mask = preprocess_image(image_path, mask_path)

        # Get prediction
        pred = model.predict(np.expand_dims(image, axis=0))[0]

        # Calculate metrics
        iou, dice = calculate_metrics(mask, pred)

        # Save results
        results.append({
            'image_id': os.path.basename(image_path).split('_')[0],
            'face_id': os.path.basename(image_path).split('.')[0],
            'iou': float(iou),
            'dice': float(dice)
        })

        # Visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.title(f'Prediction (IoU: {iou:.3f})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'prediction_{i}.png'))
        plt.close()

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'unet_predictions.csv'), index=False)

    # Print average metrics
    print("\nAverage Metrics:")
    print(f"IoU: {df['iou'].mean():.4f}")
    print(f"Dice Score: {df['dice'].mean():.4f}")

def compare_with_traditional(unet_results):
    """Compare U-Net results with traditional segmentation methods"""
    # Load results from the traditional method
    traditional_results_path = os.path.join('improved_mask_segmentation_results', 'segmentation_metrics.csv')
    if not os.path.exists(traditional_results_path):
        print("Traditional segmentation results not found. Run improved_mask_segmentation.py first.")
        return

    traditional_results = pd.read_csv(traditional_results_path)
    # Merge results for comparison
    merged_results = pd.merge(unet_results, traditional_results, on='image_id', how='inner')

    # Calculate average metrics
    avg_unet_iou = unet_results['iou_unet'].mean()
    avg_unet_dice = unet_results['dice_unet'].mean()

    # Comparison plot
    if not merged_results.empty:
        comparison_data = merged_results[['image_id', 'iou_unet', 'iou_improved']].copy()
        comparison_melt = pd.melt(comparison_data, id_vars=['image_id'],
                                  value_vars=['iou_unet', 'iou_improved'],
                                  var_name='Method', value_name='IoU')
        plt.figure(figsize=(12, 8))
        import seaborn as sns
        sns.barplot(x='image_id', y='IoU', hue='Method', data=comparison_melt)
        plt.title('IoU Comparison: U-Net vs Traditional Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'unet_vs_traditional_comparison.png'))
        plt.close()

        avg_traditional_iou = merged_results['iou_improved'].mean()
        improvement = ((avg_unet_iou - avg_traditional_iou) / avg_traditional_iou) * 100

        print("\nComparison Results:")
        print(f"Average U-Net IoU: {avg_unet_iou:.4f}")
        print(f"Average U-Net Dice Score: {avg_unet_dice:.4f}")
        print(f"Average Traditional IoU: {avg_traditional_iou:.4f}")
        print(f"Improvement with U-Net: {improvement:.2f}%")

        comparison_summary = {
            'Method': ['U-Net', 'Traditional (Improved)'],
            'Average IoU': [avg_unet_iou, avg_traditional_iou],
            'Improvement (%)': [improvement, 0.0]
        }
        pd.DataFrame(comparison_summary).to_csv(
            os.path.join(OUTPUT_DIR, 'comparison_summary.csv'), index=False)
    else:
        print("No matching samples for comparison.")

def train_unet_model():
    """Train the U-Net model and save results"""
    # Create datasets
    train_samples, val_samples = create_dataset()

    # Create data generators
    train_generator = data_generator(train_samples, batch_size=BATCH_SIZE, augment=True)
    val_generator = data_generator(val_samples, batch_size=BATCH_SIZE, augment=False)

    # Build and compile model
    model = build_unet_model()

    # Create callbacks with ModelCheckpoint saving only the best weights
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'unet_weights_{timestamp}.h5')
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_samples) // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=len(val_samples) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    history_path = os.path.join(OUTPUT_DIR, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        import json
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f, indent=4)

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'training_history_{timestamp}.png'))
    plt.close()

    # Visualize results
    visualize_results(model, val_samples)

    # Compare with traditional methods
    compare_with_traditional(os.path.join(RESULTS_DIR, 'unet_predictions.csv'))

    print(f"\nTraining complete! Results saved to: {OUTPUT_DIR}")
    print(f"Best weights saved to: {model_path}")
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    print("=== U-Net Mask Segmentation Implementation ===")
    # Train U-Net model
    train_unet_model()