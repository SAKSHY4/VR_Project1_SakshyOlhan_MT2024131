import cv2
import numpy as np
import pandas as pd
import os

face_cascade_path = '/workspace/MSFD/1/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def load_dataset(csv_path):
    if not os.path.isfile(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, blurred

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def apply_threshold(region):
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def segment_mask_region(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        significant_contours = [c for c in contours[:3] if cv2.contourArea(c) > 100]
        cv2.drawContours(mask, significant_contours, -1, 255, thickness=cv2.FILLED)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

def compute_metrics(segmented, ground_truth):
    # Resize segmented mask to match ground truth if needed.
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

def process_images(input_dir, dataset_csv, sample_count=12000, show_images=False):
    output_dir = os.path.join(os.path.dirname(input_dir), "result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = load_dataset(dataset_csv)
    if dataset is None:
        print("Dataset CSV not found. Exiting.")
        return
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) > sample_count:
        print(f"Processing the first {sample_count} images sequentially")
        image_files = image_files[:sample_count]
    else:
        print(f"Processing all {len(image_files)} images found")
    
    processed_count = 0
    metrics_list = []
    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        
        image, blurred = process_image(image_path)
        if image is None or blurred is None:
            continue
        
        faces = detect_face(image)
        full_mask = np.zeros_like(blurred)  # Create a blank mask of full image size
        
        if len(faces) > 0:
            # Here we choose the first detected face (alternate approach where input image was not cropped for faces).
            (x, y, w, h) = faces[0]
            # Define the lower half of the detected face:
            lower_y = y + h // 2
            lower_half = blurred[lower_y:y+h, x:x+w]
            
            binary_lower = apply_threshold(lower_half)
            mask_lower = segment_mask_region(binary_lower)
            
            # Place the segmented lower-half mask into the corresponding location in the full mask
            full_mask[lower_y:y+h, x:x+w] = mask_lower
        else:
            # If no face is detected, process the entire image as before
            binary = apply_threshold(blurred)
            full_mask = segment_mask_region(binary)
        
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, full_mask)
        
        ground_truth_dir = '/workspace/MSFD/1/face_crop_segmentation'
        ground_truth_path = os.path.join(ground_truth_dir, image_name)

        if os.path.exists(ground_truth_path):
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth is None:
                print(f"Error: Could not read ground truth mask at {ground_truth_path}")
                continue
            ground_truth = (ground_truth > 0).astype(np.uint8)
            segmented = (full_mask > 0).astype(np.uint8)
            
            iou, dice, precision, recall, f1_score = compute_metrics(segmented, ground_truth)
            metrics_list.append([image_name, iou, dice, precision, recall, f1_score])
            
            print(f"{image_name} - IoU: {iou:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
        
        processed_count += 1
        
        if show_images:
            cv2.imshow("Original", image)
            cv2.imshow("Segmented Mask", full_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        metrics_df = pd.DataFrame(metrics_list, columns=["Image", "IoU", "Dice", "Precision", "Recall", "F1-Score"])
    metrics_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)

    average_iou = metrics_df["IoU"].mean()
    average_dice = metrics_df["Dice"].mean()
    average_precision = metrics_df["Precision"].mean()
    average_recall = metrics_df["Recall"].mean()
    average_f1_score = metrics_df["F1-Score"].mean()

    print(f"Average IoU: {average_iou:.4f}")
    print(f"Average Dice: {average_dice:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average F1-Score: {average_f1_score:.4f}")
    
    print(f"Processing completed. {processed_count} images processed.")
    if show_images:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_dir = '/workspace/MSFD/1/face_crop'
    dataset_csv = '/workspace/MSFD/1/dataset.csv'
    
    process_images(input_dir, dataset_csv, sample_count = 100, show_images=False)