# Project Report: Mask Detection & Segmentation

## i. Introduction
This project aims to develop robust image processing solutions for mask detection and segmentation using both traditional and deep learning techniques. The primary objectives are to:
- Classify images of subjects with and without masks.
- Segment the mask regions from facial images accurately.
- Evaluate the performance of various approaches using metrics such as accuracy, IoU, Dice score, Precision, Recall, F1-score, and AUC.

## ii. Dataset
- **Classification Dataset:**  
  - **Total Images:** 4092 images  
  - **Structure:** Organized into two folders (`with_mask` and `without_mask`) with corresponding labels.
  
- **Segmentation Dataset:**  
  - **Training:** 9383 face images with masks extracted from 8226 source images, with provided ground truth masks.
  - **Testing:** 8932 images without ground truth for additional evaluation.

## iii. Methodology

### Task 1: Traditional ML & Neural Network Classification (File: `part_a`)
- **Feature Extraction:**  
  Images are preprocessed to extract a feature matrix (4092 × 8126) and associated label array.
- **Models Used:**  
  - **SVM:** Evaluated with various kernels (e.g., RBF, Sigmoid, Linear) to achieve optimal performance.
  - **MLP:** A Multi-Layer Perceptron is implemented using scikit-learn's `MLPClassifier` with one hidden layer of 100 neurons, trained for 300 iterations. This model learns non-linear patterns from the features and achieved an accuracy of 91.70%.

### Task 2: CNN Classification (File: `part_b`)
- **Dataset:**  
  Uses the 4092-image classification dataset.
- **Model Architecture:**  
  A deep Convolutional Neural Network (CNN) was designed with three convolutional blocks. Each block comprises convolutional layers with ReLU activations, batch normalization, and max pooling. After flattening, a dense layer with dropout is applied, followed by a sigmoid output layer for binary classification.
- **Hyperparameters:**  
  - Input image size: 128 × 128 × 3  
  - Optimizer: Adam with a learning rate of 1e-3  
  - Batch size: 16  
  - Training for approximately 10 epochs.
- **Performance:**  
  - **Accuracy:** 95%
  - **AUC:** 0.9932  
  

### Task 3: Traditional Segmentation (File: `part_c`)
- **Workflow:**  
  1. **Face Detection:** Faces are detected using a Haar cascade classifier.
  2. **Region Selection:** The lower half of each detected face is extracted.
  3. **Thresholding:** Otsu's method is applied to threshold the lower half.
  4. **Contour Detection & Morphology:** Contours are extracted and refined using morphological operations (closing and opening) to generate the final mask.
- **Performance:**  
  - **Average IoU:** 0.3673  
  - **Average Dice:** 0.4941  
  - **Average Precision:** 0.5156  
  - **Average Recall:** 0.5896  
  - **Average F1-Score:** 0.4941  

### Task 4: Deep Learning Segmentation (U-Net) (File: `part_d`)
- **Dataset:**  
  Uses the segmentation dataset with 9383 training images (with ground truth) and 8932 testing images.
- **U-Net Model Architecture:**  
  The U-Net model features a contracting path (encoder) to capture context and an expansive path (decoder) that enables precise localization via skip connections. The model is designed to output a probability map that is thresholded to produce binary masks.
- **Hyperparameters:**  
  - Input image size: 128 × 128 × 3  
  - Optimizer: Adam with a learning rate of 1e-3  
  - Loss function: Binary crossentropy  
  - Dropout of 0.2 in the bottleneck to reduce overfitting.
- **Performance (Average over Segmentation Dataset):**  
  - **Average IoU:** 0.8243  
  - **Average Dice:** 0.8980  
  - **Average Precision:** 0.9766  
  - **Average Recall:** 0.8393  
  - **Average F1-Score:** 0.8980  
  - Evaluation metrics are saved to `/workspace/MSFD/2/unet_test_results/unet_evaluation_metrics.csv`.

## iv. Hyperparameters and Experiments
- **Task 1:**  
  - **MLP:** Hidden layer of 100 neurons; max_iter of 300; random_state of 42.  
  - **SVM:** Various kernels were tested, with the RBF kernel achieving the best performance.
- **Task 2 (CNN):**  
  - Optimizer: Adam at 1e-3, batch size of 16, ~10 epochs.  
  - Experiments included testing different cnn architectures, optimizers and learning rates.
- **Task 4 (U-Net):**  
  - Dropout rate of 0.2 in the bridge section.  
  - Binary crossentropy loss with accuracy and IoU as evaluation metrics.
  - Experimented with various network depths and hyperparameters to maximize segmentation performance.

## v. Results
### Task 1: Traditional ML & Neural Network Classification
- **SVM Classifier:**
  - **Accuracy:** 94.14%
  - **Classification Report:**
    ```
                   precision    recall  f1-score   support

       with_mask       0.95      0.94      0.95       461
    without_mask       0.93      0.94      0.93       358

        accuracy                           0.94       819
       macro avg       0.94      0.94      0.94       819
    weighted avg       0.94      0.94      0.94       819
    ```
- **MLP Classifier:**
  - **Accuracy:** 91.70%
  - **Classification Report:**
    ```
                   precision    recall  f1-score   support

       with_mask       0.93      0.92      0.93       461
    without_mask       0.90      0.91      0.91       358

        accuracy                           0.92       819
       macro avg       0.92      0.92      0.92       819
    weighted avg       0.92      0.92      0.92       819
    ```

### Task 2: CNN Classification
- **Performance:**  
  - **Accuracy:** 95%  
  - **AUC:** 0.9932  
  - Detailed metrics:
    ```
                  precision    recall  f1-score   support

       with_mask       0.92      0.99      0.95      2162
    without_mask       0.98      0.91      0.94      1930

         accuracy                           0.95      4092
        macro avg       0.95      0.95      0.95      4092
     weighted avg       0.95      0.95      0.95      4092
    ```
  - **Confusion Matrix:**
    ```
    [[2133   29]
     [ 176 1754]]
    ```

### Task 3: Traditional Segmentation
- **Average Metrics:**
  - **IoU:** 0.3673  
  - **Dice:** 0.4941  
  - **Precision:** 0.5156  
  - **Recall:** 0.5896  
  - **F1-Score:** 0.4941

### Task 4: Deep Learning Segmentation (U-Net)
- **Average Metrics:**
  - **Average IoU:** 0.8243  
  - **Average Dice:** 0.8980  
  - **Average Precision:** 0.9766  
  - **Average Recall:** 0.8393  
  - **Average F1-Score:** 0.8980  

## vi. Observations and Analysis
- **Classification:**  
  - Both traditional ML methods (SVM, MLP) and the CNN provided high accuracy, with the CNN achieving superior performance and robustness.
  - The MLP classifier, while slightly less accurate than SVM, demonstrated the effectiveness of neural network approaches on feature-based inputs.
- **Segmentation:**  
  - Traditional segmentation techniques struggled due to sensitivity to image conditions, resulting in low IoU and Dice scores.
  - The U-Net model, leveraging deep hierarchical features and skip connections, significantly improved segmentation performance.
- **Challenges:**  
  - Variability in image quality, face orientation, and lighting were common challenges across tasks.
  - Hyperparameter tuning and appropriate preprocessing played a critical role in addressing these issues.

## vii. How to Run the Code
1. **Environment Setup:**
   - Install Docker with GPU support.
   - Use the Docker image pre-configured with TensorFlow 2.10.0 GPU support.
   - Install project dependencies:
     pip install -r /workspace/requirements.txt (if cv2 gives an error at run time, run the following `!apt-get update && apt-get install -y libgl1-mesa-glx`)

2. **Data Preparation:**
   - **Classification:**  
     - Place the 4092 classification images into the `with_mask` and `without_mask` folders.
   - **Segmentation:**  
     - Ensure the segmentation dataset (9383 training images with masks and corresponding ground truths, and 8932 testing images) is organized in the specified directories.

3. **Executing Each Task:**
   - **Task 1 (Traditional ML & MLP):**  
     - Run `part_a.py` to perform feature extraction, train SVM and MLP models, and evaluate their performance.
   - **Task 2 (CNN Classification):**  
     - Run `part_b.py` to train and evaluate on test dataset.
     - Run `part_b_test.py` to evaluate using the entire dataset (provide the weights directory `/workspace/best_weights.h5`, weights converged during my training are present at `/workspace/Classification/best_model.h5`).
   - **Task 3 (Traditional Segmentation):**  
     - Run `part_c.py` to detect faces, apply thresholding on the lower half of faces, perform contour detection and morphology, and evaluate segmentation.
   - **Task 4 (U-Net Segmentation):**  
     - Run `part_d.py` to construct and train the u-net model and test on training data split.
     - Run `part_d_test.py` to evaluate using the entire dataset (model weights present in /workspace/MSFD/2/models). 

4. **Results:**  
   - Evaluation metrics, predicted masks, and CSV files with detailed results will be saved to designated output directories.
   - Review console outputs and CSV files for detailed metrics and comparisons.

---

NOTE: Segmentation file contains output (logging different average metrics over epochs), results (metrics of 10 samples) and plots (visually showing results of some images and a plot for loss and accuracy over epochs). 
