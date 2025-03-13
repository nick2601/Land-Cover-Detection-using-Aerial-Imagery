# Land Cover Detection Using Aerial Imagery

## Project Overview
This project focuses on land cover detection using the UC Merced Land Use dataset, which contains 21 categories of land cover. The goal is to classify images into these categories using various machine learning models. The project demonstrates practical skills in machine learning, image processing, and technical writing.

## Dataset Description
The dataset consists of:

- **Images**: 2100 RGB images in low resolution (61 × 61 × 3) stored in `DS_Xdata.npy`.
- **Labels**: Corresponding labels for the images stored in `Ydata.npy`. Each category contains 100 images.

The images are divided into three sets:
- **Training Set**: 80% of the data
- **Validation Set**: 10% of the data
- **Test Set**: 10% of the data

## Models
The project implements three models:

### Model 1: Classification/Clustering
A classification or clustering model applied to images from 5 selected classes. Algorithms used may include K-Means or Gaussian Mixture Models.

### Model 2: Preprocessing and Classification
Image preprocessing using dimensionality reduction methods (PCA or LDA) or feature extraction methods (e.g., Histogram of Oriented Gradients), followed by classification using models from Model 1.

### Model 3: Convolutional Neural Network (CNN)
A CNN model for classifying images across all 21 classes. Strategies to improve results will be employed, such as data augmentation and hyperparameter tuning.

## Usage Instructions
### Setup
Ensure you have the necessary libraries installed. You can use the following command:
```bash
pip install numpy scikit-learn scikit-image keras
```

### Running the Models
1. Load the dataset using NumPy:
   ```python
   import numpy as np
   X_data = np.load('DS_Xdata.npy')
   Y_data = np.load('Ydata.npy')
   ```
2. Shuffle and split the dataset into training, validation, and test sets.
3. Implement the models as described in the project.

### Evaluating the Models
After training, evaluate the models on the test set and visualize the results using confusion matrices and accuracy metrics.

## Results
Present your experimental results, including:
- Confusion matrices for each model.
- Training and validation curves.
- Quantitative metrics such as accuracy.

## Conclusion
Summarize the findings and the effectiveness of the models in classifying land cover from aerial imagery.