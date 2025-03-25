# **Brain Tumor Classification using ResNet50**

## **Overview**
This project implements a deep learning-based approach for brain tumor classification using the ResNet50 model. The model is trained to classify MRI images into four categories of brain tumors.
---

## **Features**

- **Brain Tumor Classification**:
  - Utilizes a pre-trained ResNet50 model fine-tuned on brain MRI images.
  - Predicts one of four tumor classes based on MRI scans.

- **High Accuracy**:
  - Achieved **98.70% training accuracy** and **97.56% test accuracy**.
  - Balanced class distribution in the dataset for reliable predictions.

---

## **Tech Stack**

- **Programming Language**: Python
- **Libraries Used**:
  - Deep Learning: `torch`, `torchvision`
  - Image Processing: `PIL`, `numpy`
  - Visualization: `matplotlib`

---

## **Dataset**

The dataset consists of MRI images categorized into four classes of brain tumors:

1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **No Tumor**

The dataset is split into **training** and **testing** sets to evaluate model performance.

---

## **Model Architecture**

- The **ResNet50** model is used as the base model.
- The final fully connected layer is modified to classify **4 tumor classes**.
- **Cross-entropy loss** is used as the loss function.
- **Adam optimizer** is used for training.

---

## **Workflow**

1. **Data Preprocessing**:
   - Resize MRI images to **224x224 pixels**.
   - Convert images to tensors and normalize pixel values.
   - Split dataset into training and testing sets.

2. **Model Training**:
   - Fine-tune the ResNet50 model.
   - Train the model on the preprocessed MRI dataset.
   - Monitor training accuracy and loss.

3. **Prediction and Evaluation**:
   - Predict tumor class for MRI images.
   - Evaluate model performance using accuracy metrics.
---

