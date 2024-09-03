# Food Classification and Calorie Estimation using CNN

## Overview

This project focuses on food classification using a Convolutional Neural Network (CNN) built on the VGG16 architecture, combined with calorie estimation. The CNN model classifies food images into one of 36 classes, and then estimates the calorie content based on predefined average values for each food class.

## Dataset

The project uses "Fruits and Vegetables Image Recognition Dataset
" dataset, organized into training, validation, and test sets. The dataset contains images of 36 different food classes.

### Dataset Download API Command

```
kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition
```

- **Training set**: 3115 images
- **Validation set**: 351 images
- **Test set**: 359 images

## Model Architecture

The model is based on the VGG16 architecture, which is pre-trained on ImageNet. The last layers of VGG16 are replaced with custom layers:

- **Global Average Pooling Layer**
- **Dense Layer**: 1024 units with ReLU activation
- **Output Layer**: Dense layer with softmax activation for multi-class classification

### Model Summary

The model consists of 15,276,900 parameters, with 562,212 trainable parameters and 14,714,688 non-trainable parameters (from the VGG16 base).

## Data Augmentation

The training data is augmented to improve the model's generalization. The augmentation includes:

- **Rescaling**
- **Rotation**: up to 20 degrees
- **Width and Height Shifts**: up to 20%
- **Shear Transformation**
- **Zooming**: up to 20%
- **Horizontal Flipping**
- **Fill Mode**: 'nearest'

## Training and Evaluation

The model is trained with the Adam optimizer, categorical crossentropy loss, and early stopping to prevent overfitting.

### Training Performance

- **Epochs**: 30 (early stopping may reduce the number of epochs)
- **Test Accuracy**: Approximately 90.53%
- **Test Loss**: 0.2983

### Evaluation

The model's performance is evaluated on the test set, achieving a high accuracy of over 90%.

## Calorie Estimation

The calorie estimation is based on the predicted food class. Each class has an associated average caloric value, stored in a dictionary.

### Example Usage

```
img_path = '/content/apple.jpg'
predicted_class = predict_food_class(img_path)
print(f"Predicted Food Class: {predicted_class}")
estimated_calories = estimate_calories(predicted_class)
print(f"Estimated Calories: {estimated_calories}")
```

**Output**:

```
Predicted Food Class: Apple
Estimated Calories: 52
```

## How to Run

1. **Mount Google Drive**:

   ```
    from google.colab import drive
    drive.mount('/content/drive')
   ```

2. **Load and Augment Data**: Set the paths to the dataset directories and apply data augmentation.

3. **Build the Model**: Load the VGG16 base model, freeze its layers, and add custom layers.

4. **Train the Model**: Compile the model, set up early stopping, and train the model.

5. **Evaluate the Model**: Evaluate the model on the test set to check accuracy and loss.

6. **Predict and Estimate Calories**: Use the trained model to predict the food class and estimate calories.

## Dependencies

- TensorFlow
- Keras
- Pandas
- NumPy
- PIL (Python Imaging Library)
- Google Colab (for mounting and running the model)
