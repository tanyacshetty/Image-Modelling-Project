## Project Overview

This project builds an image classification model to predict whether an image contains a **cat** or a **dog** using deep learning. The model is based on **MobileNetV2** (a pre-trained CNN) and fine-tuned for binary classification. The dataset used is the Kaggle **Cats vs Dogs** dataset.

## Requirements

Install the necessary libraries:

```bash
pip install tensorflow==2.18.0
pip install numpy==1.26.0
pip install matplotlib pillow
pip install FuzzyTM>=0.4.0
pip install PyQt5==5.15.11
pip install PyQt5-sip>=12.15,<13
```

## Dataset

The dataset consists of images of cats and dogs organized into separate folders for training and validation.

## Key Steps

1. **Data Preprocessing**: 
   - Verifies and removes any corrupted images.
   - Uses `ImageDataGenerator` to split the dataset into training and validation sets (80/20).

2. **Model Architecture**:
   - Uses **MobileNetV2** pre-trained on ImageNet.
   - Freezes the base model and adds custom layers for binary classification.
   - Model architecture: `MobileNetV2 -> GlobalAveragePooling -> Dense(1024) -> Dense(1, sigmoid)`.

3. **Training**:
   - Trains the model for 10 epochs using the `Adam` optimizer and `binary_crossentropy` loss.
   - Validation accuracy is tracked.

4. **Model Evaluation**:
   - After training, the model's performance is visualized using accuracy and loss plots.

5. **Saving the Model**:
   - The trained model is saved in the `.keras` format for later use.

6. **Prediction**:
   - The model is used to predict whether a given image contains a cat or dog.

## How to Run

1. **Train the Model**: 
   - Run the main training script to preprocess the data and train the model.
   
2. **Make Predictions**:
   - After training, load the model and use it to predict images:
   ```python
   model = tf.keras.models.load_model('cat_dog_model.keras')
   img_path = '/path/to/your/image.jpg'
   prediction = model.predict(image_processing_function(img_path))
   ```

## Conclusion

The project successfully classifies images of cats and dogs with a trained deep learning model. It utilizes transfer learning with **MobileNetV2** for feature extraction and achieves good accuracy on the test data.

---




