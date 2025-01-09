🐱 Project Overview

This project implements an image classification model to predict whether an image contains a cat or a dog. The model leverages deep learning and Convolutional Neural Networks (CNNs) using a pre-trained MobileNetV2 model. The dataset used is the Cats vs Dogs dataset from Kaggle, and the model was trained using TensorFlow and Keras.
⚙️ Requirements

To run this project, make sure to install the required dependencies. You can create a virtual environment and install the dependencies using the following commands:

pip install -r requirements.txt

Alternatively, you can install the necessary packages individually:

pip install tensorflow==2.18.0
pip install numpy==1.26.0
pip install matplotlib
pip install pillow
pip install FuzzyTM>=0.4.0
pip install PyQt5==5.15.11
pip install PyQt5-sip>=12.15,<13

requirements.txt

tensorflow==2.18.0
numpy==1.26.0
matplotlib
pillow
FuzzyTM>=0.4.0
PyQt5==5.15.11
PyQt5-sip>=12.15,<13

🔧 Project Setup
1. Data Preprocessing and Image Verification

This project first ensures that all images in the dataset are valid by verifying them using the PIL library. If any corrupted images are found, they are removed to ensure that the model is trained on clean data.

# Function to check for corrupted images
def check_images(directory):
    invalid_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verifies if the image can be opened without errors
                except (IOError, SyntaxError) as e:
                    invalid_images.append(file_path)  # Collect paths of corrupted images
    return invalid_images

2. Data Generators

Data augmentation and splitting the dataset into training and validation sets is done using Keras ImageDataGenerator.

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    '/path/to/PetImages',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    '/path/to/PetImages',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

3. Model Architecture

The model uses MobileNetV2, a pre-trained CNN architecture, as the base model for feature extraction. Custom layers are added on top for binary classification.

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

4. Training

The model is trained for 10 epochs using the training and validation datasets. The accuracy and loss are tracked throughout training.

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

5. Model Evaluation

After training, the model's accuracy and loss are plotted to visualize its performance over time.

# Plotting the accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plotting the loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

6. Saving the Model

Once trained, the model is saved in the .keras format.

model.save('cat_dog_model.keras')

🧠 Predicting on New Images

After training, you can load the model and make predictions on new images. Below is the code to load a new image, preprocess it, and classify it as either a "Cat" or "Dog".

# Load the saved model
model = tf.keras.models.load_model('cat_dog_model.keras')

# Load the image for prediction
img_path = '/path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))  # Resize the image
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Normalize the image array (same as during training)
img_array = img_array / 255.0

# Make the prediction
prediction = model.predict(img_array)

# Print the prediction result
if prediction[0] > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")

🧑‍💻 Usage
1. Training the Model

To train the model, simply run the main script, which will load the data, preprocess it, and start the training process.
2. Making Predictions

To use the model for predictions, save an image as a .jpg or .png file and run the prediction code with the path to the image.
🔄 Model Evaluation

You can track the model's performance using the accuracy and loss plots provided above, and tune the hyperparameters or use different architectures as needed to improve performance.
📄 License

Feel free to use and modify this code as you like. This project is open-source and licensed under the MIT License.
Final Notes:

    If you want to train the model for more epochs, just increase the epochs parameter in the fit function.
    You can experiment with different pre-trained models like ResNet50 or InceptionV3 to see if they improve your model's performance.
