import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Load the dataset
train_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"
test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Testing"

def get_data(folder):
    x = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith("."):
            label = 0 if folderName == "nofire" else 1
            for image_filename in tqdm(os.listdir(os.path.join(folder, folderName))):
                img_file = cv2.imread(os.path.join(folder, folderName, image_filename))
                if img_file is not None:
                    img_file = resize(img_file, (128, 128, 3), mode="constant", anti_aliasing=True)
                    img_arr = np.asarray(img_file)
                    x.append(img_arr)
                    y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

x_train, y_train = get_data(train_folder)
x_test, y_test = get_data(test_folder)

# Normalize the images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Define the blocklength
blocklength = 16
# Build the autoencoder model
input_img = Input(shape=(128, 128, 3))  # Adjust the input shape based on your image size

# Encoder layers
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
encoded = tf.keras.layers.Dropout(0.25)(encoded)
encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
encoded = tf.keras.layers.Dropout(0.25)(encoded)
encoded = Flatten()(encoded)
encoded = Dense(blocklength, activation='relu')(encoded)


encoded=  Dense(blocklength, activation='relu')(encoded)

# Function to convert SNR from dB to linear scale
def db_to_linear(snr_db):
    return 10 ** (snr_db / 10)

# Define the SNR value in dB (adjust this as needed)
snr_value_db = 10

# Convert SNR from dB to linear scale
snr_value_linear = db_to_linear(snr_value_db)

# AWGN layer with linear scale value
awgn_layer = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1.0 / snr_value_linear))
encoded_with_awgn = awgn_layer(encoded)

# Encoder model
encoder = Model(inputs=input_img, outputs=encoded_with_awgn)  # Use the output with AWGN for the encoder

# Classifier model
# Use the encoder output (classifier_input) directly as the input to the classifier
classifier_input = encoder.output  # Use encoder output as the input to the classifier
#flatten = Flatten()(classifier_input) # Flatten the output
classifier_output_1 = Dense(blocklength, activation='relu')(classifier_input)
classifier_output = Dense(2, activation='softmax')(classifier_output_1)  # Assuming 2 classes: fire and nofire
classifier_model = Model(inputs=encoder.input, outputs=classifier_output)  # Use encoder.input here
classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
summary = classifier_model.summary()

# Train the autoencoder and classifier together
classifier_model.fit(x_train, y_train, epochs=2, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the classifier on test data (original images)
classifier_test_loss, classifier_test_accuracy = classifier_model.evaluate(x_test, y_test, verbose=0)
print("Classifier Test Loss:", classifier_test_loss)
print("Classifier Test Accuracy:", classifier_test_accuracy)

# Generate and print the classification report using original labels (y_test)
y_pred = classifier_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)  # Convert one-hot encoded predictions to class labels

# Generate and print the classification report using original labels (y_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

class_names = ['No Fire', 'Fire']
plot_confusion(y_test, y_pred, class_names)
