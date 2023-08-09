import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda, GaussianNoise
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from keras import backend as K

def db_to_linear(snr_db):
    return 10 ** (snr_db / 10)

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

def build_autoencoder_classifier(snr_db):
    tx_power = 1
    input_img = Input(shape=(128, 128, 3))

    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Flatten()(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = Dense(16, activation='linear')(encoded)
    encoded = Lambda(lambda x: x / np.sqrt(tx_power))(encoded)

    snr_value_linear = db_to_linear(snr_db)
    awgn_layer = GaussianNoise(stddev=np.sqrt(1.0 / snr_value_linear))
    encoded_with_awgn = awgn_layer(encoded)

    encoder = Model(inputs=input_img, outputs=encoded_with_awgn)

    classifier_input = encoder.output
    classifier_output = Dense(16, activation='relu')(classifier_input)
    #classifier_output = Flatten()(classifier_input)
    classifier_output = Dense(2, activation='softmax')(classifier_output)
    classifier_model = Model(inputs=encoder.input, outputs=classifier_output)
    classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier_model

def evaluate_classifier(classifier_model, x_test, y_test):
    classifier_test_loss, classifier_test_accuracy = classifier_model.evaluate(x_test, y_test, verbose=0)
    return classifier_test_loss, classifier_test_accuracy

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_accuracy_vs_snr(snr_values_db, accuracy_results):
    plt.plot(snr_values_db, accuracy_results, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of SNR')
    plt.grid(True)
    plt.show()

def main():
    train_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"
    #test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Testing"
    test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"

    x_train, y_train = get_data(train_folder)
    x_test, y_test = get_data(test_folder)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    snr_values_db = np.linspace(-20, 20, num=3)  # SNR values from 0 to 20 dB with 11 steps

    accuracy_results = []
    for snr_value_db in snr_values_db:
        classifier_model = build_autoencoder_classifier(snr_value_db)

        classifier_model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))

        _, classifier_test_accuracy = evaluate_classifier(classifier_model, x_test, y_test)
        accuracy_results.append(classifier_test_accuracy)

    plot_accuracy_vs_snr(snr_values_db, accuracy_results)

if __name__ == "__main__":
    main()
