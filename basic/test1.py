
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
tf.random.set_seed(3)

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

def build_autoencoder_classifier(snr_db,block_size):
    tx_power = 1
    input_img = Input(shape=(128, 128, 3))

    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Flatten()(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = Dense(block_size, activation='linear')(encoded)
    #encoded = Lambda(lambda x: x / np.sqrt(tx_power))(encoded)

    snr_value_linear = db_to_linear(snr_db)
    awgn_layer = GaussianNoise(stddev=np.sqrt(1.0 / snr_value_linear))
    encoded_with_awgn = awgn_layer(encoded)

    encoder = Model(inputs=input_img, outputs=encoded_with_awgn)

    classifier_input = encoder.output
    classifier_output = Dense(block_size, activation='relu')(classifier_input)
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
    test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Testing"
    #test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"

    x_train, y_train = get_data(train_folder)
    x_test, y_test = get_data(test_folder)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    snr_values_db = np.linspace(-20, 20, num=5)  # SNR values from 0 to 20 dB with 11 steps
    

    accuracy_results = []
    sim_num = 6
    sim_acurracy = []
    for snr_value_db in snr_values_db:
        for i in range(sim_num):
           block_size = 16
           classifier_model = build_autoencoder_classifier(snr_value_db,block_size)
   
           classifier_model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))
   
           _, sim_acurracy_tet = evaluate_classifier(classifier_model, x_test, y_test)
           sim_acurracy.append(sim_acurracy_tet)
        classifier_test_accuracy = np.mean(sim_acurracy)
        accuracy_results.append(classifier_test_accuracy)

    plot_accuracy_vs_snr(snr_values_db, accuracy_results)

def test_accurcy(snr_value_db, x_train, y_train, x_test, y_test):
    block_size = 16
    sim_num = 5
    sim_acurracy = []
    for i in range(sim_num):
        classifier_model = build_autoencoder_classifier(snr_value_db,block_size)
    
        classifier_model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))
        _, sim_acurracy_test = evaluate_classifier(classifier_model, x_test, y_test)
        sim_acurracy.append(sim_acurracy_test)
    classifier_test_accuracy = np.mean(sim_acurracy) 
    return classifier_test_accuracy

def block_size():
    train_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"
    #test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Training and Validation"
    test_folder = "/home/chathuranga_basnayaka/Desktop/my/semantic/wild/deepJSCC-feedback/wilddata/forest_fire/Testing"

    x_train, y_train = get_data(train_folder)
    x_test, y_test = get_data(test_folder)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    block_sizes = [2,4,8,12,16]  # List of block sizes

    accuracy_results = []
    for block_size_value in block_sizes:  # Changed the variable name here
        classifier_test_accuracy_sim = []
        number_of_eval = 10
        for i in range(number_of_eval):
             snr_value_db = 1
             classifier_model = build_autoencoder_classifier(snr_value_db, block_size_value)
             classifier_model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))
             _, classifier_test_accuracy_test = evaluate_classifier(classifier_model, x_test, y_test)
             classifier_test_accuracy_sim.append(classifier_test_accuracy_test)
        
        classifier_test_accuracy = np.mean(classifier_test_accuracy_sim)

        accuracy_results.append(classifier_test_accuracy)
    
    # plot accuracy vs block size
    plt.plot(block_sizes, accuracy_results, marker='o') 
    plt.xlabel('block length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of block length')
    plt.grid(True)
    plt.show() # Use block_sizes here

# Call the block_size function to execute
if __name__ == "__main__":
    main()
    #block_size()
