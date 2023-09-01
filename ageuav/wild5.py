
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
import tensorflow_compression as tfc
from tensorflow.keras import layers
# Load the dataset
tf.random.set_seed(3)
np.random.seed(3)

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn

    return y

def fading(x, stddev):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    h = tf.complex(
        tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
        tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
    )

    # additive white gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )

    return (h * x + stddev * awgn), h

def rician_fading(x, stddev):
    """Implements the Rician fading channel with multiplicative fading,
    additive white Gaussian noise, and a line-of-sight (LOS) component.
    Args:
        x: channel input symbols
        K: Rician K factor (ratio of LOS power to scattered power)
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
        h: channel gains (complex)
    """
    # Generate LOS component with magnitude K
    K = 10
    los_magnitude = np.sqrt(K / (K + 1))
    los_phase = tf.random.uniform([tf.shape(x)[0], 1], 0, 2 * np.pi)
    los = tf.complex(los_magnitude * tf.cos(los_phase), los_magnitude * tf.sin(los_phase))

    # Generate scattered fading component
    scattered_magnitude = np.sqrt(1 / (K + 1))
    scattered_real = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2))
    scattered_imag = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2))
    scattered = tf.complex(scattered_magnitude * scattered_real, scattered_magnitude * scattered_imag)

    # Combine LOS and scattered components
    h = los + scattered

    # Generate additive white Gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )

    # Apply Rician fading and noise
    y = (h * x + stddev * awgn)

    return y, h

def phase_invariant_fading(x, stddev):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.

    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain

    n1 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
    n2 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)

    h = tf.sqrt(tf.square(n1) + tf.square(n2))

    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev / np.sqrt(2), dtype=tf.float32)

    return (h * x + awgn), h


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

# Build the autoencoder model


def build_model(snrdb,blocksize):
    input_img = Input(shape=(128, 128, 3))  # Adjust the input shape based on your image size
    num_filters = 16
    conv_depth = blocksize
    # Encoder layers
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (9, 9),
                    name="layer_0",
                    corr=True,
                    strides_down=2,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_0"),
                )(input_img)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (5, 5),
                    name="layer_1",
                    corr=True,
                    strides_down=2,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_1"),
                )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    num_filters,
                    (5, 5),
                    name="layer_2",
                    corr=True,
                    strides_down=1,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_2"),
                )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded= tfc.SignalConv2D(
                    num_filters,
                    (5, 5),
                    name="layer_3",
                    corr=True,
                    strides_down=1,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(name="gdn_3"),
                )(encoded)
    encoded = layers.PReLU(shared_axes=[1, 2])(encoded)
    encoded = tfc.SignalConv2D(
                    conv_depth,
                    (5, 5),
                    name="layer_out",
                    corr=True,
                    strides_down=1,
                    padding="same_zeros",
                    use_bias=True,
                    activation=None,
                )(encoded)
    #encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    
    
    # Function to convert SNR from dB to linear scale
    def db_to_linear(snr_db):
        return 10 ** (snr_db / 10)
    
    # Define the SNR value in dB (adjust this as needed)
    snr_value_db = snrdb
    inter_shape = tf.shape(encoded)
    # reshape array to [-1, dim_z]
    z = layers.Flatten()(encoded)
    # convert from snr to std
    print("channel_snr: {}".format(snr_value_db))
    noise_stddev = np.sqrt(10 ** (-snr_value_db / 10))
    
    channel_type = "rician_fading"
    # Add channel noise
    if channel_type == "awgn":
        dim_z = tf.shape(z)[1]
        # normalize latent vector so that the average power is 1
        z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
            z, axis=1
        )
        z_out = real_awgn(z_in, noise_stddev)
        h = tf.ones_like(z_in)  # h just makes sense on fading channels

    elif channel_type == "fading":
        dim_z = tf.shape(z)[1] // 2
        # convert z to complex representation
        z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
        # normalize the latent vector so that the average power is 1
        z_norm = tf.reduce_sum(
            tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        )
        z_in = z_in * tf.complex(
            tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
        )
        z_out, h = fading(z_in, noise_stddev)
        # convert back to real
        z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)
    elif channel_type == "rician_fading":
        dim_z = tf.shape(z)[1] // 2
        # convert z to complex representation
        z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
        # normalize the latent vector so that the average power is 1
        z_norm = tf.reduce_sum(
            tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        )
        z_in = z_in * tf.complex(
            tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
        )
        z_out, h = fading(z_in, noise_stddev)
        # convert back to real
        z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)

    elif channel_type == "fading-real":
        # half of the channels are I component and half Q
        dim_z = tf.shape(z)[1] // 2
        # normalization
        z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
            z, axis=1
        )
        z_out, h = phase_invariant_fading(z_in, noise_stddev)

    else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
    z_out = tf.reshape(z_out, inter_shape)




      # Use mean squared error for image reconstruction
    
    # Encoder model
    encoder = Model(inputs=input_img, outputs=z_out)  # Use the output with AWGN for the encoder
    
    # Classifier model
    # Use the encoder output (classifier_input) directly as the input to the classifier
    classifier_input = encoder.output  # Use encoder output as the input to the classifier
    flatten = Flatten()(classifier_input) # Flatten the output
    classifier_output = Dense(blocksize, activation='relu')(flatten)
    classifier_output = Dense(2, activation='softmax')(classifier_output)  # Assuming 2 classes: fire and nofire
    classifier_model = Model(inputs=input_img, outputs=classifier_output)  # Use encoder.input here
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

def plot_accuracy_vs_snr(snr_values_db, accuracy_results, accuracy_results_2):
    plt.plot(snr_values_db, accuracy_results, color='g', linestyle='-', marker='o', label='train snr=20dB')
    plt.plot(snr_values_db, accuracy_results_2,color='b', linestyle='-', marker='o', label='train snr=10dB')
    plt.xlabel('SNR (dB)',fontsize=14)
    plt.ylabel('Classification accuracy',fontsize=14) #classification accuracy
    plt.legend()
    #plt.title('Accuracy as a function of SNR')
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

      # SNR values from 0 to 20 dB with 11 steps
    



    accuracy_results = []
    accuracy_results_2 = []
    sim_num = 1
    snr_values_db = np.linspace(-20,20 , num=10) # Define your SNR values here
    sim_acurracy = []
    sim_acurracy_2 = []
    train_snrdb = 20
    train_snrdb_2=10
    block_size = 8
    classifier_model = build_model(train_snrdb,block_size)
    classifier_model_2=build_model(train_snrdb_2,block_size)
    classifier_model.fit(x_train, y_train, epochs=150, batch_size=128, validation_data=(x_test, y_test))
    classifier_model_2.fit(x_train, y_train, epochs=150, batch_size=128, validation_data=(x_test, y_test))
    #print the model summary
    #classifier_model.summary()
    
    
    # Save the trained weights
    #delete the file if it already exists
    if os.path.exists('classifier_model_weights_main.h5'):
        os.remove('classifier_model_weights_main.h5')
    if os.path.exists('classifier_model_weights_main_2.h5'):
        os.remove('classifier_model_weights_main_2.h5')
    classifier_model.save_weights('classifier_model_weights_main.h5')
    classifier_model_2.save_weights('classifier_model_weights_main_2.h5')
    classifier_train_loss, classifier_train_accuracy = classifier_model.evaluate(x_test, y_test, verbose=0)
    classifier_train_loss_2, classifier_train_accuracy_2 = classifier_model_2.evaluate(x_test, y_test, verbose=0)
    print("Classifier Train Loss:", classifier_train_loss)
    print("Classifier Train Accuracy:", classifier_train_accuracy_2)
    print("Classifier Train Accuracy:", classifier_train_accuracy)
    print("Classifier Train Loss:", classifier_train_loss_2)
    
    for snr_value_db in snr_values_db:
        sim_acurracy.clear()  # Clear the list before each simulation
        
        for _ in range(sim_num):
            block_size = 8
            classifer_test = build_model(snr_value_db, block_size)
            classifer_test_2 = build_model(snr_value_db, block_size)
            classifer_test.load_weights('classifier_model_weights_main.h5')
            classifer_test_2.load_weights('classifier_model_weights_main_2.h5')
            _, sim_acurracy_tet = evaluate_classifier(classifer_test, x_test, y_test)
            _, sim_acurracy_tet_2 = evaluate_classifier(classifer_test_2, x_test, y_test)
            sim_acurracy.append(sim_acurracy_tet)
            sim_acurracy_2.append(sim_acurracy_tet_2)
            
        classifier_test_accuracy = np.mean(sim_acurracy)
        classifier_test_accuracy_2 = np.mean(sim_acurracy_2)
        accuracy_results.append(classifier_test_accuracy)
        accuracy_results_2.append(classifier_test_accuracy_2)


    plot_accuracy_vs_snr(snr_values_db, accuracy_results, accuracy_results_2) #train snr
    #plot_accuracy_vs_snr(snr_values_db, accuracy_results_2')

def test_accurcy(snr_value_db, x_test, y_test, block_size):
    sim_num = 1
    sim_acurracy = []
    for _ in range(sim_num):
        classifer_test = build_model(snr_value_db, block_size)
        classifer_test.load_weights('classifier_model_weights_train.h5')
        _, sim_acurracy_test = evaluate_classifier(classifer_test, x_test, y_test)
        sim_acurracy.append(sim_acurracy_test)
    classifier_test_accuracy = np.mean(sim_acurracy) 
    return classifier_test_accuracy
def train (train_snrdb, x_train, y_train, x_test, y_test,block_size):
       classifier_model = build_model(train_snrdb,block_size)
       classifier_model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))
       classifier_model.summary()
       if os.path.exists('classifier_model_weights_train.h5'):
         os.remove('classifier_model_weights_train.h5')
       classifier_model.save_weights('classifier_model_weights_train.h5')
       sim_loss_tain, sim_acurracy_train = evaluate_classifier(classifier_model, x_test, y_test)
       return sim_acurracy_train

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
             classifier_model.fit(x_train, y_train, epochs=1, batch_size=128)
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