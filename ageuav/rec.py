
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
from sklearn.model_selection import train_test_split
import data
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

def rician_fading(x, stddev,K):
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
    K=K
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




# Build the autoencoder model


def build_model(snrdb,blocksize,K):
    input_img = Input(shape=(256, 256, 3))  # Adjust the input shape based on your image size
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
    
    channel_type = "awgn"
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
        z_out, h = rician_fading(z_in, noise_stddev,K)
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

    decoded = tfc.SignalConv2D(
                num_filters,
                (5, 5),
                #name="layer_out",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_out", inverse=True),
                )(z_out)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    decoded = tfc.SignalConv2D(    
                num_filters,
                (5, 5),
                #name="layer_0",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
                )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    decoded = tfc.SignalConv2D(
                num_filters,
                (5, 5),
                #name="layer_1",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
                )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    decoded= tfc.SignalConv2D(
                num_filters,
                (5, 5),
                #name="layer_2",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
                )(decoded)
    decoded = layers.PReLU(shared_axes=[1, 2])(decoded)
    n_channels = 3
    decoded = tfc.SignalConv2D(
                n_channels,
                (9, 9),
                #name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.sigmoid,
                )(decoded)


    def psnr_metric(x_in, x_out):
      if type(x_in) is list:
        img_in = x_in[0]
      else:
         img_in = x_in
      return tf.image.psnr(img_in, x_out, max_val=1.0)
    
    
    
    # Encoder model
    recovery_model = Model(inputs=input_img, outputs=decoded)  # Use the output with AWGN for the encoder
    model_metrics = [ tf.keras.metrics.MeanSquaredError(),
            psnr_metric,
            PSNRsVar(),]
    # Use encoder.input here
    recovery_model.compile(optimizer='adam', loss='mse', metrics=model_metrics)
    return recovery_model


def plot_accuracy_vs_snr(snr_values_db, accuracy_results):
    plt.plot(snr_values_db, accuracy_results, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of SNR')
    plt.grid(True)
    plt.show()

def main():
    x_train, x_val, x_tst = get_dataset()

      # SNR values from 0 to 20 dB with 11 steps
   # train_input, val_input, train_target, val_target = train_test_split(x_train, x_test, test_size=0.2, random_state=42)


    accuracy_results = []
    sim_num = 1
    snr_values_db = np.linspace(-20,20 , num=5) # Define your SNR values here
    sim_acurracy = []
    train_snrdb = 20
    block_size = 1
    K=0.5
    rec_model = build_model(train_snrdb,block_size,K)
    rec_model.fit(x_train, batch_size=128, epochs=10, validation_data=x_val, verbose=1)
    #print the model summary
    rec_model.summary()
    
    
    # Save the trained weights
    #delete the file if it already exists
    if os.path.exists('rec_model_weights.h5'):
        os.remove('rec_model_weights.h5')
    rec_model.save_weights('rec_model_weights.h5')
    #classifier_train_loss, classifier_train_accuracy = classifier_model.evaluate(x_test, x_test,verbose=0)
    print(rec_model.evaluate(x_tst,verbose=0))
   # print("Classifier Train Loss:", classifier_train_loss)
   # print("Classifier Train Accuracy:", classifier_train_accuracy)
    
    for snr_value_db in snr_values_db:
        sim_acurracy.clear()  # Clear the list before each simulation
        
        for _ in range(sim_num):
            rec_test = build_model(snr_value_db, block_size,K=0.5)
            rec_test.load_weights('rec_model_weights.h5')
            _,_,sim_acurracy_tet,_ = rec_test.evaluate(x_tst, verbose=0)
            sim_acurracy.append(sim_acurracy_tet)
            
        classifier_test_accuracy = np.mean(sim_acurracy)
        accuracy_results.append(classifier_test_accuracy)


    plot_accuracy_vs_snr(snr_values_db, accuracy_results)


DATASETS = {
    "firedata": data,
}
def get_dataset():
    data_options = tf.data.Options()
    data_options.experimental_deterministic = False
    data_options.experimental_optimization.apply_default_optimizations = True
    data_options.experimental_optimization.map_parallelization = True
    data_options.experimental_optimization.parallel_batch = True

    def prepare_dataset(dataset, mode, parse_record_fn, bs):
        dataset = dataset.with_options(data_options)
        if mode == "train":
            dataset = dataset.shuffle(buffer_size=dataset_obj.SHUFFLE_BUFFER)
        dataset = dataset.map(
            lambda v: parse_record_fn(v, mode, tf.float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return dataset.batch(bs)

    # Provide default values directly
    dataset_obj = DATASETS.get("firedata")
    parse_record_fn = dataset_obj.parse_record
    if "kodak" != "imagenet":
        tr_val_dataset = dataset_obj.get_dataset(True, "/tmp/train_data")
        tr_dataset = tr_val_dataset.take(dataset_obj._NUM_IMAGES["train"])
        val_dataset = tr_val_dataset.skip(dataset_obj._NUM_IMAGES["train"])
    else:  # treat imagenet differently, as we usually dont use it for training
        tr_dataset = dataset_obj.get_dataset(True, "/tmp/train_data")
        val_dataset = dataset_obj.get_dataset(False, "/tmp/train_data")
    # Train
    x_train = prepare_dataset(
        tr_dataset, "train", parse_record_fn, 128
    )
    # Validation
    x_val = prepare_dataset(val_dataset, "val", parse_record_fn, 128)

    # Test
    dataset_obj = DATASETS.get("firedata")
    parse_record_fn = dataset_obj.parse_record
    tst_dataset = dataset_obj.get_dataset(False, "/tmp/train_data")
    x_tst = prepare_dataset(tst_dataset, "test", parse_record_fn, 128)
    x_tst.repeat(10)  # number of realisations per image on evaluation

    return x_train, x_val, x_tst


class PSNRsVar(tf.keras.metrics.Metric):
    """Calculate the variance of a distribution of PSNRs across batches

    """

    def __init__(self, name="variance", **kwargs):
        super(PSNRsVar, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")
        self.mean = self.add_weight(name="mean", shape=(), initializer="zeros")
        self.var = self.add_weight(name="M2", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnrs = tf.image.psnr(y_true, y_pred, max_val=1.0)
        samples = tf.cast(psnrs, self.dtype)
        batch_count = tf.size(samples)
        batch_count = tf.cast(batch_count, self.dtype)
        batch_mean = tf.math.reduce_mean(samples)
        batch_var = tf.math.reduce_variance(samples)

        # compute new values for variables
        new_count = self.count + batch_count
        new_mean = (self.count * self.mean + batch_count * batch_mean) / (
            self.count + batch_count
        )
        new_var = (
            (self.count * (self.var + tf.square(self.mean - new_mean)))
            + (batch_count * (batch_var + tf.square(batch_mean - new_mean)))
        ) / (self.count + batch_count)

        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.var.assign(new_var)

    def result(self):
        return self.var

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(np.zeros(self.count.shape))
        self.mean.assign(np.zeros(self.mean.shape))
        self.var.assign(np.zeros(self.var.shape))


# Call the block_size function to execute
if __name__ == "__main__":
    main()
    #block_size()
