import os
import keras
from keras.models import Model
from keras.layers import Activation, Input
from keras.layers import (Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, BatchNormalization)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_SAVE_FREQUENCY = 10
MODELS_DIR = "models"
RESULTS_DIR = "results"
COMPARISON_IMAGES_DIR = "comparison_images"
TRAIN_EPOCHS = 50
COMPLEX_MODEL = False
IMG_DIM = 224

if COMPLEX_MODEL:
        # Define the encoder
        def conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)
                x = Activation("relu", name = f"{block_name}_relu2")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = conv_block(encoder_input, 32, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)
        x = conv_block(x, 64, "enc_block2")
        x = MaxPooling2D(name = "enc_pool2")(x)
        x = conv_block(x, 128, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = Conv2D(256, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn")(x)
        x = Activation("relu", name = "enc_bottleneck_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        # Define the decoder
        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)
        x = Conv2D(256, (3, 3), padding = "same", name = "dec_bottleneck_conv")(x)
        x = BatchNormalization(name = "dec_bottleneck_bn")(x)
        x = Activation("relu", name = "dec_bottleneck_relu")(x)
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = conv_block(x, 128, "dec_block1")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = conv_block(x, 64, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = conv_block(x, 32, "dec_block3")
        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

else:
        # Define the encoder
        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3))
        encoded = Conv2D(16, (3, 3), padding = "same")(encoder_input)
        encoded = Activation("relu")(encoded)
        encoded = MaxPooling2D()(encoded)
        encoded = Conv2D(2, (3, 3), padding = "same")(encoded)
        encoded = Activation("relu")(encoded)
        encoded = MaxPooling2D()(encoded)
        encoded = Conv2D(2, (3, 3), padding = "same")(encoded)
        encoded = Activation("relu")(encoded)

        encoded_feature_map_shape = tuple(
                int(dimension) for dimension in encoded.shape[1:]
        )
        encoded = Flatten()(encoded)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        # Define the decoder
        decoder_input = Input(shape = (encoded_vector_length,))
        decoded = Reshape(encoded_feature_map_shape)(decoder_input)
        decoded = UpSampling2D()(decoded)
        decoded = Conv2D(16, (3, 3), padding = "same")(decoded)
        decoded = Activation("relu")(decoded)
        decoded = UpSampling2D()(decoded)
        decoded = Conv2D(3, (3, 3), padding = "same")(decoded)
        decoded = Activation("sigmoid")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

# Combine encoder and decoder into an autoencoder
autoencoder_output = decoder(encoder(encoder_input))
model = Model(encoder_input, autoencoder_output, name = "autoencoder")
model.compile(optimizer = "adam", loss = "mse")


def save_models(encoder_model, decoder_model, output_dir, epoch_number):
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch_number:04d}")
        os.makedirs(epoch_dir, exist_ok = True)
        encoder_model.save(os.path.join(epoch_dir, "model_encoder.keras"))
        decoder_model.save(os.path.join(epoch_dir, "model_decoder.keras"))


os.makedirs(MODELS_DIR, exist_ok = True)
os.makedirs(RESULTS_DIR, exist_ok = True)
save_models(encoder, decoder, MODELS_DIR, 0)



def save_loss_plot(history, output_path):
        epochs = range(1, len(history.history.get("loss", [])) + 1)
        plt.figure()
        if "loss" in history.history:
                plt.plot(epochs, history.history["loss"], label = "Training Loss")
        if "val_loss" in history.history:
                plt.plot(epochs, history.history["val_loss"], label = "Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, format = "jpg")
        plt.close()

def load_comparison_images(directory, target_size):
        if not os.path.isdir(directory):
                print(f"Comparison images directory '{directory}' does not exist.")
                return np.empty((0, target_size[0], target_size[1], 3), dtype = "float32")

        images = []
        if hasattr(Image, "Resampling"):
                resample_method = Image.Resampling.LANCZOS
        else:
                resample_method = Image.LANCZOS

        for file_name in sorted(os.listdir(directory)):
                file_path = os.path.join(directory, file_name)
                if not os.path.isfile(file_path):
                        continue

                try:
                        image = Image.open(file_path).convert("RGB")
                        image = image.resize(target_size, resample = resample_method)
                        image_array = np.asarray(image, dtype = "float32") / 255.0
                        images.append(image_array)
                except (OSError, ValueError) as error:
                        print(f"Skipping comparison image '{file_path}': {error}")

        if not images:
                print(f"No valid comparison images were found in '{directory}'.")
                return np.empty((0, target_size[0], target_size[1], 3), dtype = "float32")

        return np.stack(images, axis = 0)


def save_comparisons(model, output_dir, epoch_number, batch_x, batch_y, num_examples):
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch_number:04d}")
        os.makedirs(epoch_dir, exist_ok = True)

        if len(batch_x) == 0 or num_examples <= 0:
                return

        predicted = model.predict(batch_x, verbose = 0)

        num_examples = min(num_examples, len(batch_x))

        total_difference = 0.0

        for index in range(num_examples):
                target_array = batch_y[index]
                predicted_array = predicted[index]
                total_difference += np.sum(np.abs(target_array - predicted_array))

                target_image = np.clip(target_array * 255, 0, 255).astype("uint8")
                output_image = np.clip(predicted_array * 255, 0, 255).astype("uint8")

                comparison = np.hstack((target_image, output_image))
                comparison_image = Image.fromarray(comparison)

                comparison_path = os.path.join(epoch_dir, f"comparison_{index + 1:02d}.jpg")
                comparison_image.save(comparison_path, format = "JPEG")

        avg_difference_percentage = 100 * total_difference / (num_examples * 3 * IMG_DIM * IMG_DIM)
        print( "avg delta per pixel for epoch ", epoch_number, " : ", round(avg_difference_percentage, 2), "%")

class PeriodicModelSaver(keras.callbacks.Callback):
        def __init__(self, frequency, output_dir, encoder_model, decoder_model):
                super().__init__()
                self.frequency = frequency
                self.output_dir = output_dir
                self.encoder_model = encoder_model
                self.decoder_model = decoder_model

        def on_train_begin(self, logs = None):
                if self.frequency <= 0:
                        return
                initial_encoder_path = os.path.join(
                        self.output_dir,
                        "epoch_0000",
                        "model_encoder.keras",
                )
                if not os.path.exists(initial_encoder_path):
                        save_models(self.encoder_model, self.decoder_model, self.output_dir, 0)

        def on_epoch_end(self, epoch, logs = None):
                if self.frequency <= 0:
                        return
                epoch_number = epoch + 1
                if epoch_number % self.frequency == 0:
                        save_models(self.encoder_model, self.decoder_model, self.output_dir, epoch_number)


class PeriodicComparisonSaver(keras.callbacks.Callback):
        def __init__(self, frequency, output_dir, batch_x, batch_y, num_examples):
                super().__init__()
                self.frequency = frequency
                self.output_dir = output_dir
                self.batch_x = batch_x
                self.batch_y = batch_y
                self.num_examples = num_examples

        def on_train_begin(self, logs = None):
                if self.frequency <= 0:
                        return
                if len(self.batch_x) == 0:
                        print("Skipping comparison image saving because no comparison images are available.")
                        return
                save_comparisons(self.model, self.output_dir, 0, self.batch_x, self.batch_y, self.num_examples)

        def on_epoch_end(self, epoch, logs = None):
                if self.frequency <= 0:
                        return
                if len(self.batch_x) == 0:
                        return
                epoch_number = epoch + 1
                if epoch_number % self.frequency == 0:
                        save_comparisons(self.model, self.output_dir, epoch_number, self.batch_x, self.batch_y, self.num_examples)

model.summary()

# Generate data from the images in a folder
batch_size = 8
train_datagen = ImageDataGenerator(rescale = 1. / 255, data_format = 'channels_last')
train_generator = train_datagen.flow_from_directory(
        'cropped/',
        target_size = (IMG_DIM, IMG_DIM),
        batch_size = batch_size,
        class_mode = 'input'
)
test_datagen = ImageDataGenerator(rescale = 1. / 255, data_format = 'channels_last')
validation_generator = test_datagen.flow_from_directory(
        'cropped/',
        target_size = (IMG_DIM, IMG_DIM),
        batch_size = batch_size,
        class_mode = 'input'
)

comparison_images = load_comparison_images(COMPARISON_IMAGES_DIR, (IMG_DIM, IMG_DIM))

if comparison_images.size == 0:
        sample_batch_x, sample_batch_y = next(validation_generator)
        validation_generator.reset()
        comparison_batch_x = sample_batch_x
        comparison_batch_y = sample_batch_y
else:
        comparison_batch_x = comparison_images
        comparison_batch_y = comparison_images

# Train the model
history = model.fit(
        train_generator,
        steps_per_epoch = 1000 // batch_size,
        epochs = TRAIN_EPOCHS,
        validation_data = validation_generator,
        validation_steps = 1000 // batch_size,
        callbacks = [
                PeriodicModelSaver(MODEL_SAVE_FREQUENCY, MODELS_DIR, encoder, decoder),
                PeriodicComparisonSaver(
                        MODEL_SAVE_FREQUENCY,
                        RESULTS_DIR,
                        comparison_batch_x,
                        comparison_batch_y,
                        len(comparison_batch_x),
                ),
        ])

loss_plot_path = os.path.join(RESULTS_DIR, "loss_curve.jpg")
save_loss_plot(history, loss_plot_path)