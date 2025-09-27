import os

import keras
from keras.models import Model
from keras.layers import Activation, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_SAVE_FREQUENCY = 10
MODELS_DIR = "models"
RESULTS_DIR = "results"
NUM_RESULT_EXAMPLES = 5

# Define the encoder
encoder_input = Input(shape = (224, 224, 3))
encoded = Conv2D(16, (3, 3), padding = "same")(encoder_input)
encoded = Activation("relu")(encoded)
encoded = MaxPooling2D()(encoded)
encoded = Conv2D(2, (3, 3), padding = "same")(encoded)
encoded = Activation("relu")(encoded)
encoded = MaxPooling2D()(encoded)
encoded = Conv2D(2, (3, 3), padding = "same")(encoded)
encoded = Activation("relu")(encoded)

encoder = Model(encoder_input, encoded, name = "encoder")

# Define the decoder
decoder_input = Input(shape = encoder.output_shape[1:])
decoded = UpSampling2D()(decoder_input)
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

def save_comparisons(model, output_dir, epoch_number, batch_x, batch_y, num_examples):
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch_number:04d}")
        os.makedirs(epoch_dir, exist_ok = True)

        predicted = model.predict(batch_x, verbose = 0)

        num_examples = min(num_examples, len(batch_x))

        for index in range(num_examples):
                target_image = np.clip(batch_y[index] * 255, 0, 255).astype("uint8")
                output_image = np.clip(predicted[index] * 255, 0, 255).astype("uint8")

                comparison = np.hstack((target_image, output_image))
                comparison_image = Image.fromarray(comparison)

                comparison_path = os.path.join(epoch_dir, f"comparison_{index + 1:02d}.jpg")
                comparison_image.save(comparison_path, format = "JPEG")


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
                save_comparisons(self.model, self.output_dir, 0, self.batch_x, self.batch_y, self.num_examples)

        def on_epoch_end(self, epoch, logs = None):
                if self.frequency <= 0:
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
	target_size = (224, 224),
	batch_size = batch_size,
	class_mode = 'input'
)
test_datagen = ImageDataGenerator(rescale = 1. / 255, data_format = 'channels_last')
validation_generator = test_datagen.flow_from_directory(
        'cropped/',
        target_size = (224, 224),
        batch_size = batch_size,
        class_mode = 'input'
)

sample_batch_x, sample_batch_y = next(validation_generator)
validation_generator.reset()

# Train the model
model.fit(
        train_generator,
        steps_per_epoch = 1000 // batch_size,
        epochs = 20,
        validation_data = validation_generator,
        validation_steps = 1000 // batch_size,
        callbacks = [
                PeriodicModelSaver(MODEL_SAVE_FREQUENCY, MODELS_DIR, encoder, decoder),
                PeriodicComparisonSaver(
                        MODEL_SAVE_FREQUENCY,
                        RESULTS_DIR,
                        sample_batch_x,
                        sample_batch_y,
                        NUM_RESULT_EXAMPLES,
                ),
        ])