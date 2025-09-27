import os

import keras
from keras.models import Sequential
from keras.layers import Activation, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_SAVE_FREQUENCY = 10
MODELS_DIR = "models"
RESULTS_DIR = "results"
NUM_RESULT_EXAMPLES = 5

# Define the model
model = Sequential([
	Input(shape = (224, 224, 3)),
	Conv2D(16, (3, 3), padding = "same"), Activation("relu"),
	MaxPooling2D(),
	Conv2D(2, (3, 3), padding = "same"), Activation("relu"),
	MaxPooling2D(),
	Conv2D(2, (3, 3), padding = "same"), Activation("relu"),
	UpSampling2D(),
	Conv2D(16, (3, 3), padding = "same"), Activation("relu"),
	UpSampling2D(),
	Conv2D(3, (3, 3), padding = "same"), Activation("sigmoid"),
])
model.compile(optimizer = "adam", loss = "mse")

os.makedirs(MODELS_DIR, exist_ok = True)
os.makedirs(RESULTS_DIR, exist_ok = True)
model.save(os.path.join(MODELS_DIR, "model_0000.keras"))

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


class PeriodicModelCheckpoint(keras.callbacks.Callback):
        def __init__(self, frequency, output_dir):
                super().__init__()
                self.frequency = frequency
                self.output_dir = output_dir

        def on_epoch_end(self, epoch, logs = None):
                if self.frequency <= 0:
                        return
                epoch_number = epoch + 1
                if epoch_number % self.frequency == 0:
                        filename = f"model_{epoch_number:04d}.keras"
                        self.model.save(os.path.join(self.output_dir, filename))


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
                PeriodicModelCheckpoint(MODEL_SAVE_FREQUENCY, MODELS_DIR),
                PeriodicComparisonSaver(
                        MODEL_SAVE_FREQUENCY,
                        RESULTS_DIR,
                        sample_batch_x,
                        sample_batch_y,
                        NUM_RESULT_EXAMPLES,
                ),
        ])