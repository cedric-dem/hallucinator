import os

import keras
from keras.models import Sequential
from keras.layers import Activation, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_SAVE_FREQUENCY = 10
MODELS_DIR = "models"

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
model.save(os.path.join(MODELS_DIR, "model_0000.keras"))

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

# Train the model
model.fit(
	train_generator,
	steps_per_epoch = 1000 // batch_size,
	epochs = 20,
	validation_data = validation_generator,
	validation_steps = 1000 // batch_size,
	callbacks = [PeriodicModelCheckpoint(MODEL_SAVE_FREQUENCY, MODELS_DIR)])

validation_generator.reset()
batch_x, batch_y = next(validation_generator)

predicted = model.predict(batch_x)

plt.figure(figsize = (9, 3))
plt.subplot(1, 3, 1)
plt.title("Input")
plt.axis("off")
plt.imshow(batch_x[0])
plt.subplot(1, 3, 2)
plt.title("Target")
plt.axis("off")
plt.imshow(batch_y[0])
plt.subplot(1, 3, 3)
plt.title("Model Output")
plt.axis("off")
plt.imshow(predicted[0])
plt.tight_layout()
plt.show()