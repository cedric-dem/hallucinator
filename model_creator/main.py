import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
# Ancien import (ne marche plus)
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Nouveau import correct
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Define the model
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(16, (3,3), padding="same"), Activation("relu"),
    MaxPooling2D(),
    Conv2D(2, (3,3), padding="same"), Activation("relu"),
    MaxPooling2D(),
    Conv2D(2, (3,3), padding="same"), Activation("relu"),
    UpSampling2D(),
    Conv2D(16, (3,3), padding="same"), Activation("relu"),
    UpSampling2D(),
    Conv2D(3, (3,3), padding="same"), Activation("sigmoid"),
])
model.compile(optimizer="adam", loss="mse")


model.summary()


# Generate data from the images in a folder
batch_size = 8
train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory(
    'cropped/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input'
    )
test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
validation_generator = test_datagen.flow_from_directory(
    'cropped/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input'
    )
    
# Train the model
model.fit(
        train_generator,
        steps_per_epoch=1000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=1000 // batch_size)
        

validation_generator.reset()  # remet l’index à 0
batch_x, batch_y = next(validation_generator)  # récupère un batch

# Prédiction
predicted = model.predict(batch_x)

# Affichage (les données sont déjà [0,1] grâce à rescale=1./255)
import matplotlib.pyplot as plt

plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.title("Entrée");      plt.axis("off"); plt.imshow(batch_x[0])
plt.subplot(1,3,2); plt.title("Cible");       plt.axis("off"); plt.imshow(batch_y[0])
plt.subplot(1,3,3); plt.title("Sortie modèle"); plt.axis("off"); plt.imshow(predicted[0])
plt.tight_layout()
plt.show()