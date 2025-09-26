from keras.applications import InceptionV3

model = InceptionV3(weights="imagenet", include_top=False)
model.save("model.keras")
print("saved")

