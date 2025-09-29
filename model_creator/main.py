from keras.models import Model
from keras.layers import Activation, Input
from keras.layers import (Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, BatchNormalization)
from model_creator.misc import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

if MODEL_NAME == "big_model":
        # Define the encoder
        def conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)
                x = Activation("relu", name = f"{block_name}_relu2")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv3")(x)
                x = BatchNormalization(name = f"{block_name}_bn3")(x)
                x = Activation("relu", name = f"{block_name}_relu3")(x)
                return x

        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = conv_block(encoder_input, 64, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)
        x = conv_block(x, 128, "enc_block2")
        x = MaxPooling2D(name = "enc_pool2")(x)
        x = conv_block(x, 256, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = conv_block(x, 512, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)
        x = Conv2D(512, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
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
        x = Conv2D(512, (3, 3), padding = "same", name = "dec_bottleneck_conv")(x)
        x = BatchNormalization(name = "dec_bottleneck_bn")(x)
        x = Activation("relu", name = "dec_bottleneck_relu")(x)
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = conv_block(x, 512, "dec_block1")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = conv_block(x, 256, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = conv_block(x, 128, "dec_block3")
        x = UpSampling2D(name = "dec_upsample4")(x)
        x = conv_block(x, 64, "dec_block4")
        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "medium_model":
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

elif MODEL_NAME == "small_model":
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

else:
        print("Model not found", MODEL_NAME)


# Combine encoder and decoder into an autoencoder
autoencoder_output = decoder(encoder(encoder_input))
model = Model(encoder_input, autoencoder_output, name = "autoencoder")
model.compile(optimizer = "adam", loss = "mse")

os.makedirs(MODELS_DIR, exist_ok = True)
os.makedirs(RESULTS_DIR, exist_ok = True)
save_models(encoder, decoder, MODELS_DIR, 0)

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

average_difference_tracker = AverageDifferenceTracker(
        RESULTS_DIR,
        comparison_batch_x,
        comparison_batch_y,
        len(comparison_batch_x),
)

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
                average_difference_tracker,
        ])

loss_plot_path = os.path.join(RESULTS_DIR, "loss_curve.jpg")
save_loss_plot(history, loss_plot_path)