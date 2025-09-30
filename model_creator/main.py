from keras.models import Model
from keras.layers import Activation, Input
from keras.layers import (
        Conv2D,
        Flatten,
        MaxPooling2D,
        Reshape,
        UpSampling2D,
        BatchNormalization,
        Add,
)
from model_creator.misc import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *
import shutil

if MODEL_NAME == "6272_3_big_model":
        def residual_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)

                shortcut = layer_input
                shortcut_channels = layer_input.shape[-1]
                needs_projection = shortcut_channels is None or int(shortcut_channels) != filters
                if needs_projection:
                        shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_proj_conv")(shortcut)
                        shortcut = BatchNormalization(name = f"{block_name}_proj_bn")(shortcut)

                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = Conv2D(64, (3, 3), padding = "same", name = "enc_stem_conv")(encoder_input)
        x = BatchNormalization(name = "enc_stem_bn")(x)
        x = Activation("relu", name = "enc_stem_relu")(x)

        x = residual_block(x, 64, "enc_block1_res1")
        x = residual_block(x, 64, "enc_block1_res2")
        x = MaxPooling2D(name = "enc_pool1")(x)

        x = residual_block(x, 96, "enc_block2_res1")
        x = residual_block(x, 96, "enc_block2_res2")
        x = MaxPooling2D(name = "enc_pool2")(x)

        x = residual_block(x, 128, "enc_block3_res1")
        x = residual_block(x, 128, "enc_block3_res2")
        x = MaxPooling2D(name = "enc_pool3")(x)

        x = residual_block(x, 160, "enc_block4_res1")
        x = residual_block(x, 160, "enc_block4_res2")
        x = MaxPooling2D(name = "enc_pool4")(x)

        x = Conv2D(160, (3, 3), padding = "same", name = "enc_bottleneck_conv1")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn1")(x)
        x = Activation("relu", name = "enc_bottleneck_relu1")(x)
        x = Conv2D(32, (1, 1), padding = "same", name = "enc_bottleneck_conv2")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn2")(x)
        x = Activation("relu", name = "enc_bottleneck_relu2")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)

        x = Conv2D(160, (1, 1), padding = "same", name = "dec_bottleneck_expand")(x)
        x = BatchNormalization(name = "dec_bottleneck_expand_bn")(x)
        x = Activation("relu", name = "dec_bottleneck_expand_relu")(x)

        x = residual_block(x, 160, "dec_block4_res1")
        x = residual_block(x, 160, "dec_block4_res2")
        x = UpSampling2D(name = "dec_upsample1")(x)

        x = residual_block(x, 128, "dec_block3_res1")
        x = residual_block(x, 128, "dec_block3_res2")
        x = UpSampling2D(name = "dec_upsample2")(x)

        x = residual_block(x, 96, "dec_block2_res1")
        x = residual_block(x, 96, "dec_block2_res2")
        x = UpSampling2D(name = "dec_upsample3")(x)

        x = residual_block(x, 64, "dec_block1_res1")
        x = residual_block(x, 64, "dec_block1_res2")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(48, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "6272_2_medium_model":
        def residual_conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)

                shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_shortcut_conv")(layer_input)
                shortcut = BatchNormalization(name = f"{block_name}_shortcut_bn")(shortcut)

                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = residual_conv_block(encoder_input, 48, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)
        x = residual_conv_block(x, 96, "enc_block2")
        x = MaxPooling2D(name = "enc_pool2")(x)
        x = residual_conv_block(x, 128, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = residual_conv_block(x, 64, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)
        x = Conv2D(64, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn")(x)
        x = Activation("relu", name = "enc_bottleneck_relu")(x)
        x = Conv2D(32, (1, 1), padding = "same", name = "enc_projection_conv")(x)
        x = BatchNormalization(name = "enc_projection_bn")(x)
        x = Activation("relu", name = "enc_projection_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)
        x = Conv2D(64, (1, 1), padding = "same", name = "dec_projection_expand")(x)
        x = BatchNormalization(name = "dec_projection_expand_bn")(x)
        x = Activation("relu", name = "dec_projection_expand_relu")(x)

        x = residual_conv_block(x, 64, "dec_block4")
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = residual_conv_block(x, 128, "dec_block3")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = residual_conv_block(x, 96, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = residual_conv_block(x, 64, "dec_block1")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(48, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "6272_1_medium_small_model":
        def conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)
                x = Activation("relu", name = f"{block_name}_relu2")(x)
                return x

        def residual_block(layer_input, filters, block_name):
                x = conv_block(layer_input, filters, f"{block_name}_main")
                shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_shortcut_conv")(layer_input)
                shortcut = BatchNormalization(name = f"{block_name}_shortcut_bn")(shortcut)
                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = conv_block(encoder_input, 40, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)
        x = residual_block(x, 80, "enc_block2")
        x = MaxPooling2D(name = "enc_pool2")(x)
        x = residual_block(x, 96, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = conv_block(x, 48, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)
        x = Conv2D(48, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn")(x)
        x = Activation("relu", name = "enc_bottleneck_relu")(x)
        x = Conv2D(32, (1, 1), padding = "same", name = "enc_projection_conv")(x)
        x = BatchNormalization(name = "enc_projection_bn")(x)
        x = Activation("relu", name = "enc_projection_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)
        x = Conv2D(48, (1, 1), padding = "same", name = "dec_projection_expand")(x)
        x = BatchNormalization(name = "dec_projection_expand_bn")(x)
        x = Activation("relu", name = "dec_projection_expand_relu")(x)

        x = conv_block(x, 48, "dec_block4")
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = residual_block(x, 96, "dec_block3")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = residual_block(x, 80, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = conv_block(x, 40, "dec_block1")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(32, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "6272_0_small_model":
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
        x = conv_block(x, 64, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = conv_block(x, 32, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)
        x = Conv2D(32, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
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
        x = Conv2D(32, (3, 3), padding = "same", name = "dec_bottleneck_conv")(x)
        x = BatchNormalization(name = "dec_bottleneck_bn")(x)
        x = Activation("relu", name = "dec_bottleneck_relu")(x)
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = conv_block(x, 32, "dec_block1")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = conv_block(x, 64, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = conv_block(x, 64, "dec_block3")
        x = UpSampling2D(name = "dec_upsample4")(x)
        x = conv_block(x, 32, "dec_block4")
        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")


elif MODEL_NAME == "9604_0_small_model":
        def conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)
                x = Activation("relu", name = f"{block_name}_relu2")(x)
                return x

        def residual_block(layer_input, filters, block_name):
                x = conv_block(layer_input, filters, f"{block_name}_main")

                shortcut = layer_input
                shortcut_channels = shortcut.shape[-1]
                needs_projection = shortcut_channels is None or int(shortcut_channels) != filters
                if needs_projection:
                        shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_shortcut_conv")(shortcut)
                        shortcut = BatchNormalization(name = f"{block_name}_shortcut_bn")(shortcut)

                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = conv_block(encoder_input, 48, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)
        x = residual_block(x, 96, "enc_block2")
        x = MaxPooling2D(name = "enc_pool2")(x)
        x = residual_block(x, 112, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = conv_block(x, 64, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)
        x = Conv2D(72, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn")(x)
        x = Activation("relu", name = "enc_bottleneck_relu")(x)
        x = Conv2D(49, (1, 1), padding = "same", name = "enc_projection_conv")(x)
        x = BatchNormalization(name = "enc_projection_bn")(x)
        x = Activation("relu", name = "enc_projection_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)
        x = Conv2D(72, (1, 1), padding = "same", name = "dec_projection_expand")(x)
        x = BatchNormalization(name = "dec_projection_expand_bn")(x)
        x = Activation("relu", name = "dec_projection_expand_relu")(x)

        x = conv_block(x, 64, "dec_block4")
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = residual_block(x, 112, "dec_block3")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = residual_block(x, 96, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = conv_block(x, 48, "dec_block1")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(40, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "9604_1_small_medium_model":
        def conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)
                x = Activation("relu", name = f"{block_name}_relu2")(x)
                return x

        def residual_block(layer_input, filters, block_name):
                x = conv_block(layer_input, filters, f"{block_name}_main")
                shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_shortcut_conv")(layer_input)
                shortcut = BatchNormalization(name = f"{block_name}_shortcut_bn")(shortcut)
                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = conv_block(encoder_input, 56, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)
        x = residual_block(x, 112, "enc_block2")
        x = MaxPooling2D(name = "enc_pool2")(x)
        x = residual_block(x, 128, "enc_block3")
        x = MaxPooling2D(name = "enc_pool3")(x)
        x = conv_block(x, 72, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)
        x = Conv2D(80, (3, 3), padding = "same", name = "enc_bottleneck_conv")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn")(x)
        x = Activation("relu", name = "enc_bottleneck_relu")(x)
        x = Conv2D(49, (1, 1), padding = "same", name = "enc_projection_conv")(x)
        x = BatchNormalization(name = "enc_projection_bn")(x)
        x = Activation("relu", name = "enc_projection_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)
        x = Conv2D(80, (1, 1), padding = "same", name = "dec_projection_expand")(x)
        x = BatchNormalization(name = "dec_projection_expand_bn")(x)
        x = Activation("relu", name = "dec_projection_expand_relu")(x)

        x = conv_block(x, 72, "dec_block4")
        x = UpSampling2D(name = "dec_upsample1")(x)
        x = residual_block(x, 128, "dec_block3")
        x = UpSampling2D(name = "dec_upsample2")(x)
        x = residual_block(x, 112, "dec_block2")
        x = UpSampling2D(name = "dec_upsample3")(x)
        x = conv_block(x, 56, "dec_block1")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(48, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "9604_2_medium_model":
        def conv_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)
                x = Activation("relu", name = f"{block_name}_relu2")(x)
                return x

        def residual_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)

                shortcut = layer_input
                shortcut_channels = shortcut.shape[-1]
                needs_projection = shortcut_channels is None or int(shortcut_channels) != filters
                if needs_projection:
                        shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_shortcut_conv")(shortcut)
                        shortcut = BatchNormalization(name = f"{block_name}_shortcut_bn")(shortcut)

                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x


        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = conv_block(encoder_input, 64, "enc_block1")
        x = MaxPooling2D(name = "enc_pool1")(x)

        x = residual_block(x, 128, "enc_block2_res1")
        x = residual_block(x, 128, "enc_block2_res2")
        x = MaxPooling2D(name = "enc_pool2")(x)

        x = residual_block(x, 160, "enc_block3_res1")
        x = residual_block(x, 160, "enc_block3_res2")
        x = MaxPooling2D(name = "enc_pool3")(x)

        x = conv_block(x, 96, "enc_block4")
        x = MaxPooling2D(name = "enc_pool4")(x)

        x = Conv2D(112, (3, 3), padding = "same", name = "enc_bottleneck_conv1")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn1")(x)
        x = Activation("relu", name = "enc_bottleneck_relu1")(x)
        x = Conv2D(80, (3, 3), padding = "same", name = "enc_bottleneck_conv2")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn2")(x)
        x = Activation("relu", name = "enc_bottleneck_relu2")(x)
        x = Conv2D(49, (1, 1), padding = "same", name = "enc_projection_conv")(x)
        x = BatchNormalization(name = "enc_projection_bn")(x)
        x = Activation("relu", name = "enc_projection_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)
        x = Conv2D(80, (1, 1), padding = "same", name = "dec_projection_expand")(x)
        x = BatchNormalization(name = "dec_projection_expand_bn")(x)
        x = Activation("relu", name = "dec_projection_expand_relu")(x)
        x = Conv2D(112, (3, 3), padding = "same", name = "dec_bottleneck_conv1")(x)
        x = BatchNormalization(name = "dec_bottleneck_bn1")(x)
        x = Activation("relu", name = "dec_bottleneck_relu1")(x)

        x = conv_block(x, 96, "dec_block4")
        x = UpSampling2D(name = "dec_upsample1")(x)

        x = residual_block(x, 160, "dec_block3_res1")
        x = residual_block(x, 160, "dec_block3_res2")
        x = UpSampling2D(name = "dec_upsample2")(x)

        x = residual_block(x, 128, "dec_block2_res1")
        x = residual_block(x, 128, "dec_block2_res2")
        x = UpSampling2D(name = "dec_upsample3")(x)

        x = conv_block(x, 64, "dec_block1")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(48, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

elif MODEL_NAME == "9604_3_big_model":
        def residual_block(layer_input, filters, block_name):
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv1")(layer_input)
                x = BatchNormalization(name = f"{block_name}_bn1")(x)
                x = Activation("relu", name = f"{block_name}_relu1")(x)
                x = Conv2D(filters, (3, 3), padding = "same", name = f"{block_name}_conv2")(x)
                x = BatchNormalization(name = f"{block_name}_bn2")(x)

                shortcut = layer_input
                shortcut_channels = shortcut.shape[-1]
                needs_projection = shortcut_channels is None or int(shortcut_channels) != filters
                if needs_projection:
                        shortcut = Conv2D(filters, (1, 1), padding = "same", name = f"{block_name}_proj_conv")(shortcut)
                        shortcut = BatchNormalization(name = f"{block_name}_proj_bn")(shortcut)

                x = Add(name = f"{block_name}_add")([x, shortcut])
                x = Activation("relu", name = f"{block_name}_out")(x)
                return x

        encoder_input = Input(shape = (IMG_DIM, IMG_DIM, 3), name = "encoder_input")
        x = Conv2D(72, (3, 3), padding = "same", name = "enc_stem_conv")(encoder_input)
        x = BatchNormalization(name = "enc_stem_bn")(x)
        x = Activation("relu", name = "enc_stem_relu")(x)

        x = residual_block(x, 72, "enc_block1_res1")
        x = residual_block(x, 72, "enc_block1_res2")
        x = residual_block(x, 72, "enc_block1_res3")
        x = MaxPooling2D(name = "enc_pool1")(x)

        x = residual_block(x, 112, "enc_block2_res1")
        x = residual_block(x, 112, "enc_block2_res2")
        x = residual_block(x, 112, "enc_block2_res3")
        x = MaxPooling2D(name = "enc_pool2")(x)

        x = residual_block(x, 144, "enc_block3_res1")
        x = residual_block(x, 144, "enc_block3_res2")
        x = residual_block(x, 144, "enc_block3_res3")
        x = MaxPooling2D(name = "enc_pool3")(x)

        x = residual_block(x, 192, "enc_block4_res1")
        x = residual_block(x, 192, "enc_block4_res2")
        x = residual_block(x, 192, "enc_block4_res3")
        x = MaxPooling2D(name = "enc_pool4")(x)

        x = Conv2D(192, (3, 3), padding = "same", name = "enc_bottleneck_conv1")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn1")(x)
        x = Activation("relu", name = "enc_bottleneck_relu1")(x)
        x = Conv2D(96, (1, 1), padding = "same", name = "enc_bottleneck_conv2")(x)
        x = BatchNormalization(name = "enc_bottleneck_bn2")(x)
        x = Activation("relu", name = "enc_bottleneck_relu2")(x)
        x = Conv2D(49, (1, 1), padding = "same", name = "enc_projection_conv")(x)
        x = BatchNormalization(name = "enc_projection_bn")(x)
        x = Activation("relu", name = "enc_projection_relu")(x)

        encoded_feature_map_shape = tuple(int(dimension) for dimension in x.shape[1:])
        encoded = Flatten(name = "encoder_flatten")(x)

        encoder = Model(encoder_input, encoded, name = "encoder")
        encoded_vector_length = int(np.prod(encoded_feature_map_shape))
        print(f"Encoded vector length: {encoded_vector_length}")

        decoder_input = Input(shape = (encoded_vector_length,), name = "decoder_input")
        x = Reshape(encoded_feature_map_shape, name = "decoder_reshape")(decoder_input)

        x = Conv2D(96, (1, 1), padding = "same", name = "dec_bottleneck_expand")(x)
        x = BatchNormalization(name = "dec_bottleneck_expand_bn")(x)
        x = Activation("relu", name = "dec_bottleneck_expand_relu")(x)

        x = residual_block(x, 192, "dec_block4_res1")
        x = residual_block(x, 192, "dec_block4_res2")
        x = residual_block(x, 192, "dec_block4_res3")
        x = UpSampling2D(name = "dec_upsample1")(x)

        x = residual_block(x, 144, "dec_block3_res1")
        x = residual_block(x, 144, "dec_block3_res2")
        x = residual_block(x, 144, "dec_block3_res3")
        x = UpSampling2D(name = "dec_upsample2")(x)

        x = residual_block(x, 112, "dec_block2_res1")
        x = residual_block(x, 112, "dec_block2_res2")
        x = residual_block(x, 112, "dec_block2_res3")
        x = UpSampling2D(name = "dec_upsample3")(x)

        x = residual_block(x, 72, "dec_block1_res1")
        x = residual_block(x, 72, "dec_block1_res2")
        x = residual_block(x, 72, "dec_block1_res3")
        x = UpSampling2D(name = "dec_upsample4")(x)

        x = Conv2D(56, (3, 3), padding = "same", name = "dec_final_conv")(x)
        x = BatchNormalization(name = "dec_final_bn")(x)
        x = Activation("relu", name = "dec_final_relu")(x)

        decoded = Conv2D(3, (3, 3), padding = "same", name = "decoder_output_conv")(x)
        decoded = Activation("sigmoid", name = "decoder_output")(decoded)

        decoder = Model(decoder_input, decoded, name = "decoder")

else:
        print("Model not found", MODEL_NAME)

if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

# Combine encoder and decoder into an autoencoder
autoencoder_output = decoder(encoder(encoder_input))
model = Model(encoder_input, autoencoder_output, name = "autoencoder")
model.compile(optimizer = "adam", loss = "mse")

MODELS_DIR.mkdir(parents = True, exist_ok = True)
RESULTS_DIR.mkdir(parents = True, exist_ok = True)
save_models(encoder, decoder, MODELS_DIR, 0)

model.summary()

# Generate data from the images in a folder
batch_size = 8
train_datagen = ImageDataGenerator(rescale = 1. / 255, data_format = 'channels_last')
train_generator = train_datagen.flow_from_directory(
        str(TRAINING_DATA_DIR),
        target_size = (IMG_DIM, IMG_DIM),
        batch_size = batch_size,
        class_mode = 'input'
)
test_datagen = ImageDataGenerator(rescale = 1. / 255, data_format = 'channels_last')
validation_generator = test_datagen.flow_from_directory(
        str(TRAINING_DATA_DIR),
        target_size = (IMG_DIM, IMG_DIM),
        batch_size = batch_size,
        class_mode = 'input'
)

comparison_images = load_comparison_images(COMPARISON_SOURCE_IMAGES_DIR, (IMG_DIM, IMG_DIM))

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

epoch_time_tracker = EpochTimeTracker(RESULTS_PLOTS_DIR / EPOCH_TIME_PLOT_FILENAME)

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
                epoch_time_tracker,
        ])

loss_plot_path = RESULTS_PLOTS_DIR / LOSS_PLOT_FILENAME
save_loss_plot(history, loss_plot_path)