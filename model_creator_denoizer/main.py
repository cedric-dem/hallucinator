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

#TODO