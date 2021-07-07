## ## Bethelhem S.: CVae - Py
## ## (Version 1.0.1, built: 2020-10-01)
## ## Copyright (C)2020 Bethelhem Seifu

## import library -----------------------------------------------
#import library

from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt





