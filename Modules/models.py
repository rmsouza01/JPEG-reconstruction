from __future__ import division, print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Add, MaxPooling2D, concatenate, UpSampling2D, Dropout


def unet(input_size = (256 ,256 ,1),drop = 0.5,residual = False):
    """
    :param input_size: Image input size
    :param drop: Dropout rate
    :param residual: Residual flag: True = on, False = off
    :return: reconstructed, de-noised image. It really depends on the data you feed.
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(drop)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(drop)(conv5)

    up6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') \
        (UpSampling2D(size = (2 ,2))(drop5))
    merge6 = concatenate([drop4 ,up6], axis = 3)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') \
        (UpSampling2D(size = (2 ,2))(conv6))
    merge7 = concatenate([conv3 ,up7], axis = 3)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') \
        (UpSampling2D(size = (2 ,2))(conv7))
    merge8 = concatenate([conv2 ,up8], axis = 3)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') \
        (UpSampling2D(size = (2 ,2))(conv8))
    merge9 = concatenate([conv1 ,up9], axis = 3)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(input_size[-1], (1, 1), activation='linear')(conv9)

    if residual:
        out = Add()([conv10,inputs])
    else:
        out = conv10
    model = Model(inputs = inputs, outputs = out)

    return model