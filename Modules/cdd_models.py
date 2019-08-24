# Cross-Domain Decompression Models
from keras import backend as K, regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Add, Subtract, LeakyReLU, \
                         MaxPooling2D, concatenate, UpSampling2D,\
                         Multiply, Dense, Flatten, Reshape
from keras.losses import mean_squared_error

def tf_dct2d(im):
    return K.tf.transpose(K.tf.spectral.dct(K.tf.transpose(K.tf.spectral.dct(im,type=2,axis=-1,norm='ortho'),perm=[0,2,1,3]),type=2,axis=-1,norm='ortho'),perm=[0,2,1,3])


def tf_idct2d(mat):
    return K.tf.transpose(K.tf.spectral.idct(K.tf.transpose(K.tf.spectral.idct(mat,type=2,axis=-1,norm='ortho'),perm=[0,2,1,3]),type=2,axis=-1,norm='ortho'),perm=[0,2,1,3])


# Modified from:
# https://www.programcreek.com/python/example/90422/tensorflow.extract_image_patches
def extract_patches(x, patsize=(1,8,8,1), strides=(1,8,8,1), rates=(1,1,1,1)):
    patches = K.tf.extract_image_patches(
        x,
        patsize,
        strides,
        rates,
        padding="SAME"
    )
    patches_shape = K.tf.shape(patches)
    return K.tf.reshape(patches,
                      [K.tf.reduce_prod(patches_shape[0:3]),
                       8, 8, 1])  # returns [batch_patches, h, w, c]


# x = orig
# y = orig in patches
def extract_patches_inverse(x, y):
    _x = K.tf.zeros_like(x)
    _y = extract_patches(_x)
    grad = K.tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return K.tf.gradients(_y, _x, grad_ys=y)[0] / grad

# dlist[0] = x; dlist[1] = y
def extract_patches_inv(dlist):
    _x = K.tf.zeros_like(dlist[0])
    _y = extract_patches(_x)
    grad = K.tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return K.tf.gradients(_y, _x, grad_ys=dlist[1])[0] / grad


def dc_loss(layer):
    def loss(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + mean_squared_error(layer, y_pred)

    return loss

def cust_loss(y_true, y_pred):
    return y_pred


def dct_layer(image):
    """
    Input: single channel array representing image domain data
    Output: single channel array representing dct coefficient data
    """

    patches = extract_patches(image)
    dct_patches = tf_dct2d(patches)
    dct_image = extract_patches_inverse(image, dct_patches)

    return dct_image


def idct_layer(dctdata):
    """
    Input: input DCT-domain data. See below
    Output: single channel array representing image domain data
    """

    patches = extract_patches(dctdata)
    image_patches = tf_idct2d(patches)
    image = extract_patches_inverse(dctdata, image_patches)

    return image


def dct_dc_layer(data_list): #, dctmat, jpeg_dctmat, qmat):
    """
    param data_list: data_list[0] = dct matrix;
                     data_list[1] = jpeg-compressed dct matrix;
                     data_list[2] = quantization matrix


    param dctmat: quantization matrix for each image
    param jpeg_dctmat: single channel array representing dct coefficient data
    param qmat: quantization matrix
    Output: dct coefficients restricted to range
    """
    # Clip range: X_recon = [(X_jpeg*Q - Q/2), (X_jpeg*Q + Q/2)]

    clip_low = Subtract()([data_list[1], data_list[2] * 0.5])
    clip_high = Add()([data_list[1], data_list[2] * 0.5])
    return K.tf.clip_by_value(data_list[0], clip_low, clip_high)

    #return K.tf.clip_by_value(dctmat, clip_low, clip_high)


def fc_dct_layer(image):
    """
    Input: single channel array representing image domain data
    Output: single channel array representing dct coefficient data (in patches)
    """

    patches = extract_patches(image)
    dct_patches = tf_dct2d(patches)

    return dct_patches


def fc_idct_layer(data_list):
    """
    Input: single channel array representing dct coefficient data (in patches)
    Output: single channel array representing image domain data

    data_list: data_list[0] = dct matrix (in patches) [total_num_patches, patch_size, patch_size, chnl]
               data_list[1] = single channel image in original dimensions [batch, H, W, chnl]
    """

    image_patches = tf_idct2d(data_list[0])
    image = extract_patches_inverse(data_list[1], image_patches)

    return image


def dense_block(dense_input, chnl=1):

    inputs_flat = Flatten(input_shape=dense_input.shape)(dense_input)

    dense1 = Dense(64*chnl, input_shape=inputs_flat.shape, activation='tanh')(inputs_flat)
    dense2 = Dense(64*chnl, activation='tanh')(dense1)
    dense3 = Dense(128*chnl, activation='tanh')(dense2)
    dense4 = Dense(128*chnl, activation='tanh')(dense3)
    dense5 = Dense(256*chnl, activation='tanh')(dense4)
    dense6 = Dense(256*chnl, activation='tanh')(dense5)
    dense7 = Dense(128*chnl, activation='tanh')(dense6)
    dense8 = Dense(128*chnl, activation='tanh')(dense7)
    dense9 = Dense(64*chnl, activation='tanh')(dense8)
    last_dense = Dense(64*chnl, activation='tanh')(dense9)

    #output_layer = Reshape(8,8)(dense4)
    output_layer = Reshape((8,8,chnl), input_shape=inputs_flat.shape)(last_dense)

    return output_layer


def unet_block(unet_input, kshape=(3, 3), chnl=1):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: single channel
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(chnl, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out


"""
==============================================================================================================
"""

def keras_mse(dlist):
    return mean_squared_error(dlist[0], dlist[1])


def residual_wnet_di_dc(H=256, W=256, kshape=(3, 3), channels=1):

    # channels = 1  # inputs are represented as single channel images (grayscale)
    inputs = Input(shape=(H, W, channels))
    dct_inputs = Lambda(dct_layer)(inputs)
    qmat = Input(shape=(H, W, channels))
    layers = [inputs]

    print("Append DCT layer")
    layers.append(Lambda(dct_layer)(layers[-1]))

    print("Append U-net block")
    predc_layer1 = unet_block(layers[-1], kshape, chnl=channels)
    layers.append(predc_layer1)

    print("Append DC")
    dc_layer1 = Lambda(dct_dc_layer)([layers[-1], dct_inputs, qmat])
    layers.append(dc_layer1)

    print("Append iDCT layer")
    intermed = Lambda(idct_layer)(layers[-1])
    layers.append(intermed)

    print("Append U-net block")
    layers.append(unet_block(layers[-1], kshape, chnl=channels))

    print("Append DCT-DC-iDCT layers")
    predc_layer2 = Lambda(dct_layer)(layers[-1])
    layers.append(predc_layer2)
    dc_layer2 = Lambda(dct_dc_layer)([layers[-1], dct_inputs, qmat])
    layers.append(dc_layer2)
    layers.append(Lambda(idct_layer)(layers[-1]))

    dc1 = Lambda(keras_mse)([dc_layer1, predc_layer1])
    dc2 = Lambda(keras_mse)([dc_layer2, predc_layer2])

    model = Model(inputs=[inputs,qmat], outputs=([layers[-1], intermed,\
                                                  dc1,\
                                                  dc2]))
    return model


def residual_wnet_id_dc(H=256, W=256, kshape=(3, 3), channels=1):

    # channels = 1  # inputs are represented as single channel images (grayscale)
    inputs = Input(shape=(H, W, channels))
    dct_inputs = Lambda(dct_layer)(inputs)
    qmat = Input(shape=(H, W, channels))
    layers = [inputs]

    print("Append U-net block")
    layers.append(unet_block(layers[-1], kshape, chnl=channels))

    print("Append DCT-DC-iDCT layers")
    predc_layer2 = Lambda(dct_layer)(layers[-1])
    layers.append(predc_layer2)
    dc_layer2 = Lambda(dct_dc_layer)([layers[-1], dct_inputs, qmat])
    layers.append(dc_layer2)
    intermed = Lambda(idct_layer)(layers[-1])
    layers.append(intermed)

    print("Append DCT layer")
    layers.append(Lambda(dct_layer)(layers[-1]))

    print("Append U-net block")
    predc_layer1 = unet_block(layers[-1], kshape, chnl=channels)
    layers.append(predc_layer1)

    print("Append DC")
    dc_layer1 = Lambda(dct_dc_layer)([layers[-1], dct_inputs, qmat])
    layers.append(dc_layer1)

    print("Append iDCT layer")
    layers.append(Lambda(idct_layer)(layers[-1]))

    dc1 = Lambda(keras_mse)([dc_layer1, predc_layer1])
    dc2 = Lambda(keras_mse)([dc_layer2, predc_layer2])

    model = Model(inputs=[inputs,qmat], outputs=([layers[-1], intermed,\
                                                  dc1,\
                                                  dc2]))
    return model


# def fcu_di_dc(H=256,W=256,kshape=(3,3),chnl=1):
#
#     channels = 1
#     inputs = Input(shape=(H,W,channels))
#     qmat = Input(shape=(H,W,channels))
#
#     dct_inputs = Lambda(dct_layer)(inputs)
#
#     layers = [inputs]
#
#     print("Append DCT layer (patches)")
#     layers.append(Lambda(fc_dct_layer)(layers[-1]))
#     print("Append FC/dense block")
#     layers.append(dense_block(layers[-1]))
#     predc_layer1 = Lambda(extract_patches_inv)([inputs, layers[-1]])
#     layers.append(predc_layer1)
#     print("Append data consistency")
#     dc_layer1 = Lambda(dct_dc_layer)([predc_layer1, dct_inputs, qmat])
#     layers.append(dc_layer1)
#     print("Append iDCT layer")
#     intermed = Lambda(idct_layer)(layers[-1])
#     layers.append(intermed)
#
#     print("Append U-net block")
#     layers.append(unet_block(layers[-1], kshape))
#     print("Append DCT layer")
#     predc_layer2 = Lambda(dct_layer)(layers[-1])
#     layers.append(predc_layer2)
#     print("Append data consistency")
#     dc_layer2 = Lambda(dct_dc_layer)([predc_layer2, dct_inputs, qmat])
#     layers.append(dc_layer2)
#     print("Append iDCT layer")
#     layers.append(Lambda(idct_layer)(dc_layer2))
#
#     dc1 = Lambda(keras_mse)([dc_layer1, predc_layer1])
#     dc2 = Lambda(keras_mse)([dc_layer2, predc_layer2])
#
#     model = Model(inputs=[inputs,qmat], outputs=[layers[-1],intermed,dc1,dc2])
#     return model
#
#
# def fcu_id_dc(H=256,W=256,kshape=(3,3),chnl=1):
#
#     channels = 1
#     inputs = Input(shape=(H,W,channels))
#     qmat = Input(shape=(H,W,channels))
#
#     dct_inputs = Lambda(dct_layer)(inputs)
#
#     layers = [inputs]
#
#     print("Append U-net block")
#     layers.append(unet_block(layers[-1], kshape))
#     print("Append DCT layer")
#     predc_layer1 = Lambda(dct_layer)(layers[-1])
#     layers.append(predc_layer1)
#     print("Append data consistency")
#     dc_layer1 = Lambda(dct_dc_layer)([predc_layer1, dct_inputs, qmat])
#     layers.append(dc_layer1)
#     print("Append iDCT layer")
#     intermed = Lambda(idct_layer)(dc_layer1)
#     layers.append(intermed)
#
#     print("Append DCT layer (patches)")
#     layers.append(Lambda(fc_dct_layer)(layers[-1]))
#     print("Append FC/dense block")
#     layers.append(dense_block(layers[-1]))
#     predc_layer2 = Lambda(extract_patches_inv)([inputs, layers[-1]])
#     layers.append(predc_layer2)
#     print("Append data consistency")
#     dc_layer2 = Lambda(dct_dc_layer)([predc_layer2, dct_inputs, qmat])
#     layers.append(dc_layer2)
#     print("Append iDCT layer")
#     layers.append(Lambda(idct_layer)(layers[-1]))
#
#     dc1 = Lambda(keras_mse)([dc_layer1, predc_layer1])
#     dc2 = Lambda(keras_mse)([dc_layer2, predc_layer2])
#
#     model = Model(inputs=[inputs,qmat], outputs=[layers[-1],intermed,dc1,dc2])
#     return model
#
#
def deep_cascade_unet(depth_str='di', H=256, W=256, kshape=(3, 3), useDC = False):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image height
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    channels = 1  # inputs are represented as single channel images (grayscale)
    inputs = Input(shape=(H, W, channels))

    if useDC:
        qmat = Input(shape=(H, W, channels))
    layers = [inputs]
    dct_flag = False # flag whether input is in the image domain (false) or dct domain (true)

    # intermed_list = []

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'd':
            if not dct_flag: # if in image domain
                # Apply DCT
                layers.append(Lambda(dct_layer)(layers[-1]))
                print("Append DCT layer")
                dct_flag = True
        elif ii == 'i':
            if dct_flag: # if in dct domain
                layers.append(Lambda(idct_layer)(layers[-1]))
                print("Append iDCT layer")
                dct_flag = False
        else:
            print("Layer not recognized. Only 'd' and 'i' layers are currently supported.")
            break;

        # Append 1-channel U-net block
        layers.append(unet_block(layers[-1], kshape))
        print("Append U-net block")

        if useDC:
            # Append a data consistency layer
            print("Data consistency layer")
            if dct_flag: # if in the DCT domain
                print("Currently in the DCT domain")
                print("Append DC layer")
                print("Append iDCT layer")
                layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat]))
                layers.append(Lambda(idct_layer)(layers[-1]))
                dct_flag = False
            else: # if in the image domain
                print("Currently in the image domain")
                print("Append DCT-DC-iDCT layers")
                layers.append(Lambda(dct_layer)(layers[-1]))
                layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat]))
                layers.append(Lambda(idct_layer)(layers[-1]))

            if jj == len(depth_str) - 2:
                intermed = layers[-1]
        else:
            intermed = layers[-1]

        if not (jj + 1) % len(depth_str) > 0: # check if there is a next element
            if dct_flag: # if in DCT domain
                layers.append(Lambda(idct_layer)(layers[-1]))
                print("Append iDCT layer")
                dct_flag = False

        #intermed_list = [intermed_list, intermed]

    if useDC:
        inputs=[inputs,qmat]
        outputs = [layers[-1], intermed]
    else:
        inputs=[inputs]
        outputs = [layers[-1], layers[1]]

    model = Model(inputs=inputs, outputs=outputs)
    return model


def fcu_di_dc(H=256,W=256,kshape=(3,3),chnl=1):

#     channels = 1
#     inputs = Input(shape=(H,W,channels))
#     qmat = Inputs(shape=(H,W,channels))

#     dct_inputs = Lambda(dct_layer)(inputs)
#     layers = [inputs]

#     print("Append DCT layer (patches)")
#     layers.append(Lambda(fc_dct_layer)(layers[-1]))
#     print("Append FC/dense block")
#     layers.append(dense_block(layers[-1]))

    channels = 1
    inputs = Input(shape=(H,W,channels))
    qmat = Input(shape=(H,W,channels))

    dct_inputs = Lambda(dct_layer)(inputs)

    layers = [inputs]

    print("Append DCT layer (patches)")
    layers.append(Lambda(fc_dct_layer)(layers[-1]))
    print("Append FC/dense block")
    layers.append(dense_block(layers[-1]))
    predc_layer1 = Lambda(extract_patches_inv)([inputs, layers[-1]])
    layers.append(predc_layer1)
    print("Append data consistency")
    dc_layer1 = Lambda(dct_dc_layer)([predc_layer1, dct_inputs, qmat])
    layers.append(dc_layer1)
    print("Append iDCT layer")
    intermed = Lambda(idct_layer)(layers[-1])
    layers.append(intermed)

    print("Append U-net block")
    layers.append(unet_block(layers[-1], kshape))
    print("Append DCT layer")
    predc_layer2 = Lambda(dct_layer)(layers[-1])
    layers.append(predc_layer2)
    print("Append data consistency")
    dc_layer2 = Lambda(dct_dc_layer)([predc_layer2, dct_inputs, qmat])
    layers.append(dc_layer2)
    print("Append iDCT layer")
    layers.append(Lambda(idct_layer)(dc_layer2))

    dc1 = Lambda(keras_mse)([dc_layer1, predc_layer1])
    dc2 = Lambda(keras_mse)([dc_layer2, predc_layer2])

    model = Model(inputs=[inputs,qmat], outputs=[layers[-1],intermed,dc1,dc2])
    return model


def fcu_id_dc(H=256,W=256,kshape=(3,3),chnl=1):

#     channels = 1
#     inputs = Input(shape=(H,W,channels))
#     qmat = Inputs(shape=(H,W,channels))

#     dct_inputs = Lambda(dct_layer)(inputs)
#     layers = [inputs]

#     print("Append DCT layer (patches)")
#     layers.append(Lambda(fc_dct_layer)(layers[-1]))
#     print("Append FC/dense block")
#     layers.append(dense_block(layers[-1]))

    channels = 1
    inputs = Input(shape=(H,W,channels))
    qmat = Input(shape=(H,W,channels))

    dct_inputs = Lambda(dct_layer)(inputs)

    layers = [inputs]

    print("Append U-net block")
    layers.append(unet_block(layers[-1], kshape))
    print("Append DCT layer")
    predc_layer1 = Lambda(dct_layer)(layers[-1])
    layers.append(predc_layer1)
    print("Append data consistency")
    dc_layer1 = Lambda(dct_dc_layer)([predc_layer1, dct_inputs, qmat])
    layers.append(dc_layer1)
    print("Append iDCT layer")
    intermed = Lambda(idct_layer)(dc_layer1)
    layers.append(intermed)

    print("Append DCT layer (patches)")
    layers.append(Lambda(fc_dct_layer)(layers[-1]))
    print("Append FC/dense block")
    layers.append(dense_block(layers[-1]))
    predc_layer2 = Lambda(extract_patches_inv)([inputs, layers[-1]])
    layers.append(predc_layer2)
    print("Append data consistency")
    dc_layer2 = Lambda(dct_dc_layer)([predc_layer2, dct_inputs, qmat])
    layers.append(dc_layer2)
    print("Append iDCT layer")
    layers.append(Lambda(idct_layer)(layers[-1]))

    dc1 = Lambda(keras_mse)([dc_layer1, predc_layer1])
    dc2 = Lambda(keras_mse)([dc_layer2, predc_layer2])

    model = Model(inputs=[inputs,qmat], outputs=[layers[-1],intermed,dc1,dc2])
    return model
