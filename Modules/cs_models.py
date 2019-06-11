from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Add, Subtract, LeakyReLU, \
                         MaxPooling2D, concatenate, UpSampling2D,\
                         Multiply, Dense, Flatten, Reshape


def tf_dct2d(im):
    return K.tf.transpose(K.tf.spectral.dct(K.tf.transpose(K.tf.spectral.dct(im,type=2,axis=-1,norm='ortho'),perm=[0,2,1,3]),type=2,axis=-1,norm='ortho'),perm=[0,2,1,3])
    
    
def tf_idct2d(mat): 
    return K.tf.transpose(K.tf.spectral.idct(K.tf.transpose(K.tf.spectral.idct(mat,type=2,axis=-1,norm='ortho'),perm=[0,2,1,3]),type=2,axis=-1,norm='ortho'),perm=[0,2,1,3])


# Modified from:
# https://www.programcreek.com/python/example/90422/tensorflow.extract_image_patches
def extract_patches(x):
    patches = K.tf.extract_image_patches(
        x,
        (1, 8, 8, 1),
        (1, 8, 8, 1),
        (1, 1, 1, 1),
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


def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Squared Error (NRMSE) - Euclidean distance normalization
    :param y_true: Reference
    :param y_pred: Predicted
    :return:
    """

    denom = K.max(y_true, axis=(1,2,3)) - K.min(y_true, axis=(1,2,3))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom


def nrmse_min_max(y_true, y_pred):
    """
     Normalized Root Mean Squared Error (NRMSE) - min-max normalization
     :param y_true: Reference
     :param y_pred: Predicted
     :return:
     """

    denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom


def fft_layer(image):
    """
    Input: 2-channel array representing image domain complex data
    Output: 2-channel array representing k-space complex data
    """

    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = K.tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = K.tf.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(kspace_complex), -1)
    imag = K.tf.expand_dims(K.tf.imag(kspace_complex), -1)

    # generate 2-channel representation of k-space
    kspace = K.tf.concat([real, imag], -1)
    return kspace


def ifft_layer(kspace_2channel):
    """
    Input: 2-channel array representing k-space
    Output: 2-channel array representing image domain
    """
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)
    
    kspace_complex = K.tf.complex(real,imag) # Make complex-valued tensor
    image_complex = K.tf.ifft2d(kspace_complex)
    
    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(image_complex),-1)
    imag = K.tf.expand_dims(K.tf.imag(image_complex),-1)
    
    # generate 2-channel representation of image domain
    image_complex_2channel = K.tf.concat([real, imag], -1)
    return image_complex_2channel


def abs_layer(complex_data):
    """
    Input: 2-channel array representing complex data
    Output: 1-channel array representing magnitude of complex data
    """
    #get real and imaginary portions
    real = Lambda(lambda complex_data : complex_data[:,:,:,0])(complex_data)
    imag = Lambda(lambda complex_data : complex_data[:,:,:,1])(complex_data)
    
    mag = K.tf.abs(K.tf.complex(real,imag))
    mag = K.tf.expand_dims(mag, -1)
    return mag


def concat_empty_channel(singl_chnl_data):
    """
    Input: 2-channel array representing complex data
    Output: 1-channel array representing magnitude of complex data
    """
    #get real and imaginary portions
    
    return K.tf.concat([singl_chnl_data, K.tf.zeros(K.tf.shape(singl_chnl_data))], -1)


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
    Input: single channel array representing dct coefficient data
    Output: single channel array representing image domain data
    """
    
    patches = extract_patches(dctdata)
    image_patches = tf_idct2d(patches)
    image = extract_patches_inverse(dctdata, image_patches)
    
    return image


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
    
#     clip_low = Subtract()([Multiply()([data_list[1], data_list[2]]), data_list[2] * 0.5])
#     clip_high = Add()([Multiply()([data_list[1], data_list[2]]), data_list[2] * 0.5])
#     return K.tf.clip_by_value(data_list[0], clip_low, clip_high)

    clip_low = Subtract()([data_list[1], data_list[2] * 0.5])
    clip_high = Add()([data_list[1], data_list[2] * 0.5])
    return K.tf.clip_by_value(data_list[0], clip_low, clip_high)

    #return K.tf.clip_by_value(dctmat, clip_low, clip_high)


def cnn_block(cnn_input, depth, nf, kshape):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.1),padding='same')(layers[-1]))#LeakyReLU(alpha=0.1)
    final_conv = Conv2D(2, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1


def dense_block(dense_input):
    
    inputs_flat = Flatten()(dense_input)
    
#     dense1 = Dense(128, input_shape=(64,), activation='relu')(inputs_flat)
#     dense2 = Dense(128, activation='relu')(dense1)
#     dense3 = Dense(64, activation='relu')(dense2)
#     dense4 = Dense(64, activation='relu')(dense3)

#     dense1 = Dense(128, input_shape=(64,), activation='tanh')(inputs_flat)
#     dense2 = Dense(128, activation='tanh')(dense1)
#     dense3 = Dense(64, activation='tanh')(dense2)
#     dense4 = Dense(64, activation='tanh')(dense3)
    
    dense1 = Dense(256, input_shape=(64,), activation='tanh')(inputs_flat)
    dense2 = Dense(256, activation='tanh')(dense1)
    dense3 = Dense(128, activation='tanh')(dense2)
    dense4 = Dense(128, activation='tanh')(dense3)
    dense5 = Dense(64, activation='tanh')(dense4)
    dense6 = Dense(64, activation='tanh')(dense5)
    dense7 = Dense(64, activation='tanh')(dense6)
    last_dense = Dense(64, activation='tanh')(dense7)

    #output_layer = Reshape(8,8)(dense4)
    output_layer = Reshape((8,8,1), input_shape=(64,))(last_dense)
    
    return output_layer


def unet_block(unet_input, kshape=(3, 3)):
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

    conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out

def unet_block_2chnl(unet_input, kshape=(3, 3)):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
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

    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out


def DC_block(rec,mask,sampled_kspace,kspace = False):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    if kspace:
        rec_kspace = rec
    else:
        rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc =  Multiply()([rec_kspace,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    return rec_kspace_dc


def deep_cascade_unet_no_dc(depth_str='di', H=256, W=256, kshape=(3, 3)):
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
    layers = [inputs]
    dct_flag = False # flag whether input is in the image domain (false) or dct domain (true)

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'd' and not dct_flag:
            # Add a DCT U-net; Input should be in image domain
            layers.append(Lambda(dct_layer)(layers[-1])) 
            print("Append DCT layer")
            dct_flag = True
        elif ii != 'd' and ii != 'i':
            print("Layer not recognized. Only 'd' and 'i' layers are currently supported.")
            break;

        # Add CNN block
        layers.append(unet_block(layers[-1], kshape))
        print("Append U-net block")

        if dct_flag:
            if (jj + 1) % len(depth_str) > 0: # check if there is a next element
                if depth_str[jj + 1] == 'd':
                    print("DCT layer again. Don't append iDCT layer.")
                else:
                    layers.append(Lambda(idct_layer)(layers[-1])) 
                    print("Append iDCT layer")
                    dct_flag = False
            else:
                layers.append(Lambda(idct_layer)(layers[-1])) 
                print("Append iDCT layer")
                dct_flag = False
            # normalize idct image values at this step

    model = Model(inputs=inputs, outputs=layers[-1])
    return model


def deep_cascade_unet(depth_str='di', H=256, W=256, kshape=(3, 3)):
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
    qmat = Input(shape=(H, W, channels))
    layers = [inputs]
    dct_flag = False # flag whether input is in the image domain (false) or dct domain (true)

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'd' and not dct_flag:
            # Add a DCT U-net; Input should be in image domain
            layers.append(Lambda(dct_layer)(layers[-1])) 
            print("Append DCT layer")
            layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat]))
            dct_flag = True
        elif ii != 'd' and ii != 'i':
            print("Layer not recognized. Only 'd' and 'i' layers are currently supported.")
            break;

        # Add CNN block
        layers.append(unet_block(layers[-1], kshape))
        print("Append U-net block")

        if dct_flag:
            if (jj + 1) % len(depth_str) > 0: # check if there is a next element
                if depth_str[jj + 1] == 'd':
                    print("DCT layer again. Don't append iDCT layer.")
                else:
                    layers.append(Lambda(idct_layer)(layers[-1])) 
                    print("Append iDCT layer")
                    dct_flag = False
            else:
                layers.append(Lambda(idct_layer)(layers[-1])) 
                print("Append iDCT layer")
                dct_flag = False
            # normalize idct image values at this step

    model = Model(inputs=[inputs,qmat], outputs=layers[-1])
    return model


def deep_cascade_unet_ksp_no_dc(depth_str='ki', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image height
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model ######
    """

    channels = 2  # inputs are represented as two-channel images (complex)
    inputs = Input(shape=(H, W, channels))
    layers = [inputs]
    kspace_flag = False # flag whether input is in the image domain (false) or frequency domain (true)

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'k' and not kspace_flag:
            # Add a frequency U-net; Input should be in image domain
            layers.append(Lambda(fft_layer)(layers[-1])) 
            print("Append FFT layer")
            kspace_flag = True
        elif ii != 'k' and ii != 'i':
            print("Layer not recognized. Only 'k' and 'i' layers are currently supported.")
            break;

        # Add CNN block
        layers.append(unet_block_2chnl(layers[-1], kshape))
        print("Append U-net block")

        if kspace_flag:
            if (jj + 1) % len(depth_str) > 0: # check if there is a next element
                if depth_str[jj + 1] == 'k':
                    print("FFT layer again. Don't append iFFT layer.")
                else:
                    layers.append(Lambda(ifft_layer)(layers[-1]))
                    layers.append(Lambda(abs_layer)(layers[-1]))
                    print("Append iFFT layer")
                    print("Append magnitude layer")
                    layers.append(Lambda(concat_empty_channel)(layers[-1]))
                    kspace_flag = False
            else:
                layers.append(Lambda(ifft_layer)(layers[-1]))
                layers.append(Lambda(abs_layer)(layers[-1]))
                print("Append iFFT layer")
                print("Append magnitude layer")
                layers.append(Lambda(concat_empty_channel)(layers[-1]))
                kspace_flag = False

    model = Model(inputs=inputs, outputs=layers[-1])
    return model


def dequantization_network():
    
    inputs = Input(shape=(8,8,1))
    inputs_flat = Flatten()(inputs)
    
    dense1 = Dense(128, activation='relu')(inputs_flat)
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(64, activation='relu')(dense2)
    dense4 = Dense(64, activation='relu')(dense3)

    #output_layer = Reshape(8,8)(dense4)
    output_layer = Reshape((8,8,1), input_shape=(64,))(dense4)
    
    model = Model(inputs=inputs, outputs=output_layer)
    return model


# NEED TO FIX:
# Don't append iDCT layer for cases like 'ff'
def deep_cascade_fc_unet(depth_str='fi', H=256, W=256, kshape=(3, 3)):
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
    layers = [inputs]
    fc_flag = False # flag whether input is in the image domain (false) or dct domain (true)
                    # i.e. input is in image domain
    
    for (jj,ii) in enumerate(depth_str):
        print(jj,ii)
        if ii == 'f':
            if not fc_flag: # image domain
                layers.append(Lambda(fc_dct_layer)(layers[-1]))
                print("Append DCT layer (patches)")
                layers.append(dense_block(layers[-1]))
                print("Append FC/dense block")
                fc_flag = True
            else: # DCT domain
                layers.append(dense_block(layers[-1]))
                print("Append FC/dense block")
        elif ii == 'i':
            if not fc_flag: # image domain
                layers.append(unet_block(layers[-1], kshape))
                print("Append U-net block")
                fc_flag = False
            else: # DCT domain
                layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
                print("Append iDCT layer (patches)")
                layers.append(unet_block(layers[-1], kshape))
                print("Append U-net block")
                fc_flag = False
        else:
            print("Layer not recognized. Only 'f' and 'i' layers are currently supported.")
            break;
            
        if not ((jj + 1) % len(depth_str) > 0) and ii == 'f' and fc_flag:
            layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
            print("Append iDCT layer (patches)")
            

#     for (jj,ii) in enumerate(depth_str):
#         print(jj, ii)
#         if ii == 'f' and not fc_flag:
#             # Add a FC layer; Input should be in DCT domain
#             layers.append(Lambda(fc_dct_layer)(layers[-1]))
#             layers.append(dense_block(layers[-1])) 
#             # layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat])) # data consistency layer
#             # layers.append(Lambda(fc_idct_layer)([layers[-1], inputs]))
#             print("Append DCT layer (patches)")
#             print("Append FC/dense block")
#             # print("Append data-consistency layer")
#             # print("Append iDCT layer (patches)")
#             fc_flag = True
#         elif ii == 'i':
#             # Add CNN block
#             if fc_flag:
#                 layers.append(Lambda(fc_idct_layer)([layers[-1], inputs]))
#             layers.append(unet_block(layers[-1], kshape))
#             print("Append iDCT layer (patches)")
#             print("Append U-net block")
#             fc_flag = False
#         elif ii != 'f' and ii != 'i':
#             print("Layer not recognized. Only 'f' and 'i' layers are currently supported.")
#             break;

#         if fc_flag:
#             if (jj + 1) % len(depth_str) > 0: # check if there is a next element
#                 if depth_str[jj + 1] == 'f':
#                     print("FC block again. Don't append iDCT layer.")
# #                 else:
# #                     layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
# #                     print("Append iDCT layer (patches)")
# #                     fc_flag = False
#             else:
#                 layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
#                 print("Append iDCT layer (patches)")
#                 fc_flag = False
#             # normalize idct image values at this step

    model = Model(inputs=inputs, outputs=layers[-1])
    return model
    
##############################################################################################################
# 2019-06-10
##############################################################################################################
    
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Add, Subtract, LeakyReLU, \
                         MaxPooling2D, concatenate, UpSampling2D,\
                         Multiply, Dense, Flatten, Reshape


def tf_dct2d(im):
    return K.tf.transpose(K.tf.spectral.dct(K.tf.transpose(K.tf.spectral.dct(im,type=2,axis=-1,norm='ortho'),perm=[0,2,1,3]),type=2,axis=-1,norm='ortho'),perm=[0,2,1,3])
    
    
def tf_idct2d(mat): 
    return K.tf.transpose(K.tf.spectral.idct(K.tf.transpose(K.tf.spectral.idct(mat,type=2,axis=-1,norm='ortho'),perm=[0,2,1,3]),type=2,axis=-1,norm='ortho'),perm=[0,2,1,3])


# Modified from:
# https://www.programcreek.com/python/example/90422/tensorflow.extract_image_patches
def extract_patches(x):
    patches = K.tf.extract_image_patches(
        x,
        (1, 8, 8, 1),
        (1, 8, 8, 1),
        (1, 1, 1, 1),
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


def fft_layer(image):
    """
    Input: 2-channel array representing image domain complex data
    Output: 2-channel array representing k-space complex data
    """

    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = K.tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = K.tf.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(kspace_complex), -1)
    imag = K.tf.expand_dims(K.tf.imag(kspace_complex), -1)

    # generate 2-channel representation of k-space
    kspace = K.tf.concat([real, imag], -1)
    return kspace


def ifft_layer(kspace_2channel):
    """
    Input: 2-channel array representing k-space
    Output: 2-channel array representing image domain
    """
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)
    
    kspace_complex = K.tf.complex(real,imag) # Make complex-valued tensor
    image_complex = K.tf.ifft2d(kspace_complex)
    
    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(image_complex),-1)
    imag = K.tf.expand_dims(K.tf.imag(image_complex),-1)
    
    # generate 2-channel representation of image domain
    image_complex_2channel = K.tf.concat([real, imag], -1)
    return image_complex_2channel


def rfft_layer(image):
    """
    Input: single-channel array representing image domain real data
    Output: 2-channel array representing k-space complex data
    """

    image_patches = extract_patches(image)
    kspace_complex = K.tf.rfft2d(image_patches)

    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(kspace_complex), -1)
    imag = K.tf.expand_dims(K.tf.imag(kspace_complex), -1)

    # generate 2-channel representation of k-space
    kspace = K.tf.concat([real, imag], -1)
    return kspace


def irfft_layer(data_list):
    """
    Input: 2-channel array representing k-space
    Output: single-channel array representing real image domain
    
    data_list: data_list[0] = dft matrix (kspace_2channel) (in patches) 
                [total_num_patches, patch_size, patch_size, chnl]
               data_list[1] = single channel image in original dimensions [batch, H, W, chnl]
    """
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)
    
    kspace_complex = K.tf.complex(real,imag) # Make complex-valued tensor
    image_patches = K.tf.irfft2d(data_list[0])
    image = extract_patches_inverse(data_list[1], image_patches)
    
    return image


def abs_layer(complex_data):
    """
    Input: 2-channel array representing complex data
    Output: 1-channel array representing magnitude of complex data
    """
    #get real and imaginary portions
    real = Lambda(lambda complex_data : complex_data[:,:,:,0])(complex_data)
    imag = Lambda(lambda complex_data : complex_data[:,:,:,1])(complex_data)
    
    mag = K.tf.abs(K.tf.complex(real,imag))
    mag = K.tf.expand_dims(mag, -1)
    return mag


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

def dense_block(dense_input):
    
    inputs_flat = Flatten()(dense_input)
    
    dense1 = Dense(64, input_shape=(64,), activation='tanh')(inputs_flat)
    dense2 = Dense(64, activation='tanh')(dense1)
    dense3 = Dense(128, activation='tanh')(dense2)
    dense4 = Dense(128, activation='tanh')(dense3)
    dense5 = Dense(256, activation='tanh')(dense4)
    dense6 = Dense(256, activation='tanh')(dense5)
    dense7 = Dense(128, activation='tanh')(dense6)
    dense8 = Dense(128, activation='tanh')(dense7)
    dense9 = Dense(64, activation='tanh')(dense8)
    last_dense = Dense(64, activation='tanh')(dense9)

    #output_layer = Reshape(8,8)(dense4)
    output_layer = Reshape((8,8,1), input_shape=(64,))(last_dense)
    
    return output_layer


def unet_block(unet_input, kshape=(3, 3)):
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

    conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out

# NEED TO FIX:
# Don't append iDCT layer for cases like 'ff'
def deep_cascade_fc_unet(depth_str='fi', H=256, W=256, kshape=(3, 3)):
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
    qmat = Input(shape=(H, W, channels))
    qmat = Lambda(extract_patches)(qmat)
    inputs_pat = Lambda(extract_patches)(inputs)
    layers = [inputs]
    fc_flag = False # flag whether input is in the image domain (false) or dct domain (true)
                    # i.e. input is in image domain
    
    for (jj,ii) in enumerate(depth_str):
        print(jj,ii)
        if ii == 'f':
            if not fc_flag: # image domain
                layers.append(Lambda(fc_dct_layer)(layers[-1]))
                print("Append DCT layer (patches)")
                layers.append(dense_block(layers[-1]))
                print("Append FC/dense block")
                layers.append(Lambda(dct_dc_layer)([layers[-1], inputs_pat, qmat]))
                print("Append data consistency block")
                fc_flag = True
            else: # DCT domain
                layers.append(dense_block(layers[-1]))
                print("Append FC/dense block")
        elif ii == 'i':
            if not fc_flag: # image domain
                layers.append(unet_block(layers[-1], kshape))
                print("Append U-net block")
                fc_flag = False
            else: # DCT domain
                layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
                print("Append iDCT layer (patches)")
                layers.append(unet_block(layers[-1], kshape))
                print("Append U-net block")
                fc_flag = False
        else:
            print("Layer not recognized. Only 'f' and 'i' layers are currently supported.")
            break;
            
        if not ((jj + 1) % len(depth_str) > 0) and ii == 'f' and fc_flag:
            layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
            print("Append iDCT layer (patches)")

    model = Model(inputs=[inputs,qmat], outputs=layers[-1])
    return model


def deep_cascade_fc_unet_no_dc(depth_str='fi', H=256, W=256, kshape=(3, 3)):
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
    layers = [inputs]
    fc_flag = False # flag whether input is in the image domain (false) or dct domain (true)
                    # i.e. input is in image domain
    
    for (jj,ii) in enumerate(depth_str):
        print(jj,ii)
        if ii == 'f':
            if not fc_flag: # image domain
                layers.append(Lambda(fc_dct_layer)(layers[-1]))
                print("Append DCT layer (patches)")
                layers.append(dense_block(layers[-1]))
                print("Append FC/dense block")
                fc_flag = True
            else: # DCT domain
                layers.append(dense_block(layers[-1]))
                print("Append FC/dense block")
        elif ii == 'i':
            if not fc_flag: # image domain
                layers.append(unet_block(layers[-1], kshape))
                print("Append U-net block")
                fc_flag = False
            else: # DCT domain
                layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
                print("Append iDCT layer (patches)")
                layers.append(unet_block(layers[-1], kshape))
                print("Append U-net block")
                fc_flag = False
        else:
            print("Layer not recognized. Only 'f' and 'i' layers are currently supported.")
            break;
            
        if not ((jj + 1) % len(depth_str) > 0) and ii == 'f' and fc_flag:
            layers.append(Lambda(fc_idct_layer)([layers[-1], inputs])) 
            print("Append iDCT layer (patches)")

    model = Model(inputs=inputs, outputs=layers[-1])
    return model


def deep_cascade_unet_ksp_no_dc(depth_str='ki', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image height
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model ######
    """

    channels = 2  # inputs are represented as two-channel images (complex)
    inputs = Input(shape=(H, W, channels))
    layers = [inputs]
    kspace_flag = False # flag whether input is in the image domain (false) or frequency domain (true)

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'k' and not kspace_flag:
            # Add a frequency U-net; Input should be in image domain
            layers.append(Lambda(fft_layer)(layers[-1])) 
            print("Append FFT layer")
            kspace_flag = True
        elif ii != 'k' and ii != 'i':
            print("Layer not recognized. Only 'k' and 'i' layers are currently supported.")
            break;

        # Add CNN block
        layers.append(unet_block_2chnl(layers[-1], kshape))
        print("Append U-net block")

        if kspace_flag:
            if (jj + 1) % len(depth_str) > 0: # check if there is a next element
                if depth_str[jj + 1] == 'k':
                    print("FFT layer again. Don't append iFFT layer.")
                else:
                    layers.append(Lambda(ifft_layer)(layers[-1]))
                    layers.append(Lambda(abs_layer)(layers[-1]))
                    print("Append iFFT layer")
                    print("Append magnitude layer")
                    layers.append(Lambda(concat_empty_channel)(layers[-1]))
                    kspace_flag = False
            else:
                layers.append(Lambda(ifft_layer)(layers[-1]))
                layers.append(Lambda(abs_layer)(layers[-1]))
                print("Append iFFT layer")
                print("Append magnitude layer")
                layers.append(Lambda(concat_empty_channel)(layers[-1]))
                kspace_flag = False

    model = Model(inputs=inputs, outputs=layers[-1])
    return model


def deep_cascade_unet_rksp_no_dc(depth_str='ki', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image height
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model ######
    """

    channels = 1  # Real input image
    inputs = Input(shape=(H, W, channels))
    qmat = Input(shape=(H, W, channels))
    layers = [inputs]
    kspace_flag = False # flag whether input is in the image domain (false) or frequency domain (true)

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'k' and not kspace_flag:
            # Add a frequency U-net; Input should be in image domain
            layers.append(Lambda(rfft_layer)(layers[-1]))
            print("Append rFFT layer")
            kspace_flag = True
        elif ii != 'k' and ii != 'i':
            print("Layer not recognized. Only 'k' and 'i' layers are currently supported.")
            break;

        # Add CNN block
        layers.append(unet_block(layers[-1], kshape))
        print("Append U-net (single-channel) block")

        if kspace_flag:
            if (jj + 1) % len(depth_str) > 0: # check if there is a next element
                if depth_str[jj + 1] == 'k':
                    print("FFT layer again. Don't append iFFT layer.")
                else:
                    layers.append(Lambda(irfft_layer)(layers[-1], inputs))
                    #layers.append(Lambda(abs_layer)(layers[-1]))
                    print("Append irFFT layer")
                    #print("Append magnitude layer")
                    #layers.append(Lambda(concat_empty_channel)(layers[-1]))
                    kspace_flag = False
            else:
                layers.append(Lambda(irfft_layer)(layers[-1], inputs))
                #layers.append(Lambda(abs_layer)(layers[-1]))
                print("Append irFFT layer")
                #print("Append magnitude layer")
                #layers.append(Lambda(concat_empty_channel)(layers[-1]))
                kspace_flag = False

    model = Model(inputs=inputs, outputs=layers[-1])
    return model


def deep_cascade_unet_rksp(depth_str='ki', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image height
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model ######
    """

    channels = 1  # Real input image
    inputs = Input(shape=(H, W, channels))
    qmat = Input(shape=(H, W, channels))
    layers = [inputs]
    kspace_flag = False # flag whether input is in the image domain (false) or frequency domain (true)

    for (jj,ii) in enumerate(depth_str):
        print(jj, ii)
        if ii == 'k' and not kspace_flag:
            # Add a frequency U-net; Input should be in image domain
            layers.append(Lambda(rfft_layer)(layers[-1]))
            print("Append rFFT layer")
            kspace_flag = True
        elif ii != 'k' and ii != 'i':
            print("Layer not recognized. Only 'k' and 'i' layers are currently supported.")
            break;

        # Add CNN block
        layers.append(unet_block(layers[-1], kshape))
        print("Append U-net (single-channel) block")

        if kspace_flag:
            if (jj + 1) % len(depth_str) > 0: # check if there is a next element
                if depth_str[jj + 1] == 'k':
#                     print("FFT layer again. Don't append iFFT layer.")
                    layers.append(Lambda(irfft_layer)(layers[-1], inputs))
                    #layers.append(Lambda(abs_layer)(layers[-1]))
                    print("Append irFFT layer")
                    print("Append data consistency layer (DCT-DC-iDCT)")
                    layers.append(Lambda(dct_layer)(layers[-1])) 
                    layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat]))
                    layers.append(Lambda(idct_layer)(layers[-1]))
                else:
                    layers.append(Lambda(irfft_layer)(layers[-1], inputs))
                    #layers.append(Lambda(abs_layer)(layers[-1]))
                    print("Append irFFT layer")
                    print("Append data consistency layer (DCT-DC-iDCT)")
                    layers.append(Lambda(dct_layer)(layers[-1])) 
                    layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat]))
                    layers.append(Lambda(idct_layer)(layers[-1]))
                    #print("Append magnitude layer")
                    #layers.append(Lambda(concat_empty_channel)(layers[-1]))
                    kspace_flag = False
            else:
                layers.append(Lambda(irfft_layer)(layers[-1], inputs))
                #layers.append(Lambda(abs_layer)(layers[-1]))
                print("Append irFFT layer")
                print("Append data consistency layer (DCT-DC-iDCT)")
                layers.append(Lambda(dct_layer)(layers[-1])) 
                layers.append(Lambda(dct_dc_layer)([layers[-1], inputs, qmat]))
                layers.append(Lambda(idct_layer)(layers[-1]))
                #print("Append magnitude layer")
                #layers.append(Lambda(concat_empty_channel)(layers[-1]))
                kspace_flag = False

    model = Model(inputs=[inputs,qmat], outputs=layers[-1])
    return model

##############################################################################################################
# 2019-06-10
##############################################################################################################


##############################################################################################################
# Used for MICCAI submission... Prior to 2019-04-26
##############################################################################################################
# from keras import backend as K
# from keras.models import Model, Sequential
# from keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU, \
#                          MaxPooling2D, concatenate, UpSampling2D,\
#                          Multiply


# def dct2d(im):
#     return K.tf.transpose(K.tf.spectral.dct(K.tf.transpose(K.tf.spectral.dct(im,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))
    
    
# def idct2d(mat):
#     return K.tf.transpose(K.tf.spectral.idct(K.tf.transpose(K.tf.spectral.idct(mat,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))


# # Modified from:
# # https://stackoverflow.com/questions/44047753/reconstructing-an-image-after-using-extract-image-patches
# def extract_patches(x):
#     return K.tf.extract_image_patches(
#         x,
#         (1, 8, 8, 1),
#         (1, 8, 8, 1),
#         (1, 1, 1, 1),
#         padding="SAME"
#     )


# def extract_patches_inverse(x, y):
#     _x = K.tf.zeros_like(x)
#     _y = extract_patches(_x)
#     grad = K.tf.gradients(_y, _x)[0]
#     # Divide by grad, to "average" together the overlapping patches
#     # otherwise they would simply sum up
#     return K.tf.gradients(_y, _x, grad_ys=y)[0] / grad


# def nrmse(y_true, y_pred):
#     """
#     Normalized Root Mean Squared Error (NRMSE) - Euclidean distance normalization
#     :param y_true: Reference
#     :param y_pred: Predicted
#     :return:
#     """

#     denom = K.max(y_true, axis=(1,2,3)) - K.min(y_true, axis=(1,2,3))
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
#     /denom


# def nrmse_min_max(y_true, y_pred):
#     """
#      Normalized Root Mean Squared Error (NRMSE) - min-max normalization
#      :param y_true: Reference
#      :param y_pred: Predicted
#      :return:
#      """

#     denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
#     /denom


# def dct_layer(image):
#     """
#     Input: single channel array representing image domain data
#     Output: single channel array representing dct coefficient data
#     """
    
#     patches = extract_patches(image)
#     dct_patches = dct2d(patches)
#     dct_image = extract_patches_inverse(image, dct_patches)
    
#     return dct_image


# def idct_layer(dctdata):
#     """
#     Input: single channel array representing dct coefficient data
#     Output: single channel array representing image domain data
#     """
    
#     patches = extract_patches(dctdata)
#     image_patches = idct2d(patches)
#     image = extract_patches_inverse(dctdata, image_patches)
    
#     return image


# def cnn_block(cnn_input, depth, nf, kshape):
#     """
#     :param cnn_input: Input layer to CNN block
#     :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
#     2 channels
#     :param nf: Number of filters of convolutional layers, except for the last
#     :param kshape: Shape of the convolutional kernel
#     :return: 2-channel, complex reconstruction
#     """
#     layers = [cnn_input]

#     for ii in range(depth):
#         # Add convolutional block
#         layers.append(Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.1),padding='same')(layers[-1]))#LeakyReLU(alpha=0.1)
#     final_conv = Conv2D(2, (1, 1), activation='linear')(layers[-1])
#     rec1 = Add()([final_conv,cnn_input])
#     return rec1


# def unet_block(unet_input, kshape=(3, 3)):
#     """
#     :param unet_input: Input layer
#     :param kshape: Kernel size
#     :return: single channel
#     """

#     conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
#     conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
#     conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
#     conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
#     conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
#     conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

#     up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
#     conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
#     conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
#     conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

#     up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
#     conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
#     conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
#     conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

#     up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
#     conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
#     conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
#     conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

#     conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
#     out = Add()([conv8, unet_input])
#     return out


# def DC_block(rec,mask,sampled_kspace,kspace = False):
#     """
#     :param rec: Reconstructed data, can be k-space or image domain
#     :param mask: undersampling mask
#     :param sampled_kspace:
#     :param kspace: Boolean, if true, the input is k-space, if false it is image domain
#     :return: k-space after data consistency
#     """

#     if kspace:
#         rec_kspace = rec
#     else:
#         rec_kspace = Lambda(fft_layer)(rec)
#     rec_kspace_dc =  Multiply()([rec_kspace,mask])
#     rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
#     return rec_kspace_dc


# def deep_cascade_unet_no_dc(depth_str='di', H=256, W=256, kshape=(3, 3)):
#     """
#     :param depth_str: string that determines the depth of the cascade and the domain of each
#     subnetwork
#     :param H: Image height
#     :param W: Image width
#     :param kshape: Kernel size
#     :param nf: number of filters in each convolutional layer
#     :return: Deep Cascade Flat Unrolled model
#     """

#     channels = 1  # inputs are represented as single channel images (grayscale)
#     inputs = Input(shape=(H, W, channels))
#     layers = [inputs]
#     dct_flag = False # flag whether input is in the image domain (false) or dct domain (true)

#     for (jj,ii) in enumerate(depth_str):
#         print(jj, ii)
#         if ii == 'd' and not dct_flag:
#             # Add a DCT U-net; Input should be in image domain
#             layers.append(Lambda(dct_layer)(layers[-1]))
#             print("Append DCT layer")
#             dct_flag = True
#         elif ii != 'd' and ii != 'i':
#             print("Layer not recognized. Only 'd' and 'i' layers are currently supported.")
#             break;

#         # Add CNN block
#         layers.append(unet_block(layers[-1], kshape))
#         print("Append U-net block")

#         if dct_flag:
#             if (jj + 1) % len(depth_str) > 0: # check if there is a next element
#                 if depth_str[jj + 1] == 'd':
#                     print("DCT layer again. Don't append iDCT layer.")
#                 else:
#                     layers.append(Lambda(idct_layer)(layers[-1]))
#                     print("Append iDCT layer")
#                     dct_flag = False
#             else:
#                 layers.append(Lambda(idct_layer)(layers[-1]))
#                 print("Append iDCT layer")
#                 dct_flag = False
#             # normalize idct image values at this step

#     model = Model(inputs=inputs, outputs=layers[-1])
#     return model

