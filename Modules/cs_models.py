from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU, \
                         MaxPooling2D, concatenate, UpSampling2D,\
                         Multiply


def dct2d(im):
    return K.tf.transpose(K.tf.spectral.dct(K.tf.transpose(K.tf.spectral.dct(im,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))
    
    
def idct2d(mat):
    return K.tf.transpose(K.tf.spectral.idct(K.tf.transpose(K.tf.spectral.idct(mat,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))


# Modified from:
# https://stackoverflow.com/questions/44047753/reconstructing-an-image-after-using-extract-image-patches
def extract_patches(x):
    return K.tf.extract_image_patches(
        x,
        (1, 8, 8, 1),
        (1, 8, 8, 1),
        (1, 1, 1, 1),
        padding="SAME"
    )


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


def dct_layer(image):
    """
    Input: single channel array representing image domain data
    Output: single channel array representing dct coefficient data
    """
    
    patches = extract_patches(image)
    dct_patches = dct2d(patches)
    dct_image = extract_patches_inverse(image, dct_patches)
    
    return dct_image


def idct_layer(dctdata):
    """
    Input: single channel array representing dct coefficient data
    Output: single channel array representing image domain data
    """
    
    patches = extract_patches(dctdata)
    image_patches = idct2d(patches)
    image = extract_patches_inverse(dctdata, image_patches)
    
    return image


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

