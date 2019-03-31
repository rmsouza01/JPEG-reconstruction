from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU, \
                         MaxPooling2D, concatenate, UpSampling2D,\
                         Multiply


<<<<<<< HEAD
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

=======
>>>>>>> 3c13ec7b2e8bb53c58c53d09bd529551940423bc

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


<<<<<<< HEAD
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

=======
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
    
>>>>>>> 3c13ec7b2e8bb53c58c53d09bd529551940423bc

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
<<<<<<< HEAD
    :return: single channel
=======
    :return: 2-channel, complex reconstruction
>>>>>>> 3c13ec7b2e8bb53c58c53d09bd529551940423bc
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

<<<<<<< HEAD
    conv8 = Conv2D(1, (1, 1), activation='linear')(conv7)
=======
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
>>>>>>> 3c13ec7b2e8bb53c58c53d09bd529551940423bc
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


<<<<<<< HEAD
def deep_cascade_unet_no_dc(depth_str='di', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image height
=======

def deep_cascade_flat_unrolled(depth_str = 'ikikii', H=256,W=256,depth = 5,kshape = (3,3), nf = 48):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
>>>>>>> 3c13ec7b2e8bb53c58c53d09bd529551940423bc
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

<<<<<<< HEAD
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
=======
    channels = 2 # inputs are represented as 2-channel images
    inputs = Input(shape=(H,W,channels))
    mask = Input(shape=(H,W,channels))
    layers = [inputs]
    
    for ii in depth_str:
        kspace_flag = True
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(cnn_block(layers[-1],depth,nf,kshape))

        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,kspace=kspace_flag))

    out = Lambda(ifft_layer)(layers[-1])
    out2 = Lambda(abs_layer)(out)
    model = Model(inputs=[inputs,mask], outputs=[out,out2])
    return model


def deep_cascade_unet(depth_str='ki', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    channels = 2  # inputs are represented as 2-channel images
    inputs = Input(shape=(H, W, channels))
    mask = Input(shape=(H, W, channels))
    layers = [inputs]

    for ii in depth_str:
        kspace_flag = True
        if ii == 'i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(unet_block(layers[-1], kshape))

        # Add DC block
        layers.append(DC_block(layers[-1], mask, inputs, kspace=kspace_flag))

    out = Lambda(ifft_layer)(layers[-1])
    out2 = Lambda(abs_layer)(out)
    model = Model(inputs=[inputs,mask], outputs=[out, out2])
    return model

def deep_cascade_unet_no_dc(depth_str='ki', H=256, W=256, kshape=(3, 3)):
    """
    :param depth_str: string that determines the depth of the cascade and the domain of each
    subnetwork
    :param H: Image heigh
    :param W: Image width
    :param kshape: Kernel size
    :param nf: number of filters in each convolutional layer
    :return: Deep Cascade Flat Unrolled model
    """

    channels = 2  # inputs are represented as 2-channel images
    inputs = Input(shape=(H, W, channels))
    mask = Input(shape=(H, W, channels))
    layers = [inputs]
    kspace_flag = True
        
    for (jj,ii) in enumerate(depth_str):
        if ii == 'i' and kspace_flag:
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        elif ii == 'k' and not kspace_flag:
            layers.append(Lambda(fft_layer)(layers[-1]))
            kspace_flag = True
        # Add CNN block
        layers.append(unet_block(layers[-1], kshape))
    
    if  kspace_flag:
        layers.append(Lambda(ifft_layer)(layers[-1]))
            
    out = Lambda(abs_layer)(layers[-1])
    model = Model(inputs=inputs, outputs=out)
>>>>>>> 3c13ec7b2e8bb53c58c53d09bd529551940423bc
    return model

