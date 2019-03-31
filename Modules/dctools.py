import numpy as np
from tensorflow.spectral import dct as tf_dct, idct as tf_idct
from tensorflow import transpose as tpose
from scipy.fftpack import dct as py_dct, idct as py_idct

def py_dct2d(im):
    return py_dct(py_dct(im,type=2,axis=0,norm='ortho'),type=2,axis=1,norm='ortho')
def py_idct2d(mat):
    return py_idct(py_idct(mat,type=2,axis=0,norm='ortho'),type=2,axis=1,norm='ortho')
def tf_dct2d(tens):
    return transpose(tf_dct(transpose(tf_dct(mat,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))
def tf_idct2d(tens):
    return transpose(tf_idct(transpose(tf_idct(mat,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))

def dct_in_blocks(im, block_size = 8):

    rows, cols = im.shape[0], im.shape[1]

    # block size: 8x8
    if rows % block_size == cols % block_size == 0:
        blocks_count = rows // block_size * cols // block_size
    else:
        raise ValueError(("the width and height of the image "
                          "should both be mutiples of %block_size"))

    dct_matrix = np.zeros((rows, cols))

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):

            block = im[i:i+block_size, j:j+block_size]
            dct_matrix[i:i+block_size,j:j+block_size] = py_dct2d(block)

    return dct_matrix

def idct_in_blocks(dct_mat, block_size = 8):

    rows, cols = dct_mat.shape[0], dct_mat.shape[1]

    # block size: 8x8
    if rows % block_size == cols % block_size == 0:
        blocks_count = rows // block_size * cols // block_size
    else:
        raise ValueError(("the width and height of the image "
                          "should both be mutiples of %block_size"))

    im_matrix = np.zeros((rows, cols))

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):

            block = dct_mat[i:i+block_size, j:j+block_size]
            im_matrix[i:i+block_size,j:j+block_size] = py_idct2d(block)

    return im_matrix


# def dct2d(im, backend='py'):
#     if backend == 'py':
#         return py_dct(py_dct(im,type=2,axis=0,norm='ortho'),type=2,axis=1,norm='ortho')
#     else:
#         return tpose(tf_dct(tpose(py_dct(mat,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))
    
    
# def idct2d(mat, backend='py'):
#     if backend == 'py':
#         return py_idct(py_idct(mat,type=2,axis=0,norm='ortho'),type=2,axis=1,norm='ortho')
#     else:
#         return tpose(tf_idct(tpose(py_idct(mat,type=2,axis=-1,norm='ortho')),type=2,axis=-1,norm='ortho'))

    
# def dct_in_blocks(im, block_size = 8, backend='py'):

#     rows, cols = im.shape[0], im.shape[1]

#     # block size: 8x8
#     if rows % block_size == cols % block_size == 0:
#         blocks_count = rows // block_size * cols // block_size
#     else:
#         raise ValueError(("the width and height of the image "
#                           "should both be mutiples of %block_size"))

#     dct_matrix = np.zeros((rows, cols))

#     for i in range(0, rows, block_size):
#         for j in range(0, cols, block_size):

#             block = im[i:i+block_size, j:j+block_size]
#             dct_matrix[i:i+block_size,j:j+block_size] = dct2d(block,backend)

#     return dct_matrix

# def idct_in_blocks(dct_mat, block_size = 8,backend='py'):

#     rows, cols = dct_mat.shape[0], dct_mat.shape[1]

#     # block size: 8x8
#     if rows % block_size == cols % block_size == 0:
#         blocks_count = rows // block_size * cols // block_size
#     else:
#         raise ValueError(("the width and height of the image "
#                           "should both be mutiples of %block_size"))

#     im_matrix = np.zeros((rows, cols))

#     for i in range(0, rows, block_size):
#         for j in range(0, cols, block_size):

#             block = dct_mat[i:i+block_size, j:j+block_size]
#             im_matrix[i:i+block_size,j:j+block_size] = idct2d(block,backend)

#     return im_matrix
