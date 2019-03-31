import numpy as np

def resize_by_pad_or_crop(img, sz=[256,256], padVal=0):
    X,Y = img.shape()

    offx = abs(sz[0] - X)/2
    offy = abs(sz[1] - Y)/2

    print (X, Y)
    print (offx, offy)

    tmpx = np.zeros((sz[0], Y), dtype=np.uint8) + padVal
    if X < sz[0]:
        tmpx[floor(offx):-ceil(offx),:] = img[:,:];
    elif X > sz[0]:
        tmpx[:,:] = img[floor(offx):-ceil(offx),:];
    else:
        tmpx[:,:] = img[:,:]

    tmpy = np.zeros((sz[0], sz[1]), dtype=np.uint8) + padVal
    if Y < sz[1]:
        tmpy[:,floor(offy):-ceil(offy)] = tmpx[:,:];
    elif Y > sz[1]:
        tmpy[:,:] = tmpx[:,floor(offy):-ceil(offy)];
    else:
        tmpy[:,:] = tmpx[:,:]
        
    return tmpy
