# JPEG-reconstruction
Improve JPEG reconstruction using deep learning. Quantization in the discrete cosine transform (DCT) domain can be defined as a sparse-sampling problem. This repository contains the source code for the W-net (a cascade of U-nets operating in different domain representations) investigated for MR reconstruction (https://github.com/rmsouza01/Hybrid-CS-Model-MRI) on JPEG decompression of natural and brain MR images. 

## Dataset
45,000 (30,000:15,000 training:validation) natural images from the ImageNet competition were used to pre-train the networks and fine-tuned with 16,110 (10,740:5,370 training:validation) T1-weighted MR brain image slices from the axial, sagittal, and coronal planes. These slices were extracted from 179 brain MR volumes from the Calgary-Campinas dataset (https://sites.google.com/view/calgary-campinas-dataset/home). All images were cropped or zero-padded to 256 $\times$ 256 and saved as uncompressed 8-bit TIFF files.

## Code
The code was developed using Python 3.6 on Jupyter Notebook, NumPy, TensorFlow and Keras. JPEG compression was performed by the Python Imaging Library (Pillow). We appreciate any feedback on how to improve our repository.

## Proposed Networks
