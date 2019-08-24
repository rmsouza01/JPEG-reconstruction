# JPEG-reconstruction
Improve JPEG reconstruction using deep learning. Quantization in the discrete cosine transform (DCT) domain can be defined as a sparse-sampling problem. This repository contains the source code for the W-net [1] (a cascade of U-nets operating on different domain representations of images) investigated for MR reconstruction (https://github.com/rmsouza01/Hybrid-CS-Model-MRI) on JPEG decompression of natural and brain MR images. 

## Dataset
45,000 (30,000:15,000 training:validation) grayscale natural images from the ImageNet competition were used to pre-train the networks and fine-tuned with 16,110 (10,740:5,370 training:validation) T1-weighted MR brain image slices from the axial, sagittal, and coronal planes. These slices were extracted from 179 brain MR volumes from the Calgary-Campinas dataset (https://sites.google.com/view/calgary-campinas-dataset/home). All images were cropped or zero-padded to 256 x 256 and saved as uncompressed 8-bit TIFF files.

## Code
The code was developed using Python 3.6 on Jupyter Notebook, NumPy, TensorFlow and Keras. JPEG compression was performed by the Python Imaging Library (Pillow). We appreciate any feedback on how to improve our repository.

## Proposed Networks
The deep neural networks were applied in the discrete cosine transform (DCT) domain and the image domain on which the JPEG algorithm operates. We investigated a W-net [1] that attempts to 'de-quantize' the DCT coefficients lost during JPEG quantization by leveraging information from adjacent 8 x 8 DCT coefficient blocks and directly refine compression artifacts at the image pixel level. The W-net was compared to an adaptation of the  Automated transform by Manifold Approximation (AUTOMAP) [2], which attempted to de-quantize DCT coefficients in 8 x 8 blocks using a fully-connected layers and also refine compression artifacts in the image domain using a U-net block. The domain processing order (DCT-domain first then image-domain vs. image-domain first then DCT-domain) was also investigated for both the W-net and AUTOMAP adaptations. A data consistency (DC) method was implemented to constrain predicted DCT coefficient values to a range facilitated by the quantization element. The inputs to the networks were the JPEG-compressed image and the associated quantization matrix with which it was compressed, while the ground-truth reference was the uncompressed image. 

![Block diagram of investigated deep learning networks for JPEG decompression](./Figures/networks_diagram.png?raw=True)
DI = DCT-Image domain network (i.e. DCT-domain first); ID = Image-DCT domain network (i.e. image-domain first)

## Sample Decompressions

![Sample decompressions using the proposed deep learning networks on natural images](./Figures/natural_images.png?raw=True)
![Sample decompressions using the proposed deep learning networks on brain MR images](./Figures/brain_images.png?raw=True)

JPEG decompression performance was evaluated by structural similarity (SSIM), peak signal-to-noise ratio (PSNR), and normalized root mean squared error (NRMSE). The ground-truth reference was the uncompressed image when computing performance metrics.

## References
[1] R. Souza, R. M. Lebel, and R. Frayne, “A hybrid, dual domain, cascade of convolutional neural networks for magnetic resonance image reconstruction,” in International Conference on Medical Imaging with Deep Learning – Full Paper Track, London, United Kingdom, 08–10 Jul 2019. [Online]. Available: https://openreview.net/forum?id=HJeJx4XxlN

[2] B. Zhu, J. Z. Liu, S. F. Cauley, B. R. Rosen, and M. S. Rosen, “Image reconstruction by domain-transform manifold learning,” Nature, 2018.
