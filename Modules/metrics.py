import numpy as np
import skimage.measure as metrics

def calculate_metrics(test_unc, test_compr):
    metr = np.zeros((len(test_compr),3,2),dtype=np.float32)
    print(metr.shape)

    count = 0
    for ii in range(len(test_compr)): 

        # Check if the metric values are finite
        # JPEG-Compressed Metrics
        a = metrics.compare_ssim(test_unc[ii,:,:,0],test_compr[ii,:,:,0],data_range=(test_compr[ii,:,:,0].max()-test_compr[ii,:,:,0].min()))
        if ~np.isfinite(a):
            print("removing %i SSIM from list." %ii)
            count += 1
            continue

        b = metrics.compare_psnr(test_unc[ii,:,:,0],test_compr[ii,:,:,0],\
                                            data_range=(test_compr[ii,:,:,0].max()-test_compr[ii,:,:,0].min()))
        if ~np.isfinite(b):
            print("removing %i PSNR from list." %ii)
            count += 1
            continue

        c = metrics.compare_nrmse(test_unc[ii,:,:,0],test_compr[ii,:,:,0],'min-max') *100.0
        if ~np.isfinite(c):
            print("removing %i NRMSE from list." %ii)
            count += 1
            continue

#         # Network metrics
#         d = metrics.compare_ssim(test_unc[ii,:,:,0],pred[ii,:,:,0], data_range=(pred[ii,:,:,0].max()-pred[ii,:,:,0].min()))
#         if ~np.isfinite(d):
#             print("removing %i SSIM from list." %ii)
#             count += 1
#             continue

#         e = metrics.compare_psnr(test_unc[ii,:,:,0],pred[ii,:,:,0],\
#                                            data_range=(pred[ii,:,:,0].max()-pred[ii,:,:,0].min()))
#         if ~np.isfinite(e):
#             print("removing %i PSNR from list." %ii)
#             count += 1
#             continue

#         f = metrics.compare_nrmse(test_unc[ii,:,:,0],pred[ii,:,:,0],'min-max') *100.0
#         if ~np.isfinite(f):
#             print("removing %i NRMSE from list." %ii)
#             count += 1
#             continue

        metr[ii,0,0] = a
        metr[ii,1,0] = b
        metr[ii,2,0] = c
#         metr[ii,0,1] = d
#         metr[ii,1,1] = e    
#         metr[ii,2,1] = f

    # remove empty rows
    metr = np.delete(metr,range(len(metr)-count, len(metr)),axis=0)
    print(metr.shape)

    print("\nPerformance Metrics")
    print("SSIM: %.3f +/- %.3f" %(metr[:,0,0].mean(),metr[:,0,0].std()))
    print("PSNR: %.3f +/- %.3f" %(metr[:,1,0].mean(),metr[:,1,0].std()))
    print("NRMSE: %.3f +/- %.3f" %(metr[:,2,0].mean(),metr[:,2,0].std()))
#     print('\n')
#     print("Network Metrics")
#     print("SSIM: %.3f +/- %.3f" %(metr[:,0,1].mean(),metr[:,0,1].std()))
#     print("PSNR: %.3f +/- %.3f" %(metr[:,1,1].mean(),metr[:,1,1].std()))
#     print("NRMSE: %.3f +/- %.3f" %(metr[:,2,1].mean(),metr[:,2,1].std()))
    
    return metr