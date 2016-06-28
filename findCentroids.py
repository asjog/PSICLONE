import os
import numpy as np
import nibabel as nib
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph

img_file_name = '/Users/amod/PycharmProjects/PSICLONE/data/KKI2009-01-MPRAGE.nii'
img = nib.load(img_file_name)
data = img.get_data()
dataf = data.flatten()
nz_idxs = np.where(dataf > 0)[0]
fg = dataf[dataf>0]
fg = fg.reshape(fg.__len__(),1)
bw = 150
#estimate_bandwidth(fg,quantile=0.2,n_samples=500)
ms = MeanShift(bandwidth=bw,bin_seeding=True)
spectral = cluster.SpectralClustering(n_clusters=3,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
spectral.fit(fg)
ms.fit(fg)
dataf[nz_idxs] = ms.labels_
labeldata = np.reshape(dataf,(256,256,95))
labelimg = nib.Nifti1Image(labeldata,img.affine)
labelimg.set_data_dtype(np.int16)
labelimg.header.set_xyzt_units(img.header.get_xyzt_units()[0], img.header.get_xyzt_units()[1])
labelimg.to_filename('/Users/amod/PycharmProjects/PSICLONE/data/KKI2009-01-MPRAGE_MeanShift.nii')
nib.save(labelimg, '/Users/amod/PycharmProjects/PSICLONE/data/KKI2009-01-MPRAGE_MeanShift.nii')