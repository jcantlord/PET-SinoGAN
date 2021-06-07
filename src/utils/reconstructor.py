import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, rescale

from PIL import Image


for idx in range(1000):

    try:
        arr_sinogram = np.load('/Users/jcantero/Desktop/oncovision_generated/sino_{}.npy'.format(idx))
        arr_sinogram = (arr_sinogram + 1) * 128
        theta = np.linspace(0., 180., max(arr_sinogram.shape), endpoint=False)

        reconstruction_fbp = iradon(arr_sinogram, output_size=128, theta=theta, filter_name='ramp', circle=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                    sharex=True, sharey=True)
        ax1.set_title("Synthetic Sinogram")
        ax1.imshow(arr_sinogram, cmap=plt.cm.Greys_r)
        ax2.set_title("Reconstruction error\nFiltered back projection")
        ax2.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

        plt.savefig('/Users/jcantero/Desktop/oncovision_reconstructed/{}.png'.format(idx))
        plt.clf()
        plt.close()

    except:
        pass
