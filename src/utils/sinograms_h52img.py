import h5py
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

data_file = '/Users/jcantero/Desktop/oncovision_mammi.h5'
file = h5py.File(data_file, 'r')

def get_sinogram(image_path):
    sinogram = file['{}'.format(image_path)]
    channel = np.array(sinogram[:,:])

    return channel

# Get all filenames in h5 file
data_filenames = [f for f in file.keys()]

for idx_fc, fc in enumerate(file.keys()):

    # Extract sinogram
    arr_sinogram = get_sinogram(fc)

    # Transform to range 0-255
    arr_sinogram *= 255.0/arr_sinogram.max()

    # Get image and transform to grayscale
    im = Image.fromarray(arr_sinogram)
    im = im.convert('L')

    # Save image
    im.save(f"/Users/jcantero/Desktop/oncovision_mammi/{idx_fc}.jpeg")
