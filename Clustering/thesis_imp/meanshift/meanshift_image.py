# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:34:48 2018
mean shift for image
@author: shifuddin
"""
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image

image = Image.open('toy1.PNG')
image = np.array(image)


flat_image=np.reshape(image, [-1, 3])

bandwidth2 = estimate_bandwidth(flat_image,
                                quantile=.2, n_samples=500)

ms = MeanShift(bandwidth2, bin_seeding=True)
ms.fit(flat_image)
labels=ms.labels_

plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(image)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [851,1280]))
plt.axis('off')