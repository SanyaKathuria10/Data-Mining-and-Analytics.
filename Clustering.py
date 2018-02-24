import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import cv2

#image read and processing
image = cv2.imread("ilk-3b-1024.tif")
image = np.array(image, dtype=np.float64) / 255
old_shape = image.shape
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))
n =(int) (0.05 * len(image_array))
image_array_sample = shuffle(image_array, random_state=0)[:n]

#kmeans
kmeans = KMeans(n_clusters=4).fit(image_array_sample)
labels = kmeans.predict(image_array)

#plotting kmeans
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image')
labels = labels.reshape(old_shape[0], old_shape[1])
plt.imshow(label


#gmm
gmm = mixture.GaussianMixture(n_components = 4)
gmm = gmm.fit(image_array_sample)
labels = gmm.predict(image_array)

#plotting gmm
plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('New Image')
labels = labels.reshape(old_shape[0], old_shape[1])
plt.imshow(labels)

plt.show()
