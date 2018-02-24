import pickle
import os
from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from collections import Counter

IMG = 'ilk-3b-1024.tif'
colors = [(100, 150, 0), (20, 90, 200), (0, 0, 0),
          (32, 132, 129), (36, 132, 32), (161, 217, 155), 
          (189, 189, 189), (0, 0, 0), (255, 255, 255)]
		  

def process(k, train_sample_frac=0.05, image_filepath=IMG):
    img = Image.open(image_filepath)
    # print(np.array(img.getdata()))
    image = np.array(img) / 255.0
    X = image.reshape(-1, 3)
    sample_idx = np.random.choice(X.shape[0], int(X.shape[0] * train_sample_frac))
    X_sample = X[sample_idx, :]
    print(X_sample.shape)
    gmm = GaussianMixture(n_components=k, random_state=0).fit(X_sample)
    y = gmm.predict(X)
    y.resize(image.shape[:-1])

    likelihood = gmm.predict_proba(X)
    # print Counter(likelihood)
    vals = []
    combined = []
    
    for index, value in enumerate(likelihood):
        if max(value) > 0.005:
            vals.append((0,0,0))
            combined.append(colors[list(value).index(max(value))])
        else:
            vals.append((255,255,255))
            combined.append((255,255,255))

    label_img = Image.new('RGB', img.size)
    label_img.putdata(vals)
    # label_img.show()    

    label_img_combined = Image.new('RGB', img.size)
    label_img_combined.putdata(combined)

    plt.imshow(np.array(label_img_combined))
    plt.show()

    plt.imshow(np.array(label_img))
    plt.show()


def main():
    process(5, 0.2)


if __name__ == '__main__':
    main()