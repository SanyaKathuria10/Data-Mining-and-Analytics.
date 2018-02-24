# Data-Mining-and-Analytics.
It contains mini-projects as part of Spatial and Temporal Data mining Spring 2018 class.

#Implementaion of Clustering:

Question :
Extend Gaussian Mixture Model (GMM) based clustering by exploiting the fact that a GMM is also a density model. Use the data from the image. (ilk-3b-1024.tif).

Perform the clustering as in hw2, say with k=5.
Then, for every pixel calculate its likelihood under the GMM density model.
Tune and fix a threshold on the likelihood values such that points that have low likelihood now fall into a new class – anomalies.

Input:
ilk-3b-1024.tif
Solution:
ExtendingGMM.py
Output:
Threshold=0.0005.jpeg	
Threshold=0.5.jpeg

Question :
For given satellite image (ilk-3b-1024.tif; 3-dimensional RGB, 1024x1024), do the following in RGB space:
(a) K-Means Clustering
(b) Model-based Clustering (GMM) 
(c) Compare (a) and (b) on the following criteria: a. Finding optimal clusters b. Quality of results in terms of visual agreement with thematic classes (e.g., buildings, roads, waters, grass, trees, etc)

Input:
ilk-3b-1024.tif
Solution:
Clustering.py
Output:
Kmeans.jpeg
GMM.jpeg
Comparison of the 2 methods:

Comparing K-Means and Gaussian Mixture Model, we observed the following:
Finding optimal clusters: Both K-Means and GMM have drastic difference in the results when the number of clusters are changes. However as number of clusters increase K-Means starts to overfit the data leading to unclear distinction between objects. Overall, K-Means with k = 4 and GMM with n_clust = 5 seem to give satisfactory results.
Quality of results: GMM gives sharper distinction between different objects as compared to K-Means. 
We rate the quality of visualization on a scale of 5 from the above output:
	
Thematic Class
K-Means (k = 4)
GMM (n_clust = 5)
Thus we see that overall GMM gives better results than K-Means


