This repository includes methods for picking cluster sizes during unsupervised learning. It follows the ideas of Francisco J. Pérez-Reche in their paper 
"The elbow statistic: Multiscale clustering statistical significance". An elbow statistic is created by calculating the negative second difference of cluster heterogeneity
with respect to the number of clusters and normalizing it by dividing by the first difference of heterogeneity. The model selects cluster sizes with statistically significant 
elbow statistics. The p-values are estimated by performing Monte Carlo simulations where the data (or its principal components) are assumed to be uniformly 
distributed. And then multiple hypothesis test methods for controlling either the Family Wise Error Rate or the False Discovery Rate are used to find thresholds for determining 
which cluster sizes show a significant drop in heterogeneity.

The file elbow.py contains the classes for creating clustering models with methods for calculating elbow statistics and their p-values. There is also a method for plotting
the heterogeneity, elbow statistics and p-values (once calculated) together in one diagram. A simple example of how to use these models can be seen in the Jupyter notebook 
titled toy_example. And more intensive tests of the models can be found in the Jupyter notebook titled results. I repeat some of the tests performed by Pérez-Reche on both 
real and synthetic data, but some of the real data sets they used were hard to find. So I only used ones that can be found in Sklearn.
