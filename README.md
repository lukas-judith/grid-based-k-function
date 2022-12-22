# Grid-based K-function

This repository contains code for a grid-based approximation of Ripley's K-function [1] for quantifying the clustering of point patterns in images. It was created during an internship project at the [Heidelberg Collaboratory for Image Processing (HCI)](https://hci.iwr.uni-heidelberg.de), where I focused on analyzing the aggretation of different proteins in cell images. The method was developed in particular for computing the K-function directly on the image by taking the pixel intensities into acccount. This is especially useful in the case of blurred and densely overlapping points, where the extraction of the true point coordinates is difficult (this would normally be necessary in order to apply the standard K-function).  

The notebook test-K-function.ipynb contains a short demo for the grid-based K-function.

*TODO: Add theoretical background for the grid-based K-function. Add examples for preprocessing and K-function-computation on real images, using process_images.py. Also add demos for two-sample testing on functional data as implemented in the scripts two_sample_testing.py and functional_data.py*



\
[1] B. D. Ripley, “Modelling spatial patterns,” Journal of the Royal Statistical Society: Series B (Methodological), vol. 39, no. 2, pp. 172–192, 1977.
