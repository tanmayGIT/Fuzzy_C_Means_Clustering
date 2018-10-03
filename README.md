# Fuzzy C Means Clustering
This project folder contains the code of the various Fuzzy C means algorithm for image grascale image clustering. 

This project is developed in C++ with OpenCV-3. I have used CLon IDE as the development platform. 

As this project folder contains C-Make file, you can build this project from terminal and can run it in any platform (Mac, Ubntu (I have not checked in windows))

Please note that this project is implemented from the Matlab code obtained from this following link : 

                      https://github.com/marearth/fcm_m
                      
I have tried my best to achive the identical result of the implementation done in Matlab. As as example, one image is provided in the project folder. 

You just need to pass the path of the image in the main function and you should get the clusterd image as output. 

I am passing the image path as command line argument but if you prefer then you can directly provide the image path in the code.

The following Fuzzy-C-Means Clustering algos are implemented here :
1) FCM_S1
2) FCM_S2
3) FCM_M1
4) FLICM
5) FGFCM
6) FGFCM_S1
7) FGFCM_S2
8) EnFCM    





#
# Further Reading..
#

# Introduction 
Image segmentation is widely used in a variety of applications such as robot vision, object recognition, geographical imaging and medical imaging. Classically, image segmentation is defined as the partitioning of an image into non-overlapped, consistent regions which are homogeneous with respect to some characteristics such as gray value or texture.
Fuzzy c-mean (FCM) is one of the most used methods for image segmentation and its success chiefly attributes to the introduction of fuzziness for the belongingness of each image pixels. 
# Fuzzy C Means (FCM) Algorithm 
Compared with crisp or hard segmentation methods, FCM is able to retain more information from the original image. However, one disadvantage of standard FCM is not to consider any spatial information in image context, which makes it very sensitive to noise and other imaging artifacts. Recently, many researchers have incorporated local spatial information into the original FCM algorithm to improve the performance of image segmentation. 

## FCM_S Algorithm 
Ahmed et al. modified the objective function of FCM to compensate for the gray (intensity) inhomogeneity and to allow the labeling of a pixel to be influenced by the labels in its immediate neighborhood, and they call the algorithm as FCM_S.

### FCM_S1 and FCM_S2 Algorithm 
One disadvantage of FCM_S is that it computes the neighborhood term in each iteration step, which is very time-consuming. In order to reduce the computational loads of FCM_S, Chen and Zhang proposed two variants, FCM_S1 and FCM_S2, which simplified the neighborhood term of the objective function of FCM_S. These two algorithms introduce the extra mean-filtered image and median-filtered image respectively, which can be computed in advance, to replace the neighborhood term of FCM_S. Thus the execution times of both FCM_S1 and FCM_S2 are considerably reduced.

## Enhanced-FCM (EnFCM) Algorithm 
Szilagyi et al. proposed the enhanced FCM (EnFCM)algorithm to accelerate the image segmentation process. The structure of the EnFCM is different from that of FCM_S and its variants. First, a linearly-weighted sum image is formed from both original image and each pixel’s local neighborhood average gray level. Then clustering is performed on the basis of the gray level histogram instead of pixels of the summed image. Since, the number of gray levels in an image is generally much smaller than the number of its pixels, the computational time of EnFCM algorithm is reduced, while the quality of the segmented image is comparable to that of FCM_S.

## Fast Generalized FCM (FG-FCM) Algorithm 
More recently, Cai et al. proposed the fast generalized FCM algorithm (FGFCM), which incorporates the spatial information, the intensity of the local pixel neighborhood and the number of gray levels in an image. This algorithm forms a nonlinear weighted sum image from both original image and its local spatial and gray level neighborhood. The computational time of FGFCM is very small, since clustering is performed on the basis of the gray level histogram. The quality of the segmented image is well enhanced.

However, EnFCM as well as FGFCM, share a common
crucial parameter 
![symbol a](https://latex.codecogs.com/gif.latex?a) or (![symbol lamda](https://latex.codecogs.com/gif.latex?%5Clambda)) This parameter is used to control the tradeoff between the original image and its corresponding mean or median-filtered image. It has a crucial impact on the performance of those methods, but its selection is generally difficult because it should keep a balance between robustness to noise and effectiveness of preserving the details. 

### FG-FCM_S1 and FG-FCM_S2 Algorithm 
Motivated by FCM_S1 and FCM_S2 algorithms, in the similar manner FG-FCM_S1 and FG-FCM_S2 algorithms are proposed. 
These two algorithms introduce the mean and median of the neighbors within a specified window respectively, which can be computed in advance, to replace the neighborhood term of FG-FCM.


## Fuzzy Local Information C-Means Clustering (FLICM) Algorithm 
This algorithm can handle the defect of the selection of parameter ![symbol a](https://latex.codecogs.com/gif.latex?a) or (![symbol lamda](https://latex.codecogs.com/gif.latex?%5Clambda)), as well as promoting the image segmentation performance. In FLICM, a novel fuzzy factor is defined to replace the parameter a used in EnFCM and FCM_S and its variants, and the parameter  used in FGFCM and its variants. 
The new fuzzy local neighborhood factor can automatically determine the spatial and gray level relationship and is fully free of any parameter selection.


### References 

[1] W. Cai and D. Zhang, “Fast and Robust Fuzzy C-Means Clustering Algorithms Incorporating Local Information for Image Segmentation,” vol. 2, pp. 1–27.

[2] T. Lei, X. Jia, Y. Zhang, S. Member, L. He, and S. Member, “Significantly Fast and Robust Fuzzy C-Means Clustering Algorithm Based on Morphological Reconstruction and Membership Filtering,”, pp. 1–15, 2017.

[3] S. Krinidis and V. Chatzis, “A Robust Fuzzy Local Information C-means Clustering Algorithm,” no. May, 2010.

[4] H. L. Capitaine, C. Fre ́licot, M. Laboratoire, I. A, and U. D. L. Rochelle, “A fast fuzzy c-means algorithm for color image segmentation,” no. July, 2011.

[5] W. Cai, S. Chen, and D. Zhang, “Fast and robust fuzzy c -means cluster- ing algorithms incorporating local information for image segmentation,” vol. 40, pp. 825–838, 2007.
