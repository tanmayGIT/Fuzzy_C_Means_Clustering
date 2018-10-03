//
// Created by tmondal on 18/09/2018.
//

#include "K_Means_Clustering.h"


K_Means_Clustering *K_Means_Clustering::instance = 0;

K_Means_Clustering *K_Means_Clustering::getInstance() {
    if (!instance)
        instance = new K_Means_Clustering();

    return instance;
}

K_Means_Clustering::K_Means_Clustering() {
}

K_Means_Clustering::~K_Means_Clustering() {
}

void K_Means_Clustering::applyOpenCV_KMeans_1(Mat& foreGdImage, Mat& labels, Mat& centers,  int clusterCount, int attempts, Mat& newClusteredImage ) {
    Mat clusterData;
    foreGdImage.convertTo(clusterData, CV_32F);
    clusterData = clusterData.reshape(1, clusterData.total());

    // do K-Means
    kmeans(clusterData, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 1.0), attempts,
           KMEANS_PP_CENTERS, centers);

    newClusteredImage = Mat::zeros(foreGdImage.size(), CV_32FC1);
    for (int y = 0; y < foreGdImage.rows; y++)
        for (int x = 0; x < foreGdImage.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * foreGdImage.rows, 0);
            newClusteredImage.at<float>(y, x) = centers.at<float>(cluster_idx, 0);
        }
    //BasicAlgo::getInstance()->writeImageGivenPath(newClusteredImage, "/Users/tmondal/Documents/8_clusteredImage.jpg");
}

void K_Means_Clustering::applyOpenCV_KMeans_2(Mat& foreGdImage, Mat& labels, Mat& centers,  int clusterCount, int attempts, Mat& newClusteredImage) {
   // The idea of this part of the code is to convert gray image into color image then apply
   Mat colorImage;
   if (foreGdImage.channels() == 1) {
       cvtColor(foreGdImage,colorImage, CV_GRAY2BGR);
   }

    Mat p = Mat::zeros(colorImage.cols*colorImage.rows, 5, CV_32F);
   // Mat bestLabels, centers, clustered;

    vector<Mat> bgr;
    cv::split(colorImage, bgr);
    // i think there is a better way to split pixel bgr color
    for(int i=0; i < colorImage.cols*colorImage.rows; i++) {
        p.at<float>(i,0) = (i / colorImage.cols) / colorImage.rows;
        p.at<float>(i,1) = (i % colorImage.cols) / colorImage.cols;
        p.at<float>(i,2) = bgr[0].data[i] / 255.0;
        p.at<float>(i,3) = bgr[1].data[i] / 255.0;
        p.at<float>(i,4) = bgr[2].data[i] / 255.0;
    }

    int K = 8;
    cv::kmeans(p, clusterCount, labels,
               TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), attempts, KMEANS_PP_CENTERS, centers);

    int colors[K];
    for(int i=0; i<K; i++) {
        colors[i] = 255/(i+1);
    }
    // I think there is a better way to do this mayebe some Mat::reshape?
    newClusteredImage = Mat(colorImage.rows, colorImage.cols, CV_32F);
    for(int i=0; i<colorImage.cols*colorImage.rows; i++) {
        newClusteredImage.at<float>(i/colorImage.cols, i%colorImage.cols) = (float)(colors[labels.at<int>(0,i)]);
    }

    newClusteredImage.convertTo(newClusteredImage, CV_8U);
   // imshow("clustered", newClusteredImage);

}