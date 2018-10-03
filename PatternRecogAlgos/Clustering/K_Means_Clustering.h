//
// Created by tmondal on 18/09/2018.
//

#ifndef CLION_PROJECT_K_MEANS_CLUSTERING_H
#define CLION_PROJECT_K_MEANS_CLUSTERING_H

#include<iostream>
#include<opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include "../../util/hdr/BasicAlgo.h"

#include <math.h>
#include <cstring>
#include <random>
#include <chrono>

using namespace std;
using namespace cv;


class K_Means_Clustering {
public:
    static K_Means_Clustering* getInstance();

    K_Means_Clustering();
    virtual ~K_Means_Clustering();
    void applyOpenCV_KMeans_1(Mat&, Mat&, Mat&,  int, int, Mat&);
    void applyOpenCV_KMeans_2(Mat&, Mat&, Mat&,  int, int, Mat&);
private:
    static K_Means_Clustering* instance;

};


#endif //CLION_PROJECT_K_MEANS_CLUSTERING_H
