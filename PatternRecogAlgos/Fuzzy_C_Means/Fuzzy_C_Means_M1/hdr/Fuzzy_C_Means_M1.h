//
// Created by tmondal on 24/08/2018.
//

#ifndef CLION_PROJECT_Fuzzy_C_Means_M1_H
#define CLION_PROJECT_Fuzzy_C_Means_M1_H

#include<iostream>
#include<opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include "util/hdr/BasicAlgo.h"

#include <math.h>
#include <cstring>
#include <random>
#include <chrono>

using namespace std;
using namespace cv;


class Fuzzy_C_Means_M1 {
public:
    static Fuzzy_C_Means_M1* getInstance();

    Fuzzy_C_Means_M1();
    virtual ~Fuzzy_C_Means_M1();
    void applyAlgoOnImage(Mat&, Mat&);
    void multiplyMat(std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, std::vector<std::vector<int> >&);
    void multiplyOpenCVMat2(Mat&, Mat&, Mat&);
    void multiplyTwoFloatMat(Mat&, Mat&, Mat&);
    void compareTwoMatrixSimilarity(Mat&, Mat&, vector<int>&);
    void matlab_reshape(const Mat &, Mat&, int, int, int);
private:
    static Fuzzy_C_Means_M1* instance;

    void padMatrix(Mat&, int, Mat&);
    void printFloatMatrix(cv::Mat imgMat);
    void padMatrix_1(Mat&, Mat& );
    void ud_FCM_M1(Mat&, int, Mat&, Mat&, vector<float>&);

    void udDist_FCM_M1(Mat&, Mat&, Mat&);
    void udInit_FCM_M1(int, int, Mat&);
    void udStepFCM_M1(Mat&, Mat&, int&, int, Mat&, float&);

    void neighborCalculation(Mat&, std::vector<std::vector<int> >&, int);

    inline static void dm(int x, int y, Mat& d, float& out) {
        out = (d.at<float>(x - 1, y) + d.at<float>(x - 1, y - 1) + d.at<float>(x - 1, y + 1) + d.at<float>(x + 1, y) +
               d.at<float>(x + 1, y - 1) + d.at<float>(x + 1, y + 1) + d.at<float>(x, y - 1) + d.at<float>(x, y + 1)) / 8;
    }
};


#endif //CLION_PROJECT_Fuzzy_C_Means_M1_H
