//
// Created by tmondal on 24/08/2018.
//

#ifndef CLION_PROJECT_Fuzzy_C_Means_S1_H
#define CLION_PROJECT_Fuzzy_C_Means_S1_H

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


class Fuzzy_C_Means_S1 {
public:
    static Fuzzy_C_Means_S1* getInstance();

    Fuzzy_C_Means_S1();
    virtual ~Fuzzy_C_Means_S1();
    void applyAlgoOnImage(Mat&, Mat&);
    void multiplyMat(std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, std::vector<std::vector<int> >&);
    void multiplyOpenCVMat(Mat& Mat1, Mat& Mat2, Mat& result);
    void multiplyOpenCVMat2(Mat&, Mat&, Mat&);
    void multiplyTwoFloatMat(Mat&, Mat&, Mat&);
    void compareTwoMatrixSimilarity(Mat&, Mat&, vector<int>&);
    void matlab_reshape(const Mat &, Mat&, int, int, int);
private:
    static Fuzzy_C_Means_S1* instance;

    void padMatrix(Mat&, int, Mat&);
    void printFloatMatrix(cv::Mat imgMat);
    void padMatrix_1(Mat&, Mat& );
    void ud_FCM_S1(Mat&, int, float,  Mat&, Mat&, vector<float>&);

    void udDist_FCM_S1(Mat&, Mat&, Mat&, float, Mat&);
    void udInit_FCM_S1(int, int, Mat&);
    void udStepFCM_S1(Mat&, Mat&, Mat&, int&, double, int, Mat&, float&);

    void obj_mat(Mat&, Mat&, Mat&, Mat, int, Mat&, Mat& );
    void neighborCalculation(Mat&, std::vector<std::vector<int> >&, int);
    void removeOneRow(std::vector<std::vector<int> >&, int);
    void removeManyRow(Mat&, Mat&, int, int);

    inline static void dm(int x, int y, Mat& d, float& out) {
        out = (d.at<float>(x - 1, y) + d.at<float>(x - 1, y - 1) + d.at<float>(x - 1, y + 1) + d.at<float>(x + 1, y) +
               d.at<float>(x + 1, y - 1) + d.at<float>(x + 1, y + 1) + d.at<float>(x, y - 1) + d.at<float>(x, y + 1)) / 8;
    }
};


#endif //CLION_PROJECT_Fuzzy_C_Means_S1_H
