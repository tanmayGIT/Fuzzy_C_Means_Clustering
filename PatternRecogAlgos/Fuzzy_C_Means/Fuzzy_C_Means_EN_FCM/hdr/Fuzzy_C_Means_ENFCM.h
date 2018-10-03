//
// Created by tmondal on 21/09/2018.
//

#ifndef CLION_PROJECT_FUZZY_C_MEANS_ENFCM_H
#define CLION_PROJECT_FUZZY_C_MEANS_ENFCM_H

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

class Fuzzy_C_Means_ENFCM {

public:
    static Fuzzy_C_Means_ENFCM* getInstance();

    Fuzzy_C_Means_ENFCM();
    virtual ~Fuzzy_C_Means_ENFCM();
    Mat neighborCalculation(std::vector<std::vector<int> >&, int);
    void multiply_2_UChar_Mat(Mat &, Mat &, Mat &);
    void applyAlgoOnImage(Mat&, Mat&);
    void printMatrixUChar(cv::Mat);
    void printMatrixFloat(cv::Mat);
private:
    static Fuzzy_C_Means_ENFCM* instance;
    void dist_enFCM(Mat &, Mat &, Mat &);
    void enFCM_Init(int, int, Mat&);
    void padMatrix_1(Mat&, Mat&);
    void padMatrix(Mat&, int, Mat&);
    void step_En_FCM(Mat&, Mat&, Mat&, int, int, Mat&, float&);
    void multiplyOpenCVMat2(Mat&, Mat&, Mat&);
    void multiplyTwoFloatMat(Mat&, Mat&, Mat&);
    float lnaiFun(int, Mat&, std::vector<std::vector<int> >&, float);
    void en_FSM(Mat &, int, float, Mat &, Mat &, vector<float>&);
    void calculateHistogram_2(const Mat&, const vector<float>&, Mat&, int, vector<unsigned int>&);
    void compareTwoMatrixSimilarity(Mat&, Mat&, vector<int>&);
    void matlab_reshape(const Mat &, Mat&, int, int, int);
};


#endif //CLION_PROJECT_FUZZY_C_MEANS_ENFCM_H
