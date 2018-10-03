//
// Created by tmondal on 21/09/2018.
//

#ifndef CLION_PROJECT_FUZZY_C_MEANS_FLICM_H
#define CLION_PROJECT_FUZZY_C_MEANS_FLICM_H

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
class KeepAllNeighbour{
public :
    Mat out1;
    Mat out2;
    Mat out3;
};

class Fuzzy_C_Means_FLIcm {

public:
    static Fuzzy_C_Means_FLIcm* getInstance();

    Fuzzy_C_Means_FLIcm();
    virtual ~Fuzzy_C_Means_FLIcm();
    void applyAlgoOnImage(Mat&, Mat&);
private:
    static Fuzzy_C_Means_FLIcm* instance;

    KeepAllNeighbour attachAllNeigh;
    void padMatrix(Mat &, int, Mat &);
    void padMatrix_1(Mat &, Mat &);
    void udStepFLICM(Mat &, Mat &, Mat &, int &, int, Mat&, float&);
    void udDist_FCM(Mat&, Mat&, Mat&, Mat&, int, Mat&, Mat&);
    void udInit_FCM(int, int, Mat&);
    float * neighborCalculation(const Mat, int);
    void fLICM(Mat &, int, Mat &, Mat &, vector<float>&);
    void multiplyTwoFloatMat(Mat&, Mat&, Mat&);
    void multiplyOpenCVMat2(Mat&, Mat&, Mat&);
    void compareTwoMatrixSimilarity(Mat&, Mat&, vector<int>&);
    void multiplyMat(std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, std::vector<std::vector<int> > &);
    void matlab_reshape(const Mat&, Mat&, int, int, int);
    void multiply_2_UChar_Mat(Mat &, Mat &, Mat &);
};


#endif //CLION_PROJECT_FUZZY_C_MEANS_FLICM_H
