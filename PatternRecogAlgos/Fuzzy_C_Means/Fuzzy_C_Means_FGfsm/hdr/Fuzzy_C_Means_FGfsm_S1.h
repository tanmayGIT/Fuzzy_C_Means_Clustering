//
// Created by tmondal on 20/09/2018.
//

#ifndef CLION_PROJECT_FUZZY_C_MEANS_FGFSM_S1_H
#define CLION_PROJECT_FUZZY_C_MEANS_FGFSM_S1_H


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


class Fuzzy_C_Means_FGfsm_S1 {

public:
    static Fuzzy_C_Means_FGfsm_S1* getInstance();

    Fuzzy_C_Means_FGfsm_S1();
    virtual ~Fuzzy_C_Means_FGfsm_S1();
    void multiplyTwo2DVector(std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<std::vector<int> >&);
    Mat neighborCalculation(std::vector<std::vector<int> >&, int);
    void multiply_2_UChar_Mat(Mat &, Mat &, Mat &);
    void multiply_Float_Uchar_Mat(Mat &, Mat &, Mat &);
    void applyAlgoOnImage(Mat&, Mat&);
private:

    static Fuzzy_C_Means_FGfsm_S1* instance;
    void udDist_FCM_M1(Mat &, Mat &, Mat &);
    void udInit_FCM_M1(int, int, Mat&);
    void padMatrix_1(Mat&, Mat&);
    void padMatrix(Mat&, int, Mat&);
    void step_FG_FCM(Mat&, Mat&, Mat&, int, int, Mat&, float&);
    void multiplyOpenCVMat2(Mat&, Mat&, Mat&);
    void multiplyTwoFloatMat(Mat&, Mat&, Mat&);
    void lnwiFun(int ii, Mat& , std::vector<std::vector<int> >&,Mat&);

    void fg_FSM(Mat &, int, Mat &, Mat &, vector<float>&);
    void calculateHistogram_2(const Mat&, const vector<float>&, Mat&, int, vector<unsigned int>&);
    void compareTwoMatrixSimilarity(Mat&, Mat&, vector<int>&);
    void matlab_reshape(const Mat &, Mat&, int, int, int);
};

#endif //CLION_PROJECT_Fuzzy_C_Means_FGfsm_S1_H
