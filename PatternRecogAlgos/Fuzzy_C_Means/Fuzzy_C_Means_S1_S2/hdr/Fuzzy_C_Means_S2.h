//
// Created by tmondal on 24/08/2018.
//

#ifndef CLION_PROJECT_Fuzzy_C_Means_S2_H
#define CLION_PROJECT_Fuzzy_C_Means_S2_H

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


class Fuzzy_C_Means_S2 {
public:
    static Fuzzy_C_Means_S2* getInstance();

    Fuzzy_C_Means_S2();
    virtual ~Fuzzy_C_Means_S2();
    void applyAlgoOnImage(Mat&, Mat&);
    void multiplyMat(std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, std::vector<std::vector<int> >&);
    void multiplyOpenCVMat2(Mat&, Mat&, Mat&);
    void multiplyTwoFloatMat(Mat&, Mat&, Mat&);
    void compareTwoMatrixSimilarity(Mat&, Mat&, vector<int>&);
    void matlab_reshape(const Mat &, Mat&, int, int, int);
private:
    static Fuzzy_C_Means_S2* instance;

    void padMatrix(Mat&, int, Mat&);
    void printFloatMatrix(cv::Mat imgMat);
    void padMatrix_1(Mat&, Mat& );
    void ud_FCM_S2(Mat&, int, float,  Mat&, Mat&, vector<float>&);

    void udDist_FCM_S2(Mat&, Mat&, Mat&, float, Mat&);
    void udInit_FCM_S2(int, int, Mat&);
    void udStepFCM_S2(Mat&, Mat&, Mat&, int&, double, int, Mat&, float&);

    void obj_mat(Mat&, Mat&, Mat&, Mat, int, Mat&, Mat& );
    void neighborCalculation(Mat&, std::vector<std::vector<int> >&, int);

    inline static void dm(int x, int y, Mat& d, float& out) {
        vector<float> temp = { {(d.at<float>(x - 1, y))}, {(d.at<float>(x - 1, y - 1))}, {(d.at<float>(x - 1, y + 1))}, {(d.at<float>(x + 1, y))},
        {(d.at<float>(x + 1, y - 1))}, {(d.at<float>(x + 1, y + 1))}, {(d.at<float>(x, y - 1))}, {(d.at<float>(x, y + 1))} };
        out = medianMe(temp);
    }
    static double medianMe(vector<float>& vec)
    {
        typedef vector<int>::size_type vec_sz;

        vec_sz size = vec.size();
        if (size == 0)
            throw domain_error("median of an empty vector");

        sort(vec.begin(), vec.end());

        vec_sz mid = size/2;

        return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
    }
};


#endif //CLION_PROJECT_Fuzzy_C_Means_S2_H
