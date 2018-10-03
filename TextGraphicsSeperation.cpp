
//
//  TextGraphicsSeperation.cpp
//  DocScanImageProcessing
//
//  Created by tmondal on 16/07/2018.
//  Copyright © 2018 Tanmoy. All rights reserved.
//


#include <stdio.h>

#include "PatternRecogAlgos/Clustering/K_Means_Clustering.h"

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/hdr/Fuzzy_C_Means_S1.h"
#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/hdr/Fuzzy_C_Means_S2.h"

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_M1/hdr/Fuzzy_C_Means_M1.h"

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm.h"
#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm_S1.h"
#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm_S2.h"

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FLIcm/hdr/Fuzzy_C_Means_FLIcm.h"

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_EN_FCM/hdr/Fuzzy_C_Means_ENFCM.h"
#include <opencv2/ximgproc.hpp>
#include "opencv2/core/utility.hpp"
#include <highgui.h>
#include <cv.h>


int main(int argc, char** argv) {

    cout << "You have entered " << argc << " arguments" << "\n";
    if (argc < 2) {
        std::cerr << "Usage:    " << argv[0] << "       --- You should provide File Path here" << std::endl;
        return 1;
    }

/*  **********************  ##########     Important Info       ##########  ***************************
 * just pass the image path as the input argument. In CLion, if I have to pass one argument (file path) only,
 * it considers it as 2nd argument. I don't understand the reason. So, when you run this code from terminal or other IDE
 * then be careful and change :  argv[1]   --->  argv[0]
 * */
    cout << "Image Path is : " << argv[1] << "\n";
    Mat imgOrig = imread(argv[1]); // imread("/Users/tmondal/Documents/wheel.png");
    if( imgOrig.empty())
    {
        cout << "File not available for reading"<<endl;
        return -1;
    }

    Mat outputClusteredImg;
    auto start = chrono::steady_clock::now();


  //  Fuzzy_C_Means_S1::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);
  //  Fuzzy_C_Means_S2::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);



  //  Fuzzy_C_Means_M1::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);



  //  Fuzzy_C_Means_FGfsm::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);
  //  Fuzzy_C_Means_FGfsm_S1::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);
  //  Fuzzy_C_Means_FGfsm_S2::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);



    Fuzzy_C_Means_FLIcm::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);



  //  Fuzzy_C_Means_ENFCM::getInstance()->applyAlgoOnImage(imgOrig, outputClusteredImg);


    //  *****************    K-Means Clustering   *********************
/*    int clusterCount = 5; Mat labels; int attempts = 5; Mat centers;
    K_Means_Clustering::getInstance()->applyOpenCV_KMeans_2(imgOrig, labels, centers, clusterCount, attempts, outputClusteredImg );*/


    BasicAlgo::getInstance()->showImage(outputClusteredImg);


    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << chrono::duration <double, milli> (diff).count() << " ms time taken for execution" << endl;
}