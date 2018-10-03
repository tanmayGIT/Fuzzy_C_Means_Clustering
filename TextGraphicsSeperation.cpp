
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

    cout << "You have entered " << argc << " arguments:" << "\n";
    cout << "Image Path is : " << argv[1] << "\n";  // just pass the image path as the input argument

    Mat imgOrig = imread(argv[1]); // imread("/Users/tmondal/Documents/wheel.png");
    if( imgOrig.empty())
    {
        cout << "File not available for reading"<<endl;
        return -1;
    }

    Mat outputFuzzyImg;
    auto start = chrono::steady_clock::now();


  //  Fuzzy_C_Means_S1::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);
  //  Fuzzy_C_Means_S2::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);



  //  Fuzzy_C_Means_M1::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);



  //  Fuzzy_C_Means_FGfsm::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);
  //  Fuzzy_C_Means_FGfsm_S1::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);
  //  Fuzzy_C_Means_FGfsm_S2::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);



    Fuzzy_C_Means_FLIcm::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);



  //  Fuzzy_C_Means_ENFCM::getInstance()->applyAlgoOnImage(imgOrig, outputFuzzyImg);



    BasicAlgo::getInstance()->showImage(outputFuzzyImg);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << chrono::duration <double, milli> (diff).count() << " ms time taken for execution" << endl;

}

