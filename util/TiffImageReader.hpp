//
//  TiffImageReader.hpp
//  DocScanImageProcessing
//
//  Created by Tanmoy on 7/2/17.
//  Copyright © 2017 Tanmoy. All rights reserved.
//

#ifndef TiffImageReader_hpp
#define TiffImageReader_hpp
#include <stdio.h>
#include <dirent.h>
#include <ios>

#include <sys/uio.h>     // For access().
#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

#include <fstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include <vector>
#include <sstream>// used for istringstream
#include <map>
#include <utility>
#include <algorithm>
//namespace libtiff {
//#include "tiffio.h"
//}
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

//using namespace std;
//class TiffImageReader{
//    public :
//    void readTiffFiles(string);
//};
#endif /* TiffImageReader_hpp */
