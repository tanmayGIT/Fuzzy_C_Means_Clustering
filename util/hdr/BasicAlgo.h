/*
 * BasicAlgo.h
 *
 *  Created on: Feb 18, 2015
 *      Author: tanmoymondal
 */

#ifndef BASICALGO_H_
#define BASICALGO_H_
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/features2d/features2d.hpp>

#include "opencv/cv.h"
#include "opencv/cxcore.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <set>
#include <iterator>
#include <string.h>
#include <math.h>
#include <limits>
#include <stdexcept>

using namespace cv;
namespace std {
    typedef vector<CvPoint> PointArr;
    typedef vector< vector<CvPoint> > PointIn2DArr;
    typedef vector<CvRect> RectArr;
    
    struct pointData {
        float cornerResponse;
        cv::Point point;
    };
    class BasicAlgo{
    public:
        BasicAlgo();
        static BasicAlgo* getInstance();
        virtual ~BasicAlgo();
        
        int* quicksort (int*, int, int);
        
        int partition(vector<pair<int,double> >&, double, double);
        vector<int> instersection(vector<int> &, vector<int> &);
        void calculateHistogram_1(const unsigned char*, const int, const int, const int, unsigned*, int);
        void calculateHistogram_2(const unsigned char*, const int, const int, const int, int*, bool, const int );

        vector<double> generateRange(double, double, double);
        vector<float> linspace(double, double, int, double &);
        void MatToVector(const Mat&, vector<float>&);
        void VectorToMat(const vector<float>&,  Mat&);

        void showImage(cv::Mat& );
        void showIplImage(IplImage* );
        void writeImage(cv::Mat& );
//        int* convertVectorToArray(vector<int>&);
        void writeImageGivenPath(cv::Mat&, string);
        
        vector<std::pair<int,double> >& quickSort(vector<pair<int,double> >&, double, double);
        
        void writeMatrixIntoFiles(string filePath, cv::Mat imgMat);
        void printMatrix(cv::Mat imgMat);
        IplImage MatToIplImage(cv::Mat);
        IplImage* MatToIplImageStar(Mat& image1);
        static void displayPointVector(vector<cv::Point>);
        static cv::Mat markInImage(cv::Mat&, vector<pointData>, int);

        double getPointDistant(const CvPoint, const CvPoint);
        inline void convertCvPointVectorToArray(const PointArr &CvPointVector, CvPoint* CvPointArray)
        {
            for (unsigned int i = 0; i < CvPointVector.size(); ++i)
                CvPointArray[i] = CvPointVector[i];
        }
        inline void shiftCvPointBuffer(PointArr &cvPointBuffer, CvPoint newPoint)
        {
            for (int i = (int)(cvPointBuffer.size()-2); i >= 0; i--)
                cvPointBuffer[i+1] = cvPointBuffer[i];
            
            cvPointBuffer[0] = newPoint;
            
        }
        template <class myType>
        myType* convertVectorToArray(vector<myType>& arrayOfVals){
            int arrSz = (int) arrayOfVals.size();
            
            myType* keepWhiteRunLengths = NULL;
            keepWhiteRunLengths = new myType[arrSz];
            
            if(arrayOfVals.size() == arrSz){
                for (int i = 0; i<arrSz; i++){
                    keepWhiteRunLengths[i] = arrayOfVals.at(i);
                }
            }
            else{
                assert("There is some issue becuase the lengh of the vector and white run should be same");
            }
            return keepWhiteRunLengths;
        }
        
        
        template <class myType>
        int indexofSmallestElement(myType array[], int size)
        {
            int index = 0;
            
            for(int i = 1; i < size; i++)
            {
                if(array[i] < array[index])
                    index = i;
            }
            
            return index;
        }
        
        template <class myType>
        inline void shiftVector(vector<myType> &vec, myType element)
        {
            for (int i = (vec.size()-2); i >= 0; i--)
                vec[i+1] = vec[i];
            
            vec[0] = element;
        }
        template <class myVectorType>
        inline int getMaxVector(vector<myVectorType> *vec, int startPt, int endPt){
            // remember that the maximum value can be present several time but so in that case we will consider the furthest index of this maximum value.
            int maxVal = 0;
            std::vector<myVectorType> maxIndex;
            for (int i = startPt; i <= endPt; ++i ){
                if(vec->at(i) >= maxVal){
                    maxVal = vec->at(i);
                    maxIndex.push_back(i);
                }
            }
            auto biggest = std::max_element(std::begin(maxIndex), std::end(maxIndex));
            
            return *biggest;
        }
        template <class myVectorType>
        inline int getMinVector(vector<myVectorType> *vec, int startPt, int endPt){
            // remember that the maximum value can be present several time but so in that case we will consider the furthest index of this maximum value.
            int maxVal = 0;
            std::vector<myVectorType> maxIndex;
            for (int i = startPt; i <= endPt; ++i ){
                if(vec->at(i) >= maxVal){
                    maxVal = vec->at(i);
                    maxIndex.push_back(i);
                }
            }
            auto smallest = std::min_element(std::begin(maxIndex), std::end(maxIndex));
            
            return *smallest;
        }
        
        std::vector<float> uniqueValuesInMat(const cv::Mat& rawData, bool sort = false)
        {
            Mat input;
            rawData.convertTo(input, CV_32F);
            if (input.channels() > 1 || input.type() != CV_32F)
            {
                std::cerr << "unique !!! Only works with CV_32F 1-channel Mat" << std::endl;
                return std::vector<float>();
            }
            
            std::vector<float> out;
            for (int y = 0; y < input.rows; ++y)
            {
                const float* row_ptr = input.ptr<float>(y);
                for (int x = 0; x < input.cols; ++x)
                {
                    float value = row_ptr[x];
                    
                    if ( std::find(out.begin(), out.end(), value) == out.end() )
                        out.push_back(value);
                }
            }
            
            if (sort)
                std::sort(out.begin(), out.end());
            
            return out;
        }
        
        double GetMedian(cv::Mat Input, int);
        cv::Point2f convert_pt(cv::Point2f,int,int);
        void connectComponent(IplImage* inputframe, const int poly_hull0, const float perimScale,
                              int *num, RectArr &rects, PointArr &centers);
        
        void drawCC(IplImage* targetFrame, const int num, const RectArr &rects, const PointArr &centers);
        
        void resampleByPoints(const IplImage* input, const int srMar, const PointArr &points,
                              IplImage* output);
        
        void packClusters(int clusterCount, const CvMat* points, CvMat* clusters, PointIn2DArr &clusterContainer);
        
        void removeEmptyClusters(PointIn2DArr &clusterContainer);
        
        void getClusterCenters(PointIn2DArr &points2DArr, PointArr &centers);
        
        int mergeClusters(PointArr &clusterCenters, PointIn2DArr &clusterPoints, double MERGE_TH);
        
        int clustering(const PointArr &points, int clusterNum, double mergeRange,
                       PointIn2DArr &clusterContainer, PointArr &clusterMass);

        void refineSkinArea(const IplImage* skin,int clusterNum, PointArr &points,IplImage* output, int* num, RectArr &rects, PointArr &centers);
        
        CvPoint avgCvPoints(const PointArr &points, int num);
        cv::Mat& convertImage2D_3D(cv::Mat&);
        void drawHands(IplImage* targetFrame, const PointArr &currentHand, int num, CvSize sz);
        
        void drawHandsBig(IplImage* targetFrame, const PointArr &currentHand, int num, CvSize sz);
        
        void showMaskPart(const IplImage* src, const IplImage* mask, IplImage* result);
        
        void maskByRects(const IplImage* src, const RectArr &rects, IplImage* result);
        
        int getMaxRect(RectArr &rects);

        cv::Mat mat2gray(const cv::Mat& src)
        {
            Mat dst;
            cv::normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

            return dst;
        }

        inline void resetPointIn2DArr(PointIn2DArr &p2D)
        {
            for (unsigned int i = 0; i < p2D.size(); ++i)
                for (unsigned int j = 0; j < p2D[0].size(); ++j)
                    p2D[i][j] = cvPoint(-1, -1);
            
        }
        
        inline void resetPointArr(PointArr &p)
        {
            for (unsigned int i = 0; i < p.size(); ++i)
                p[i] = cvPoint(-1, -1);
        }
        
        template <class myType>
        inline string num2str(myType &i)
        {
            string s;
            stringstream ss(s);
            ss << i;
            
            return ss.str();
        }
        
        template <class myType>
        inline void attachText(IplImage* img, string prefix, myType para, int x, int y, float size = 0.5)
        {
            string out = prefix + num2str<myType>(para);
            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, size, size, 0, 2, CV_AA);
            cvPutText(img, out.c_str(), cvPoint(x, y), &font, cvScalar(0, 0, 255, 0));
        }
        inline void attachText1(IplImage* img, string prefix, int x, int y, float size = 0.5)
        {
            string out = prefix;
            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, size, size, 0, 2, CV_AA);
            cvPutText(img, out.c_str(), cvPoint(x, y), &font, cvScalar(0, 0, 255, 0));
        }
    private:
        CvPoint findMostaSimilar(PointArr &clusterCenters, double MERGE_TH, bool* mergeDone);
        void kMeans(const PointArr &dataVector, const int clusterCount, PointIn2DArr &clusterContainer);
        void merging(CvPoint toFrom, PointArr &clusterCenters, PointIn2DArr &clusterPoints);
        
        static BasicAlgo* instance;
    };
    
} /* namespace std */

#endif /* BASICALGO_H_ */
