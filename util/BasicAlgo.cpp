/*
 * BasicAlgo.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: tanmoymondal
 */

#include "hdr/BasicAlgo.h"
namespace std {
    
    BasicAlgo* BasicAlgo::instance = 0;
    
    BasicAlgo* BasicAlgo::getInstance() {
        if (!instance)
            instance = new BasicAlgo();
        
        return instance;
    }
    
    BasicAlgo::BasicAlgo() {
    }
    
    BasicAlgo::~BasicAlgo() {
    }
    
    
    void BasicAlgo:: writeMatrixIntoFiles(string filePath, cv::Mat imgMat){
        ofstream myfile (filePath);
        if (myfile.is_open())
        {
            for (int i = 0; i<imgMat.rows;i++){
                for (int j = 0; j<imgMat.cols;j++){
                    myfile << (int)imgMat.at<uchar>(i,j) << " " ;
                }
                myfile << endl ;
            }
            myfile.close();
        }
        else cout << "Unable to open file";
    }

    vector<double> BasicAlgo::generateRange(double a, double b, double c) {
        vector<double> array;
        while(a <= c) {
            array.push_back(a);
            a += b;         // could recode to better handle rounding errors
        }
        return array;
    }


    vector<float> BasicAlgo::linspace(double a, double b, int n, double & step) {
        vector<float> array;
        step = (b-a) / (n-1);

        while(a <= b) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }

    void BasicAlgo::MatToVector(const Mat& in, vector<float>& out)
    // Convert a 1-channel Mat<float> object to a vector.
    {
        if (in.isContinuous()) { out.assign((float*)in.datastart, (float*)in.dataend); }
        else {
            for (int i = 0; i < in.rows; ++i)
            { out.insert(out.end(), in.ptr<float>(i), in.ptr<float>(i) + in.cols); }
        }     return;
    }


    void BasicAlgo::VectorToMat(const vector<float>& in,  Mat& out)
    {
        vector<float>::const_iterator it = in.begin();
        MatIterator_<uchar> jt, end;
        jt = out.begin<uchar>();
        for (; it != in.end(); ++it) { *jt++ = (uchar)(*it * 255); }
    }

    void BasicAlgo:: printMatrix(cv::Mat imgMat){
        for (int i = 0; i<imgMat.rows;i++){
            for (int j = 0; j<imgMat.cols;j++){
                
                cout << (int)imgMat.at<uchar>(i,j) << " " ;
            }
            cout << endl ;
        }
    }


    void  BasicAlgo::calculateHistogram_1(const unsigned char* yuv420sp, const int yuvWidth,
                            const int yuvHeight, const int canvasHeight, unsigned* histogramBins, int BINS)
    {
        // Clear the output
        memset(histogramBins, 0, BINS * sizeof(unsigned));

        // Get the bin values
        const unsigned char *last = &yuv420sp[yuvWidth*yuvHeight];

        for ( ; yuv420sp != last; ++yuv420sp)
            ++histogramBins[*yuv420sp];

        // Resolve the maximum count in bins
        unsigned maxBinValue = 1;
        const unsigned *lastbin = &histogramBins[BINS];

        for (unsigned *bin = histogramBins; bin != lastbin; ++bin)
            if (*bin > maxBinValue)
                maxBinValue = *bin;

        // Normalize to fit into the histogram UI canvas height
        unsigned scale = canvasHeight / maxBinValue;

        for (unsigned *bin = histogramBins; bin != lastbin; ++bin)
            *bin *= scale;
    }

    void BasicAlgo:: displayPointVector(vector<cv::Point> vp){
        vector<cv::Point>::iterator pIterator;
        
        for(int i=0; i<vp.size(); i++) {
            cv::Point p = vp[i];
            
            cout << p.x << "," << p.y << "; ";
        }
    }
    
    cv::Mat BasicAlgo:: markInImage(cv::Mat& img, vector<pointData> points, int radius) {
        cv::Mat retImg;
        img.copyTo(retImg);
        
        for(vector<pointData>::iterator it = points.begin(); it != points.end(); ++it) {
            cv::Point center = (*it).point;
            // down
            for(int r=-radius; r<radius; r++) {
                retImg.at<cv::Vec3b>(cv::Point(center.y+r,center.x+radius)) = Vec3b(0, 0, 255);}
            // up
            for(int r=-radius; r<radius; r++) {
                retImg.at<Vec3b>(Point(center.y+r,center.x-radius)) = Vec3b(0, 0, 255);
            }
            // left
            for(int c=-radius; c<radius; c++) {
                retImg.at<Vec3b>(Point(center.y-radius,center.x+c)) = Vec3b(0, 0, 255);
            }
            // right
            for(int c=-radius; c<radius; c++) {
                retImg.at<Vec3b>(Point(center.y+radius,center.x+c)) = Vec3b(0, 0, 255);
            }
            retImg.at<Vec3b>(Point(center.y,center.x)) = Vec3b(0, 255, 0);
        }
        return retImg;
    }
  
    /**
     *   @brief  Convert 2D image into 3D image
     */
    cv::Mat& BasicAlgo:: convertImage2D_3D(cv::Mat& originalImg) {
        
        static Mat getStripe3D = Mat::zeros(originalImg.rows, originalImg.cols,CV_8UC3 );
        // Generate the 3D image stripe to make it coloured from the 2D image stripe
        for (int iRw = 0; iRw < originalImg.rows; iRw++){
            for (int jCol = 0; jCol < originalImg.cols; jCol++){
                getStripe3D.at<cv::Vec3b>(iRw,jCol)[0] = originalImg.at<uchar>(iRw,jCol);
                getStripe3D.at<cv::Vec3b>(iRw,jCol)[1] = originalImg.at<uchar>(iRw,jCol);
                getStripe3D.at<cv::Vec3b>(iRw,jCol)[2] = originalImg.at<uchar>(iRw,jCol);
            }
        }
        return getStripe3D;
    }
    
    /**
     *   @brief  Performs partition of the list and sort each partition recursively
     *
     *   @param  The address of the 2D vector
     *   @param  Start index of the vector
     *   @param  End index of the vector
     *   @return The index of the top element
     */
    int BasicAlgo::partition(vector<pair<int,double> >& theList, double start, double end) {
        int pivot = theList[end].second;
        int bottom = start - 1;
        int top = end;
        
        bool notdone = true;
        while (notdone) {
            while (notdone) {
                bottom += 1;
                
                if (bottom == top) {
                    notdone = false;
                    break;
                }
                if (theList[bottom].second > pivot) {
                    theList[top].second = theList[bottom].second;
                    theList[top].first = bottom;
                    break;
                }
            }
            while (notdone) {
                top = top - 1;
                
                if (top == bottom) {
                    notdone = false;
                    break;
                }
                if (theList[top].second < pivot) {
                    theList[bottom].second = theList[top].second;
                    theList[bottom].first = top;
                    break;
                }
            }
        }
        theList[top].second = pivot;
        return top;
    }
    /**
     *   @brief  Displaying the images
     */
    void BasicAlgo::showImage(cv::Mat& image) {
        cv::namedWindow("Display Image",cv::WINDOW_AUTOSIZE );
        cv::imshow("Display Image", image);
        cvWaitKey(0);
        cvDestroyWindow( "Display Image" );
    }

    void BasicAlgo::showIplImage(IplImage* image) {
        cvNamedWindow("image");
        cvShowImage("foobar", image);
        cvWaitKey(0);
        // cvReleaseImage(&image);
        cvDestroyAllWindows();
    }

    void BasicAlgo::writeImage(cv::Mat& image) {
        imwrite( "/Users/tmondal/Documents/Save_Image.jpg", image );
    }

    IplImage BasicAlgo::MatToIplImage(cv::Mat image1){
        IplImage* image2;
        image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,3);
        IplImage ipltemp = image1;
        cvCopy(&ipltemp,image2);
        return ipltemp;
    }

    IplImage* BasicAlgo::MatToIplImageStar(Mat& image1){
        IplImage* image2;
        image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,image1.type());
        IplImage ipltemp = image1;
        cvCopy(&ipltemp,image2);
        IplImage* new_image = &ipltemp;
        return new_image;
    }

    void BasicAlgo::writeImageGivenPath(cv::Mat& image, string path) {
        try{
            // BasicAlgo::getInstance()->printMatrix(image);
            imwrite( path, image );
        } catch (const std::exception& e) {
            BasicAlgo::getInstance()->printMatrix(image);
            std::cout << e.what(); // information from length_error printed
        }
    }
    /**
     *   @brief  Call the function for the quick sort for each of the two parts
     *
     *   @param  The address of the 2D vector
     *   @param  Start index of the vector
     *   @param  End index of the vector
     *   @return The address of 2D vector, containing real indexes in 1st col and sorted values in 2nd col
     */
    vector<std::pair<int,double> >& BasicAlgo::quickSort( vector<std::pair<int,double> >& theList, double start, double end) {
        if (start < end) {
            int split = partition(theList, start, end); //recursion
            quickSort(theList, start, split - 1);
            quickSort(theList, split + 1, end);
            return theList;
        }
        else{
            throw invalid_argument( "received invalid function arguments" );
        }
    }
    void drawCC(IplImage* targetFrame, const int num, const RectArr &rects, const PointArr &centers)
    {
        /*
         *  Draw the connected component centers and bounding boxes to "targetFramw"
         */
        
        if (num != 0) {
            for (int i = 0; i < num; ++i) {
                cvCircle(targetFrame, centers[i], 5, CV_RGB(0x00, 0xff, 0xff), -1);
                cvRectangle(targetFrame, cvPoint(rects[i].x, rects[i].y),
                            cvPoint(rects[i].x + rects[i].width,  rects[i].y + rects[i].height),
                            CV_RGB(255, 255, 0), 1);
            }
        }
        
    }
    inline double getPointDistant(const CvPoint point1, const CvPoint point2)
    {
        // Get the distant of two CvPoints
        return sqrt( (double)( (point1.x-point2.x)*(point1.x-point2.x) +
                              (point1.y-point2.y)*(point1.y-point2.y)  ) );
    }
    void resampleByPoints(const IplImage* input, const int srMar, const PointArr &points,IplImage* output)
    {
        
        /*
         *  Pre:  "input" is the source to be resampled
         *        "srMar" is the distant from resampling center to form a square
         *        "points" I take these points as centers to select square resampling area
         *
         *  Post: "output" the resampling result
         */
        
        cvSetZero(output); // clear output image
        CvSize sz = cvGetSize(output);
        
        for (unsigned int i = 0; i < points.size(); ++i) {
            
            CvPoint leftTop     = cvPoint(points[i].x - srMar, points[i].y - srMar);
            CvPoint rightBottom = cvPoint(points[i].x + srMar, points[i].y + srMar);
            
            if (leftTop.x < 0)
                leftTop.x = 0;
            
            if (leftTop.y < 0)
                leftTop.y = 0;
            
            if (rightBottom.x > sz.width)
                rightBottom.x = sz.width;
            
            if (rightBottom.y > sz.height)
                rightBottom.y = sz.height;
            
            for (int j = leftTop.y; j < rightBottom.y; j++) {
                
                uchar* ptr = (uchar*)(input->imageData + j*input->widthStep);
                uchar* ptrNew = (uchar*)(output->imageData + j*output->widthStep);
                
                for (int k = leftTop.x; k < rightBottom.x; k++)
                    ptrNew[k] = ptr[k];
            }
        }
    }
    cv::Point2f BasicAlgo::convert_pt(cv::Point2f point,int w,int h)
    {
        //center the point at 0,0
        cv::Point2f pc(point.x-w/2,point.y-h/2);
        
        //these are your free parameters
        float f = w;
        float r = w;
        
        float omega = w/2;
        float z0 = f - sqrt(r*r-omega*omega);
        
        float zc = (2*z0+sqrt(4*z0*z0-4*(pc.x*pc.x/(f*f)+1)*(z0*z0-r*r)))/(2* (pc.x*pc.x/(f*f)+1));
        cv::Point2f final_point(pc.x*f/zc,pc.y*f/zc);
        final_point.x += w/2;
        final_point.y += h/2;
        return final_point;
    }
    double BasicAlgo::GetMedian(cv::Mat imgMat, int iSize) {
        float * testData1D = (float*)imgMat.data;
        // Allocate an array of the same size and sort it.
        double* dpSorted = new double[iSize];
        for (int i = 0; i < iSize; ++i) {
            dpSorted[i] = testData1D[i];
        }
        for (int i = iSize - 1; i > 0; --i) {
            for (int j = 0; j < i; ++j) {
                if (dpSorted[j] > dpSorted[j+1]) {
                    double dTemp = dpSorted[j];
                    dpSorted[j] = dpSorted[j+1];
                    dpSorted[j+1] = dTemp;
                }
            }
        }
        
        // Middle or average of middle values in the sorted array.
        double dMedian = 0.0;
        if ((iSize % 2) == 0) {
            dMedian = (dpSorted[iSize/2] + dpSorted[(iSize/2) - 1])/2.0;
        } else {
            dMedian = dpSorted[iSize/2];
        }
        delete [] dpSorted;
        return dMedian;
    }
    
    void packClusters(int clusterCount, const CvMat* points, CvMat* clusters, PointIn2DArr &clusterContainer)
    {
        for (int i = 0; i < clusterCount; i++) {
            
            PointArr tempClass;
            
            for (int row = 0; row < clusters->rows; row++) {
                
                float* p_point = (float*)(points->data.ptr + row*points->step);
                int X = static_cast<int>(*p_point) ;
                p_point++;
                int Y = static_cast<int>(*p_point);
                
                if (clusters->data.i[row] == i)
                    tempClass.push_back(cvPoint(X, Y));
            }
            
            clusterContainer.push_back(tempClass);
            
        }
    }
    void removeEmptyClusters(PointIn2DArr &clusterContainer)
    {
        for (unsigned int i = 0; i < clusterContainer.size(); ++i) {
            
            if ( clusterContainer[i].empty() ) {
                PointIn2DArr::iterator iter = clusterContainer.begin();
                iter = iter + i;
                clusterContainer.erase(iter);
                i = i - 1;
            }
        }
    }
    void getClusterCenters(PointIn2DArr &points2DArr, PointArr &centers)
    {
        /*
         *  Pre : "clusterContainer" is classfied point set by K-Means
         *
         *  Post: "clusterMass" is the calculated mass center of each classified cluster points
         */
        
        for (unsigned int i = 0; i < points2DArr.size(); i++) {
            
            int x = 0, y = 0;
            PointArr tempClass(points2DArr[i]);
            
            for (unsigned int j = 0; j < tempClass.size(); j++) {
                x += tempClass[j].x;
                y += tempClass[j].y;
            }
            
            centers.push_back(cvPoint(x/tempClass.size(), y/tempClass.size()));
            
        }
    }
    
    CvPoint findMostaSimilar(PointArr &clusterCenters, double MERGE_TH, bool* mergeDone)
    {
        *mergeDone = true;
        
        int mergeToIndex = 0;
        int mergeFromIndex = 0;
        double centerDis = 0;
        double minDis = 1000000;
        
        for (unsigned int i = 0; i < clusterCenters.size()-1; ++i) {
            for (unsigned int j = i+1; j < clusterCenters.size(); ++j) {
                
                centerDis = getPointDistant(clusterCenters[i], clusterCenters[j]);
                
                if ((centerDis < MERGE_TH) && (centerDis < minDis)) {
                    
                    minDis = centerDis;
                    
                    if (i < j) {
                        mergeToIndex = i;
                        mergeFromIndex = j;
                    } else {
                        mergeToIndex = j;
                        mergeFromIndex = i;
                    }
                    
                    *mergeDone = false;
                    
                }
            }
        }
        
        return cvPoint(mergeToIndex, mergeFromIndex);
        
    }
    
    
    void merging(CvPoint toFrom, PointArr &clusterCenters, PointIn2DArr &clusterPoints)
    {
        PointIn2DArr mergedClusterPoints;
        PointArr mergedClusterCenters;
        unsigned int toIdx = (unsigned int)toFrom.x;
        unsigned int frIdx = (unsigned int)toFrom.y;
        
        for (unsigned int i = 0; i < clusterCenters.size(); ++i) {
            
            if (i == toIdx) {
                
                PointArr tempClass(clusterPoints[toIdx]);
                
                tempClass.insert(tempClass.end(), clusterPoints[frIdx].begin(), clusterPoints[frIdx].end());
                
                mergedClusterPoints.push_back(tempClass);
                
            } else if ((i != toIdx) && (i != frIdx)) {
                
                mergedClusterPoints.push_back(clusterPoints[i]);
                
            }
            
        }
        
        getClusterCenters(mergedClusterPoints, mergedClusterCenters);
        
        clusterPoints.clear();
        clusterCenters.clear();
        
        clusterPoints = mergedClusterPoints;
        clusterCenters = mergedClusterCenters;
        
    }
    int mergeClusters(PointArr &clusterCenters, PointIn2DArr &clusterPoints, double MERGE_TH)
    {
        /*
         *  Pre:  "clusterMass" mass center of "clusterContainer"
         *        "clusterContainer" is the container of the clustered points
         *        "MERGE_TH" defines how near two "clusterMass" will be merged
         *
         *  Post: return the remaining cluster count after merging
         */
        
        bool mergeDone = false;
        
        while (mergeDone != true) {
            
            CvPoint toFrom = findMostaSimilar(clusterCenters, MERGE_TH, &mergeDone);
            
            if (mergeDone == false)
                merging(toFrom, clusterCenters, clusterPoints);
            
        }
        
        return clusterCenters.size();
        
    }
    CvPoint avgCvPoints(const PointArr &points, int num)
    {
        // Calculate the average of the points vector excluding (-1, -1)
        
        int bufRealCenterCount = 0;
        int bufAccCenter_x = 0, bufAccCenter_y = 0;
        
        for (int i = 0; i < num; i++) {
            if ( (points[i].x != -1) && (points[i].y != -1) ) {
                bufAccCenter_x += points[i].x;
                bufAccCenter_y += points[i].y;
                bufRealCenterCount++;
            }
        }
        
        if (bufRealCenterCount > 0) {
            int bufCenter_x = bufAccCenter_x/bufRealCenterCount;
            int bufCenter_y = bufAccCenter_y/bufRealCenterCount;
            
            return cvPoint(bufCenter_x, bufCenter_y);
            
        } else {
            
            // if all points in buffer are (-1, -1), the hand point will disappear
            return cvPoint(-1, -1);
        }
        
    }
    
    void showMaskPart(const IplImage* src, const IplImage* mask, IplImage* result)
    {
        /* src is the source image which you want to mask
         * mask is a single channel binary image as a mask
         * result is the image with the same size, depth, channel with src
         */
        
        cvZero(result);
        
        CvSize sz = cvSize(src->width, src->height);
        IplImage* refImg = cvCreateImage(sz, src->depth, src->nChannels);
        cvZero(refImg);
        
        cvOr(src, refImg, result, mask);
        
        cvReleaseImage(&refImg);
        
    }
    
    void maskByRects(const IplImage* src, const RectArr &rects, IplImage* result)
    {
        cvZero(result);
        
        for (unsigned int i = 0; i < rects.size(); ++i) {
            
            CvPoint leftTop     = cvPoint(rects[i].x, rects[i].y);
            CvPoint rightBottom = cvPoint(rects[i].x + rects[i].width, rects[i].y + rects[i].height);
            
            for (int j = leftTop.y; j < rightBottom.y; j++) {
                
                uchar* ptr = (uchar*)(src->imageData + j*src->widthStep);
                uchar* ptrNew = (uchar*)(result->imageData + j*result->widthStep);
                
                for (int k = leftTop.x; k < rightBottom.x; k++)
                    ptrNew[k] = ptr[k];
            }
        }
    }
    int getMaxRect(RectArr &rects)
    {
        RectArr::iterator it = rects.begin();
        
        int maxArea = 0;
        
        while (it != rects.end()) {
            
            if ( (it->width * it->height) > maxArea)
                maxArea = it->width * it->height;
            
            it++;
        }
        
        maxArea = (maxArea / 100) * 100;
        
        return maxArea;
        
    }
    
    /**
     *   @brief  Perform k-means clustering with data points
     *
     *   @param  "dataVector" the data to be clustered by K-Means
     *   @param  "clusterCount" how many clusters you want
     *   @return "classContainer" I pack the points with the same cluster into vector, so it
     *   is a vector of vector
     */
    void kMeans(const PointArr &dataVector, const int clusterCount, PointIn2DArr &clusterContainer)
    {
        /*
         *  Pre:  "dataVector" the data to be clustered by K-Means
         *        "clusterCount" how many clusters you want
         *
         *  Post: "classContainer" I pack the points with the same cluster into vector, so it
         *        is a vetor of vector
         */
        
        int dataLength = dataVector.size();
        
        // Put data into suitable container
        CvMat* points   = cvCreateMat(dataLength, 1, CV_32FC2);
        CvMat* clusters = cvCreateMat(dataLength, 1, CV_32SC1 );
        
        for (int row = 0; row < points->rows; row++) {
            float* ptr = (float*)(points->data.ptr + row*points->step);
            for (int col = 0; col < points->cols; col++) {
                *ptr = static_cast<float>(dataVector[row].x);
                ptr++;
                *ptr = static_cast<float>(dataVector[row].y);
            }
        }
        
        // The Kmeans algorithm function (OpenCV function)
        cvKMeans2(points, clusterCount, clusters, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1, 2));
        
        // Pack result to 'classContainer': each element in 'classContainer' means one cluster,
        // each cluster is one PointArr contain all points belong to this cluster
        packClusters(clusterCount, points, clusters, clusterContainer);
        
        removeEmptyClusters(clusterContainer);
        
        cvReleaseMat(&points);
        cvReleaseMat(&clusters);
        
    }
    int clustering(const PointArr &points, int clusterNum, double mergeRange,
                   PointIn2DArr &clusterContainer, PointArr &clusterMass)
    {
        /*
         *  Pre:  "points" are the data points you want to clustering by K-Means
         *        "clusterNum" defines how many cluster you want
         *        "mergeRange" defines the distance threshold between different clusters
         *
         *  Post: "clusterContainer" is vector of clustered points vector
         *        "clusterMass" is vector of the mass center of clustered points vector
         */
        
        kMeans(points, clusterNum, clusterContainer);
        getClusterCenters(clusterContainer, clusterMass);
        
        return mergeClusters(clusterMass, clusterContainer, mergeRange);
        
    }

    
    vector<int> instersection(vector<int> &v1, vector<int> &v2)
    {
        
        vector<int> v3;
        
        sort(v1.begin(), v1.end());
        sort(v2.begin(), v2.end());
        
        set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(),back_inserter(v3));
        
        return v3;
    }
    
    /**
     *   @brief  Performs quick sort of a 1D array
     *
     *   @param  The pointer of 1D array
     *   @param  Start index of the array
     *   @param  End index of the array
     *   @return pointer of sorted 1D array
     */
    int* BasicAlgo::quicksort (int *array, int start, int end )
    {
        //	static unsigned int calls = 0;
        
        //cout << "QuickSort Call #: " << ++calls << endl;
        
        //function allows one past the end to be consistent with most function calls
        // but we normalize to left and right bounds that point to the data
        
        int leftbound = start;
        int rightbound = end - 1;
        
        if (rightbound <= leftbound )
            return NULL;
        
        int pivotIndex = leftbound + (rand() % (end - leftbound));
        int pivot = array[pivotIndex];
        
        // cout << " Pivot: " << "[" << pivotIndex << "] " << pivot << endl;
        int leftposition = leftbound;
        int rightposition = rightbound; // accounting for pivot that was moved out
        
        while ( leftposition < rightposition )
        {
            while ( leftposition < rightposition && array[leftposition] < pivot )
                ++leftposition;
            
            while ( rightposition > leftposition && array[rightposition] > pivot )
                --rightposition;
            
            if(leftposition < rightposition)
            {
                if (array[leftposition] != array[rightposition])
                {
                    swap(array[leftposition],array[rightposition]);
                    //		cout << " Swapping RightPosition: " << right position << " and LeftPosition: " << left position << endl;
                }
                else
                    ++leftposition;
            }
        }
        
        // sort leaving the pivot out
        quicksort (array,leftbound, leftposition);  // left position is at the pivot which is one past the data
        quicksort (array,leftposition + 1,end);     // left position + 1 is past the pivot till the end
        return array;
    }
}
