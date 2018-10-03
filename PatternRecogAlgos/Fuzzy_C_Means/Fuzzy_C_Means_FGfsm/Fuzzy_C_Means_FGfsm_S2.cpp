//
// Created by tmondal on 20/09/2018.
//
#include <unitypes.h>
#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm_S2.h"


Fuzzy_C_Means_FGfsm_S2 *Fuzzy_C_Means_FGfsm_S2::instance = 0;

Fuzzy_C_Means_FGfsm_S2 *Fuzzy_C_Means_FGfsm_S2::getInstance() {
    if (!instance)
        instance = new Fuzzy_C_Means_FGfsm_S2();

    return instance;
}

Fuzzy_C_Means_FGfsm_S2::Fuzzy_C_Means_FGfsm_S2() {
}

Fuzzy_C_Means_FGfsm_S2::~Fuzzy_C_Means_FGfsm_S2() {
}

void Fuzzy_C_Means_FGfsm_S2::padMatrix(Mat &m, int n, Mat &out) {
    Mat oldOut = m; // For the first time to initialize out matrix
    for (int ii = 0; ii < n; ii++) {
        out = Mat::zeros((oldOut.rows + 2), (oldOut.cols + 2), m.type());
        padMatrix_1(oldOut, out);
        oldOut = out;
    }
}


void Fuzzy_C_Means_FGfsm_S2::padMatrix_1(Mat &m, Mat &outModified) {

    cv::Rect rect1(0, 0, m.cols, m.rows);
    cv::Rect rect2(1, 1, (outModified.cols - 2), (outModified.rows - 2));
    m(rect1).copyTo(outModified(rect2));

    cv::Rect rect3(0, 1, 1, (outModified.rows - 2)); // first-col, second-row, 1-col-width, (no-of-rows-outModified - 2)
    (m.col(0)).copyTo(outModified(rect3));  // padding left column

    cv::Rect rect4((outModified.cols - 1), 1, 1, (outModified.rows - 2));
    (m.col(m.cols - 1)).copyTo(outModified(rect4)); // padding right column

    outModified.row(0) = (outModified.row(1) + 0); // Padding first row
    outModified.row(outModified.rows - 1) = (outModified.row((outModified.rows - 2)) + 0); // Padding last row
}


void Fuzzy_C_Means_FGfsm_S2::udDist_FCM_M1(Mat &center, Mat &data, Mat &out) {
    out = Mat::zeros(center.rows, data.rows, CV_32F);
    data.convertTo(data, center.type());
    for (int ii = 0; ii < center.rows; ii++) {
        float centerVal = center.at<float>(ii,0);
        Mat obtainedRes =  abs((centerVal - data).t());
        obtainedRes.copyTo(out.row(ii));
    }
}

void Fuzzy_C_Means_FGfsm_S2::udInit_FCM_M1(int cluster_n, int data_n, Mat &U) {

    std::mt19937_64 rng;
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);

    Mat colSumMat = Mat::zeros(1, data_n, CV_32FC1);
    for (int jj = 0; jj < data_n; jj++) {
        double sumCol = 0.0;
        for (int ii = 0; ii < cluster_n; ii++) {
            double currentRandomNumber = unif(rng);
            U.at<float>(ii, jj) = currentRandomNumber;
            sumCol = sumCol + currentRandomNumber;
        }
        colSumMat.at<float>(0, jj) = sumCol;  // putting the column sum
    }

    Mat makingDeno = Mat::zeros(cluster_n, data_n, CV_32FC1);
    int dk = 0;
    while (dk < cluster_n) {
        makingDeno.row(dk) = (colSumMat.row(0) + 0);
        dk++;
    }
    colSumMat.release();
    Mat UModified;
    cv::divide(U, makingDeno, UModified);

    U = UModified.clone();
}

void Fuzzy_C_Means_FGfsm_S2::step_FG_FCM (Mat &data, Mat& N, Mat &U, int clusterN, int expo, Mat &center, float& obj_func) {
    Mat mf;
    cv::pow(U, expo, mf);

    Mat oneMatrixInitial = Mat::ones(clusterN, 1, CV_8U);
    Mat mulMeFirst;
    Mat N_Transpose = N.t();
    multiply_2_UChar_Mat(oneMatrixInitial, N_Transpose, mulMeFirst );
    mulMeFirst.convertTo(mulMeFirst, mf.type());
    Mat mf1 = mf.mul(mulMeFirst);
    mulMeFirst.release();


    Mat mf_1_Transpose = mf1.t();
    Mat columnSum = Mat::zeros( mf_1_Transpose.cols, 1, CV_32FC1);
    for (int iyt = 0; iyt < mf_1_Transpose.cols; iyt++)
        columnSum.at<float>(iyt,0) = (cv::sum(mf_1_Transpose.col(iyt))[0]);

    Mat numera;
    multiplyTwoFloatMat(mf1, data, numera); // this is 2 float matrix
    cv::divide(numera,columnSum , center);

    Mat dist;
    udDist_FCM_M1(center, data, dist);
    Mat tmpLac;
    cv::pow(dist, 2, tmpLac);
    Mat resultMat = tmpLac.mul(mf1);

    Mat tempColSum = Mat::zeros(1,resultMat.cols, CV_32FC1);
    for (int iCol = 0; iCol<resultMat.cols; iCol++) {
        float sumCol = cv::sum(resultMat.col(iCol))[0]; // actually I am doing here sum of each column but
        tempColSum.at<float>(0,iCol) = sumCol;
    }
    obj_func = cv::sum( tempColSum)[0];

    Mat anotherTmpLac;
    cv::pow(dist, (-2 / (expo - 1)), anotherTmpLac);
    Mat makeOnes = Mat::ones(clusterN, 1, CV_8UC1);

    Mat anotherTempColSum = Mat::zeros(1,anotherTmpLac.cols, CV_32FC1);
    for (int iCol = 0; iCol<anotherTmpLac.cols; iCol++) {
        float sumCol = cv::sum(anotherTmpLac.col(iCol))[0]; // actually I am doing here sum of each column but
        anotherTempColSum.at<float>(0,iCol) = sumCol;
    }
    Mat multiplyRes;
    multiplyOpenCVMat2(makeOnes,anotherTempColSum, multiplyRes);

    Mat U_New;
    divide(anotherTmpLac, multiplyRes, U_New);

    U.release();
    U = U_New.clone();
    U_New.release();
}

void Fuzzy_C_Means_FGfsm_S2::lnwiFun(int ii, Mat& data2, std::vector<std::vector<int> >& rc, Mat& out){
    Mat data2_Transpose = data2.t();
    Mat data = data2_Transpose.reshape(1, (data2_Transpose.rows * data2_Transpose.cols));
    data2_Transpose.release();

    Mat window = neighborCalculation(rc, ii);


    //  Mat dataRefined = Mat::zeros(window.rows, data.cols, data.type());
    vector<float> dataRefinedVect;
    for (int iiWin = 0; iiWin < window.rows; iiWin++){
        int getIndex = (int) window.at<float>(iiWin,0);
     //   dataRefined.at<float>(iiWin,0) = data.at<float>(getIndex,0);
        dataRefinedVect.push_back((data.at<float>(getIndex,0)));
    }
 //   float medianVal = BasicAlgo::getInstance()->GetMedian(dataRefined, dataRefined.rows);
    float anotherMedianVal = medianMe(dataRefinedVect);
    out = Mat::zeros(1,1,CV_32F);
    out.at<float>(0,0) = anotherMedianVal;
}



Mat Fuzzy_C_Means_FGfsm_S2::neighborCalculation(std::vector<std::vector<int> >&rc, int ii1) {

    int r = rc[0][0];
    // int c = rc[1][0];

    int r1 = ii1 % r; // The rule of mod i.e. remainder after division is always equal to ii1 while i < r but when i==r, r1=0
    int c1 = 0;
    if ((r1 == 0) && (ii1 > 0)) { // that means r1 == 0 but ii1 > 0
        r1 = r;
        c1 = floor(ii1 / r);
    } else {
        c1 = floor(ii1 / r);
    }
    float dataTemp[16] = {-1, 1,   -1, 0,     1,  -1,     1,  0,       0,  1,      0,  -1,         1,  1,      -1, -1};
    cv::Mat temp = cv::Mat(8, 2, CV_32F, dataTemp);

    std::vector<int> nr;
    std::vector<int> nc;

    temp.col(0) = (temp.col(0) + (r1+1));
    temp.col(1) = temp.col(1) + (c1+1);

    float tempExtraPart[2] = {(float)(r1+1), (float)(c1+1)};
    Mat tempExtraMat = cv::Mat(1, 2, CV_32F, tempExtraPart);
    CV_Assert(temp.type() == tempExtraMat.type());

    Mat tempIncreasedMat = cv::Mat((temp.rows + tempExtraMat.rows), temp.cols, CV_32F, dataTemp);
    temp.copyTo( tempIncreasedMat( Rect(0,0,temp.cols,temp.rows) ) );
    tempExtraMat.copyTo( tempIncreasedMat( Rect(0,temp.rows,tempExtraMat.cols,tempExtraMat.rows) ) );

    float refinedTemp[2] = {(float)(1), (float)(r+2)};
    Mat refinedMat = cv::Mat(2, 1, CV_32F, refinedTemp);
    Mat out = tempIncreasedMat * refinedMat;
    return out;
}





void Fuzzy_C_Means_FGfsm_S2::fg_FSM(Mat &origFloatImg, int cluster_n, Mat &center, Mat &U, vector<float>& obj_fcn) {
    Mat data = origFloatImg.clone();
    Mat data1 = data;
    data.release();

    Mat data_1_Transpose = data1.t();
    data = data_1_Transpose.reshape(1, (data_1_Transpose.rows * data_1_Transpose.cols));
    data_1_Transpose.release();

    Mat data2;
    padMatrix(data1, 1, data2);

    int data_n = data.rows;
    float default_options[5] = {2, 100, 1e-5, 1, 3};

    float expo = default_options[0];         // Exponent for U
    float max_iter = default_options[1];    // Max. iteration
    float min_impro = default_options[2]; // Min. improvement
    float display = default_options[3]; // Display info or not

    int bitNum = 8;
    int gln = pow(2, bitNum);
    obj_fcn.reserve(max_iter);


    Mat lnai = Mat::zeros(data_n, 1, CV_32F);

    vector<vector<int>> rc ={{data1.rows}, {data1.cols}};
    vector<float>keepMeAll;

    for (int ii = 0; ii < data_n; ii++){
        Mat tempCalc;
        lnwiFun(ii, data2, rc, tempCalc);
        keepMeAll.push_back(tempCalc.at<float>(0,0));
        float multiDimVal = tempCalc.at<float>(0,0) * (gln-1);
        lnai.at<float>(ii,0) = ( std::round(multiDimVal));
    }
    double noOfPtToBeGenerated = pow(2, bitNum) +1;
    int upperLimit = pow(2, bitNum);
    double gapRange;
    vector<float> edges = BasicAlgo::getInstance()->linspace(0, upperLimit, noOfPtToBeGenerated, gapRange);

    /*
    Mat hist;
    int nimages = 1; // Only 1 image, that is the Mat.
    int channels[] = {0};
    int dims = 1;
    int totalHistSiz = edges.size();
    int histSize = totalHistSiz;
    bool uniform = true; bool accumulate = false;

    float hranges[totalHistSiz];    //{ 0, 1, 2, 3, 4, 45, 54, 180 };
    for (size_t iVec = 0; iVec < totalHistSiz; ++iVec)
        hranges[iVec] = edges[iVec];
    const float *ranges[]=  {hranges};
    for (int iK = 0; iK < totalHistSiz; iK++){
        cout << &ranges[iK] << endl;
    }
    calcHist(&lnai, nimages, channels, Mat(), hist, dims, &histSize, ranges, uniform, accumulate); */

    Mat histogramBins = Mat::zeros( (edges.size()-1), 1, CV_8U );
    vector<unsigned int > keepBinIndexOfEle((lnai.rows), 0);
    calculateHistogram_2(lnai, edges, histogramBins, gapRange, keepBinIndexOfEle);

    noOfPtToBeGenerated = pow(2, bitNum);
    upperLimit = pow(2, bitNum)-1;
    vector<float> data_N = BasicAlgo::getInstance()->linspace(0, upperLimit, noOfPtToBeGenerated, gapRange);

    data_n = (edges.size()-1);

    U = Mat::zeros(cluster_n, data_n, CV_32FC1);
    udInit_FCM_M1(cluster_n, data_n, U);     // Initial fuzzy partition


    Mat data_N_Mat = Mat(data_N.size(),1,CV_32F);
    memcpy(data_N_Mat.data, data_N.data(), data_N.size()*sizeof(float));

    int ii = 0;
    for (ii = 0; ii < max_iter; ii++) {
        float tempObjectiveFuncVal;
        Mat U_New;
        center = Mat::zeros(cluster_n, 1, CV_8U);
        step_FG_FCM(data_N_Mat, histogramBins, U, cluster_n, expo, center, tempObjectiveFuncVal);
        obj_fcn[ii] = (tempObjectiveFuncVal);

        if (display)
            cout << "Iteration count = " << ii << " Objective Function = " << tempObjectiveFuncVal << endl;
        if (ii > 0)
            if ((abs(obj_fcn[ii] - obj_fcn[ii -1])) < min_impro)
                break;
    }
    int actualNumberIter = ii;      // Actual number of iterations

    std::vector<float> newVector;
    vector<float>::const_iterator firstVect = obj_fcn.begin();
    vector<float>::const_iterator lastVect = obj_fcn.begin() + actualNumberIter;
    std::copy(firstVect, lastVect, std::back_inserter(newVector));
    obj_fcn = newVector;
    newVector.clear();

    Mat U1 = U.clone();
    U.release();
    U = Mat::zeros(U1.rows, lnai.rows, U1.type());
    for (int ik = 0; ik < lnai.rows; ik ++){
        int getMe = keepBinIndexOfEle[ik];
        U1.col(getMe).copyTo(U.col(ik));
    }
    U1.release();
}



void Fuzzy_C_Means_FGfsm_S2::applyAlgoOnImage(Mat &imgOrig, Mat &matOut) {
    if (imgOrig.channels() >= 3)
        cv::cvtColor(imgOrig, imgOrig, CV_BGR2GRAY);

    Mat center;
    Mat U;
    vector<float> obj_fcn;

    Mat imgOrigFloat;
    imgOrig.convertTo(imgOrigFloat, CV_32F);
    imgOrigFloat = imgOrigFloat / 255.0;
    // printFloatMatrix(imgOrigFloat);
    float rg = 6.2;
    int cluster_n = 4;
    vector<float> options;

    fg_FSM(imgOrigFloat, cluster_n, center, U, obj_fcn);


    double minVal = 0.0;
    double maxVal = 0.0;
    Point minLoc;
    Point maxLoc;
    Mat eachColMax = Mat::zeros(1,U.cols, CV_32FC1 );
    for (int jCol = 0; jCol < U.cols; jCol++)
    {
        Mat eachCol = U.col(jCol);
        minMaxLoc(eachCol, &minVal, &maxVal, &minLoc, &maxLoc);
        eachColMax.at<float>(0, jCol) = maxVal;
    }
    vector<int> index1;
    Mat U_1_Rw = U.row(0);
    compareTwoMatrixSimilarity(U_1_Rw, eachColMax , index1);

    vector<int> index2;
    Mat U_2_Rw = U.row(1);
    compareTwoMatrixSimilarity(U_2_Rw, eachColMax , index2);

    vector<int> index3;
    Mat U_3_Rw = U.row(2);
    compareTwoMatrixSimilarity(U_3_Rw, eachColMax , index3);

    vector<int> index4;
    Mat U_4_Rw = U.row(3);
    compareTwoMatrixSimilarity(U_4_Rw, eachColMax , index4);


    Mat rr = center;
    Mat imgOrigFloatTrans = imgOrigFloat.t();
    int lenTak = (imgOrigFloatTrans.rows * imgOrigFloatTrans.cols);
    Mat d4 = imgOrigFloatTrans.reshape(1,lenTak);
    Mat d5 = d4;
    imgOrigFloatTrans.release();


    for (int i1 = 0; i1 < index1.size(); i1++){
        int getIndex = index1[i1];
        d5.at<float>(getIndex, 0) = rr.at<float>(0,0);
    }
    for (int i2 = 0; i2 < index2.size(); i2++){
        int getIndex = index2[i2];
        d5.at<float>(getIndex, 0) = rr.at<float>(1,0);
    }
    for (int i3 = 0; i3 < index3.size(); i3++){
        int getIndex = index3[i3];
        d5.at<float>(getIndex, 0) = rr.at<float>(2,0);
    }
    for (int i4 = 0; i4 < index4.size(); i4++){
        int getIndex = index4[i4];
        d5.at<float>(getIndex, 0) = rr.at<float>(3,0);
    }

    Mat d6;
    matlab_reshape(d5, d6,  imgOrigFloat.rows, imgOrigFloat.cols, 1); // send the column first then rows
    d5.release();

    Mat normImag = d6 ;
    normImag.convertTo(normImag, CV_8U);
    d6.release();
    matOut = normImag.clone();
    //BasicAlgo::getInstance()->showImage(normImag);
    normImag.release();
}

void Fuzzy_C_Means_FGfsm_S2::calculateHistogram_2(const Mat& yuv420sp, const vector<float>& edgesBin, Mat& histogramBins,
                                               int gapRange, vector<unsigned int>& keepBinIndexOfEle)
{

    // Get the bin values
    const unsigned int totalPixels = yuv420sp.rows;

    for (int index = 0; index < totalPixels; index++)
    {
        int getVal = (int)yuv420sp.at<float>(index,0);
        int binIndex = ceil(getVal / gapRange);

        // To have a small verification
        int yt;
        for (yt = (binIndex-2); yt <= (binIndex + 2); yt++){
            int getBinLimitCurrent = edgesBin[yt];
            int getBinLimitPrev = edgesBin[yt-1];
            if (( getBinLimitPrev < getVal) && (getVal <= getBinLimitCurrent)) {
                histogramBins.at<uchar>(yt,0) ++;
                break;
            }
        }
        keepBinIndexOfEle[index] = yt;
    }
}


void Fuzzy_C_Means_FGfsm_S2::compareTwoMatrixSimilarity(Mat& src1, Mat& src2, vector<int>& keepIndexes){
    // These two matrices are row matrix, having exactly the same number of columns
    keepIndexes.reserve(1);
    if(src1.cols != src2.cols)
        assert("The dimension of these 2 matrices does not match");
    for (int jj = 0; jj < src1.cols; jj++){
        if(src1.at<float>(0,jj) == src2.at<float>(0,jj))
            keepIndexes.push_back(jj);
    }
}

void Fuzzy_C_Means_FGfsm_S2:: matlab_reshape(const Mat &m, Mat& result, int new_row, int new_col, int new_ch)
{
    int old_row, old_col, old_ch;
    old_row = m.size().height;
    old_col = m.size().width;
    old_ch = m.channels();

    Mat m1 ( 1, new_row*new_col*new_ch, m.depth() );

    vector <Mat> p(old_ch);
    split(m,p);
    for ( int i=0; i<p.size(); ++i ){
        Mat t ( p[i].size().height, p[i].size().width, m1.type() );
        t = p[i].t();
        Mat aux = m1.colRange(i*old_row*old_col,(i+1)*old_row*old_col).rowRange(0,1);
        t.reshape(0,1).copyTo(aux);
    }

    vector <Mat> r(new_ch);
    for ( int i=0; i<r.size(); ++i ){
        Mat aux = m1.colRange(i*new_row*new_col,(i+1)*new_row*new_col).rowRange(0,1);
        r[i] = aux.reshape(0, new_col);
        r[i] = r[i].t();
    }
    merge(r, result);
}














void Fuzzy_C_Means_FGfsm_S2::multiplyTwo2DVector(std::vector<std::vector<int> > &Mat1, std::vector<std::vector<int> > &Mat2,
                                                 std::vector<std::vector<int> > &result) {
    int mm = Mat1.size();
    int pp = Mat2[0].size();
    int nn = Mat2.size();


    if (Mat1[0].size() != Mat2.size())
        assert("The matrices does not match in dimension");

    for (int ii = 0; ii < mm; ++ii) {
        for (int jj = 0; jj < pp; ++jj) {
            result[ii][jj] = 0;
            for (int kk = 0; kk < nn; ++kk)
                result[ii][jj] = result[ii][jj] + Mat1[ii][kk] * Mat2[kk][jj];
        }
    }
}


void Fuzzy_C_Means_FGfsm_S2::multiplyTwoFloatMat(Mat &Mat1, Mat &Mat2, Mat &result) {
    int mm = Mat1.rows;
    int pp = Mat2.cols;
    int nn = Mat2.rows;
    result = Mat::zeros(mm, pp, CV_32FC1);

    if (Mat1.cols != Mat2.rows)
        assert("The matrices does not match in dimension");

    for (int ii = 0; ii < mm; ++ii) {
        for (int jj = 0; jj < pp; ++jj) {
            result.at<float>(ii, jj) = 0;
            for (int kk = 0; kk < nn; ++kk)

                result.at<float>(ii, jj) = result.at<float>(ii, jj) + Mat1.at<float>(ii, kk) * Mat2.at<float>(kk, jj);
        }
    }
}


void Fuzzy_C_Means_FGfsm_S2::multiplyOpenCVMat2(Mat &Mat1, Mat &Mat2, Mat &result) {
    int mm = Mat1.rows;
    int pp = Mat2.cols;
    int nn = Mat2.rows;
    result = Mat::zeros(mm, pp, CV_32FC1);

    if (Mat1.cols != Mat2.rows)
        assert("The matrices does not match in dimension");

    for (int ii = 0; ii < mm; ++ii) {
        for (int jj = 0; jj < pp; ++jj) {
            result.at<float>(ii, jj) = 0;
            for (int kk = 0; kk < nn; ++kk)

                result.at<float>(ii, jj) = result.at<float>(ii, jj) + Mat1.at<uchar>(ii, kk) * Mat2.at<float>(kk, jj);
        }
    }
}

void Fuzzy_C_Means_FGfsm_S2::multiply_Float_Uchar_Mat(Mat &Mat1, Mat &Mat2, Mat &result) {
    int mm = Mat1.rows;
    int pp = Mat2.cols;
    int nn = Mat2.rows;
    result = Mat::zeros(mm, pp, CV_32FC1);

    if (Mat1.cols != Mat2.rows)
        assert("The matrices does not match in dimension");

    for (int ii = 0; ii < mm; ++ii) {
        for (int jj = 0; jj < pp; ++jj) {
            result.at<float>(ii, jj) = 0;
            for (int kk = 0; kk < nn; ++kk)

                result.at<float>(ii, jj) = result.at<float>(ii, jj) + Mat1.at<float>(ii, kk) * Mat2.at<uchar>(kk, jj);
        }
    }
}

void Fuzzy_C_Means_FGfsm_S2::multiply_2_UChar_Mat(Mat &Mat1, Mat &Mat2, Mat &result) {
    int mm = Mat1.rows;
    int pp = Mat2.cols;
    int nn = Mat2.rows;
    result = Mat::zeros(mm, pp, CV_8U);

    if (Mat1.cols != Mat2.rows)
        assert("The matrices does not match in dimension");

    for (int ii = 0; ii < mm; ++ii) {
        for (int jj = 0; jj < pp; ++jj) {
            result.at<uchar>(ii, jj) = 0;
            for (int kk = 0; kk < nn; ++kk)

                result.at<uchar>(ii, jj) = result.at<uchar>(ii, jj) + Mat1.at<uchar>(ii, kk) * Mat2.at<uchar>(kk, jj);
        }
    }
}
