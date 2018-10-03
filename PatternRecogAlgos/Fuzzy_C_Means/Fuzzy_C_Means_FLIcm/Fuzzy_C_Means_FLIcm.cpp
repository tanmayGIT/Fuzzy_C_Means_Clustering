//
// Created by tmondal on 21/09/2018.
//
#include <unitypes.h>
#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FLIcm/hdr/Fuzzy_C_Means_FLIcm.h"

Fuzzy_C_Means_FLIcm *Fuzzy_C_Means_FLIcm::instance = 0;

Fuzzy_C_Means_FLIcm *Fuzzy_C_Means_FLIcm::getInstance() {
    if (!instance)
        instance = new Fuzzy_C_Means_FLIcm();

    return instance;
}

Fuzzy_C_Means_FLIcm::Fuzzy_C_Means_FLIcm() {
}

Fuzzy_C_Means_FLIcm::~Fuzzy_C_Means_FLIcm() {
}



void Fuzzy_C_Means_FLIcm::padMatrix(Mat &m, int n, Mat &out) {
    Mat oldOut = m; // For the first time to initialize out matrix
    for (int ii = 0; ii < n; ii++) {
        out = Mat::zeros((oldOut.rows + 2), (oldOut.cols + 2), m.type());
        padMatrix_1(oldOut, out);
        oldOut = out;
    }
}

void Fuzzy_C_Means_FLIcm::padMatrix_1(Mat &m, Mat &outModified) {

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


void Fuzzy_C_Means_FLIcm::udStepFLICM(Mat &data, Mat & rc, Mat &U, int &clusterN, int expo, Mat& center,
                                    float& obj_func) {
    Mat mf;
    cv::pow(U, expo, mf);

    Mat mfTranspose = mf.t();
    Mat mfTransSum = Mat::zeros(1, mfTranspose.cols, CV_32F);
    for (int iyt = 0; iyt < mfTranspose.cols; iyt++)
        mfTransSum.at<float>(0, iyt) = (cv::sum(mfTranspose.col(iyt))[0]);
    mfTranspose.release();

    Mat makeOnesDenominator = Mat::ones(data.cols, 1, CV_8U);

    Mat calcDeno;
    multiplyOpenCVMat2(makeOnesDenominator,mfTransSum, calcDeno );

    mfTransSum.release();
    makeOnesDenominator.release();
    calcDeno = calcDeno.t();

    Mat numera;
    multiplyTwoFloatMat(mf, data, numera); // this is 2 float matrix
    center = numera / calcDeno;


    Mat dist1, dist2;
    udDist_FCM(center, data, rc, U, expo, dist1, dist2);
    // cout << "The dist1 matrix" << endl;
    // cout << dist1 << endl;

    // cout << "The dist2 matrix" << endl;
    // cout << dist2 << endl;

    Mat dist_2_ColSum = Mat::zeros(1,dist2.cols, CV_32F);
    float sumCol ;
    for (int distCol = 0; distCol<dist2.cols; distCol++) {
        sumCol = cv::sum(dist2.col(distCol))[0]; // actually I am doing here sum of each column but
        dist_2_ColSum.at<float>(0,distCol) = sumCol;
    }
    float dist_2_totalSum = cv::sum(dist_2_ColSum)[0];

    Mat powDist_1;
    cv::pow(dist1, 2, powDist_1);
    // cout << "The powDist_1 matrix :" << endl;
    // cout << powDist_1 << endl;

    Mat powMul = powDist_1.mul(mf);
    // cout << "The powMul matrix :" << endl;
    // cout << powMul << endl;

    Mat powMul_ColSum = Mat::zeros(1,powMul.cols, CV_32F);
    float sumColPowMul ;
    for (int distCol = 0; distCol < powMul.cols; distCol++) {
        sumColPowMul = cv::sum(powMul.col(distCol))[0];
        powMul_ColSum.at<float>(0,distCol) = sumColPowMul;
    }
    // cout << "The powMul_ColSum matrix :" << endl;
    // cout << powMul_ColSum << endl;

    float powMul_totalSum = cv::sum(powMul_ColSum)[0];
    powMul_ColSum.release();
    obj_func = powMul_totalSum + dist_2_totalSum;

    Mat dist = powDist_1 + dist2;
    // cout << dist << endl;
    Mat tmpLac;
    cv::pow(dist, (-1 / (expo - 1)), tmpLac);
    // cout << tmpLac << endl;

    Mat makeOnes = Mat::ones(clusterN, 1, CV_8U);

    Mat tempColSum = Mat::zeros(1,tmpLac.cols, CV_32F);
    for (int iCol = 0; iCol<tmpLac.cols; iCol++) {
        float sumCol;
        sumCol = cv::sum(tmpLac.col(iCol))[0]; // actually I am doing here sum of each column but
        tempColSum.at<float>(0,iCol) = sumCol;
    }

    Mat multiplyRes;
    multiplyOpenCVMat2(makeOnes,tempColSum, multiplyRes);
    // cout << multiplyRes << endl;

    Mat U_New;
    divide(tmpLac, multiplyRes, U_New);
   // cout << U_New << endl;

    U.release();
    U = U_New.clone();
    U_New.release();
}



void Fuzzy_C_Means_FLIcm::udDist_FCM(Mat &center, Mat &data, Mat &rc, Mat &U, int expo, Mat& out1, Mat& out2) {

    out1 = Mat::zeros(center.rows, data.rows, CV_32F);
    out2 = Mat::zeros(center.rows, data.rows, CV_32F);

    // Fill the output matrix
    if (data.type() != center.type())
        data.convertTo(data, center.type());
    for (int iFill = 0; iFill < center.rows; iFill ++) {
        float centerVal = center.at<float>(iFill,0);
        Mat obtainedRes =  abs((centerVal - data).t());
        obtainedRes.copyTo(out1.row(iFill));
    }
    // cout << out1 << endl;
    for (int kk = 0; kk < center.rows; kk++){
        for (int ii = 0; ii < data.rows; ii++){

            float* getSmallArr = neighborCalculation(rc, ii);
            Mat ct  = cv::Mat(1, 2, CV_32F, *getSmallArr);
            Mat neigh1 =  attachAllNeigh.out1;
            Mat neigh2 = attachAllNeigh.out2;

            // cout << ct << endl;
            // cout << neigh1 << endl;
            // cout << neigh2 << endl;

            Mat makeOnes = Mat::ones(neigh1.rows, 1, CV_8U);
            Mat mulFirst;
            multiplyOpenCVMat2(makeOnes, ct, mulFirst);
         //   cout << mulFirst << endl;

            if(mulFirst.type() != neigh1.type())
                mulFirst.convertTo(mulFirst, neigh1.type());

        //    cout << neigh1 << endl;
        //    cout << mulFirst << endl;
            neigh1 = neigh1 - mulFirst;
        //    cout << neigh1 << endl;

            makeOnes.release();
            mulFirst.release();

            Mat powMe;
            cv::pow(neigh1, 2, powMe);
            powMe = powMe.t();
        //    cout << powMe << endl;


            Mat tempColSum = Mat::zeros(1,powMe.cols, CV_32FC1);
            for (int iCol = 0; iCol<powMe.cols; iCol++) {
                float sumCol;
                sumCol = cv::sum(powMe.col(iCol))[0]; // actually I am doing here sum of each column but
                tempColSum.at<float>(0,iCol) = sumCol;
            }
            powMe.release();
         //   cout << tempColSum << endl;
            cv::sqrt(tempColSum, tempColSum);
         //   cout << tempColSum << endl;
            tempColSum = tempColSum +1;
        //    cout << tempColSum << endl;

            Mat d;
            cv::pow(tempColSum, (-1), d);
         //   cout << d << endl;
            tempColSum.release();

            Mat u = Mat::zeros(1,neigh2.rows, CV_32F );
            Mat xv = Mat::zeros(neigh2.rows, 1, CV_32F );

            for (int iNeig = 0; iNeig < neigh2.rows; iNeig++){
                int getIndex = (int) neigh2.at<float>(iNeig, 0);

/*                float val11 = U.at<float>(kk, getIndex);
                float val12 = data.at<float>(getIndex, 0);
                float val13 =  center.at<float>(kk,0);*/
                u.at<float>(0, iNeig) = pow( (1 - U.at<float>(kk, getIndex)), expo);
                xv.at<float>(iNeig,0) = pow(   (data.at<float>(getIndex, 0) - center.at<float>(kk,0)), 2);
            }
        //    cout << u << endl;
        //    cout << xv   << endl;


        //    cout << d << endl;
        //    cout << u << endl;
        //    cout << xv << endl;
            Mat mulMeNow =  d.mul(u) * xv;
        //    cout << mulMeNow   << endl;

            out2.at<float>(kk,ii) = mulMeNow.at<float>(0,0);

            mulMeNow.release();

            neigh1.release();
            neigh2.release();
            ct.release();
        }
    }
   // cout << out2   << endl;
}


void Fuzzy_C_Means_FLIcm::udInit_FCM(int cluster_n, int data_n, Mat &U) {
    /* INITFCM Generate initial fuzzy partition matrix for fuzzy c-means clustering. U = INITFCM(CLUSTER_N, DATA_N)
     * randomly generates a fuzzy partition matrix U that is CLUSTER_N by DATA_N, where CLUSTER_N is number of
     * clusters and DATA_N is number of data points. The summation of each column of the generated U is equal to unity,
     * as required by fuzzy c-means clustering. */
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
            // cout << "The value of U : " << U.at<float>(ii,jj) << endl;
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


float* Fuzzy_C_Means_FLIcm::neighborCalculation(const Mat rc, int ii1)
{
    int r = rc.at<float>(0,0);
    int c = rc.at<float>(0,1);

    int r1 = ii1 % r; // The rule of mod i.e. remainder after division is always equal to ii1 while i < r but when i==r, r1=0
    int c1 = 0;
    if ((r1 == 0) && (ii1 > 0)) { // that means r1 == 0 but ii1 > 0
        r1 = r;
        c1 = floor(ii1 / r);
    } else {
        c1 = floor(ii1 / r);
    }
    std::vector<std::vector<int> > temp{{-1, 1},
                                        {-1, 0},
                                        {1,  -1},
                                        {1,  0},
                                        {0,  1},
                                        {0,  -1},
                                        {1,  1},
                                        {-1, -1}};
    std::vector<int> nr;
    std::vector<int> nc;
    std::vector<std::vector<int> > refinedTemp;
    // if the coming indexes are border pixels
    for (int ii = 0;
         ii < temp.size(); ii++) { // If they are border pixels then after these following 2 lines of operation w
        temp[ii][0] = temp[ii][0] + r1; // would make some indexes to pass the boundary
        temp[ii][1] = temp[ii][1] + c1;
    }
    for (int ii = 0; ii < temp.size(); ii++) {
        if (((r1 == 0) || (c1 == 0)) || ((r1 == (r - 1)) || (c1 == (c - 1)))) {
            if ((temp[ii][0] == -1) || (temp[ii][0] == r)) // check whether we are passing the boundary or not
                nr.push_back(ii);  // keeping the index of "temp" where it is touching the boundary
            if ((temp[ii][1] == -1) || (temp[ii][1] == c)) // check whether we are passing the boundary
                nc.push_back(ii); // keeping the index of "temp" where it is touching the boundary
        }
    }

    std::vector<int> unionSet;
    std::set_union(nr.begin(), nr.end(), nc.begin(), nc.end(),
                   std::back_inserter(unionSet)); //doing union to get the indexes which are problematic
    // Remove all of them
    for (int gt = 0; gt < temp.size(); gt++) {
        int rc = 0;
        bool flagPresent = false;
        for (int gg = 0; gg < unionSet.size(); gg++) {
            rc = unionSet.at(gg);
            if (rc == gt) {
                flagPresent = true;
                break;
            }
        }
        if (!flagPresent)
            refinedTemp.push_back(temp.at(gt));
    }

    Mat tempForCal = Mat::zeros(refinedTemp.size(),refinedTemp[0].size(),CV_32F);
    for (int iTemp = 0; iTemp < tempForCal.rows; iTemp++){
        for (int jTemp = 0; jTemp < tempForCal.cols; jTemp++){
            tempForCal.at<float>(iTemp, jTemp) = refinedTemp[iTemp][jTemp];
        }
    }
    // cout << tempForCal << endl;

    attachAllNeigh.out1 = tempForCal;
    Mat mulMe;
    float tempOut23[2] = {(float)1, (float)r};
    mulMe = cv::Mat(2, 1, CV_32F, tempOut23);
    // cout << mulMe << endl;

    if(mulMe.type() != tempForCal.type())
        mulMe.convertTo(mulMe, tempForCal.type());
    multiplyTwoFloatMat(tempForCal, mulMe, attachAllNeigh.out2);

    // cout << attachAllNeigh.out2 << endl;
    tempForCal.release();
    mulMe.release();

    float tempOut3[2] = {(float)r1, (float)c1};
    //Mat tempOntop  = cv::Mat(1, 2, CV_32F, tempOut3);
    return tempOut3;
}


void Fuzzy_C_Means_FLIcm::fLICM(Mat &origFloatImg, int cluster_n, Mat &center, Mat &U, vector<float>& obj_fcn) {
    Mat data = origFloatImg.clone();
    Mat data1 = data;
    // cout << "The data1" << endl;
    // cout << data1 << endl;

    data.release();
    // data.setTo(0); // replacing all the values by 0
    Mat justTranstemp = (data1.t());
    data = justTranstemp.reshape(1, (data1.rows * data1.cols));
    justTranstemp.release();
    // cout << "The data" << endl;
    // cout << data << endl;

    Mat data2;
    padMatrix(data1, 1, data2);
    // cout << "The data2" << endl;
    // cout << data2 << endl;

    int data_n = data.rows;
    float default_options[4] = {2, 100, 1e-5, 1};

    float expo = default_options[0];         // Exponent for U
    float max_iter = default_options[1];    // Max. iteration
    float min_impro = default_options[2]; // Min. improvement
    float display = default_options[3]; // Display info or not


    U = Mat::zeros(cluster_n, data_n, CV_32FC1);
    //obj_fcn = Mat::zeros(max_iter, 1, CV_32FC1);    // Array for objective function
    obj_fcn.reserve(max_iter);
    udInit_FCM(cluster_n, data_n, U);     // Initial fuzzy partition
    int ii = 0;

    float tempOut[2] = {(float)(origFloatImg.rows), (float)(origFloatImg.cols)};
    Mat rc = cv::Mat(1, 2, CV_32F, tempOut);
    // cout << rc << endl;


/*    cout << "The matrix data" << endl;
    cout << data << endl;

    cout << "The matrix rc " << endl;
    cout << rc << endl;

    cout << "The matrix U" << endl;
    cout << U << endl;*/


    for (ii = 0; ii < max_iter; ii++) {
        float tempObjectiveFuncVal;
        Mat U_New;
        center = Mat::zeros(cluster_n, 1, CV_8U);

        udStepFLICM(data, rc, U, cluster_n, expo, center, tempObjectiveFuncVal);
        // printFloatMatrix(U_New);
        obj_fcn[ii] = (tempObjectiveFuncVal);

        if (display)
            cout << "Iteration count = " << ii << " Objective Function = " << tempObjectiveFuncVal << endl;
        if (ii > 0)
            if ((abs(obj_fcn[ii] - obj_fcn[ii -1])) < min_impro)
                break;
        // U_New = U; // keeping the modified U values in an new Mat U_New so that we can use it later
        // U = Mat::zeros(cluster_n, data_n, CV_32FC1); // reinitialize U
    }
    int actualNumberIter = ii;      // Actual number of iterations

    std::vector<float> newVector;
    vector<float>::const_iterator firstVect = obj_fcn.begin();
    vector<float>::const_iterator lastVect = obj_fcn.begin() + actualNumberIter;
    std::copy(firstVect, lastVect, std::back_inserter(newVector));
    obj_fcn = newVector;
    newVector.clear();
}


void Fuzzy_C_Means_FLIcm::applyAlgoOnImage(Mat &imgOrig, Mat &matOut) {
    if (imgOrig.channels() >= 3)
        cv::cvtColor(imgOrig, imgOrig, CV_BGR2GRAY);

    Mat center;
    Mat U;
    vector<float> obj_fcn;

    Mat imgOrigFloat;
    imgOrig.convertTo(imgOrigFloat, CV_32FC1);
    imgOrigFloat = imgOrigFloat / 255.0;
    // printFloatMatrix(imgOrigFloat);
    int cluster_n = 4;
    vector<float> options;
    fLICM(imgOrigFloat, cluster_n, center, U, obj_fcn);


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

    center = (center * 255);
    Mat rr = center;
    Mat imgOrigFloatTrans = imgOrigFloat.t();
    int lenTak = (imgOrigFloatTrans.rows * imgOrigFloatTrans.cols);
    Mat d4 = imgOrigFloatTrans.reshape(1,lenTak);
    imgOrigFloatTrans.release();
    Mat d5 = d4;
    // d4.release();

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


void Fuzzy_C_Means_FLIcm::compareTwoMatrixSimilarity(Mat& src1, Mat& src2, vector<int>& keepIndexes){
    // These two matrices are row matrix, having exactly the same number of columns
    keepIndexes.reserve(1);
    if(src1.cols != src2.cols)
        assert("The dimension of these 2 matrices does not match");
    for (int jj = 0; jj < src1.cols; jj++){
        if(src1.at<float>(0,jj) == src2.at<float>(0,jj))
            keepIndexes.push_back(jj);
    }
}

void Fuzzy_C_Means_FLIcm:: matlab_reshape(const Mat &m, Mat& result, int new_row, int new_col, int new_ch)
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


void Fuzzy_C_Means_FLIcm::multiplyTwoFloatMat(Mat &Mat1, Mat &Mat2, Mat &result) {
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


void Fuzzy_C_Means_FLIcm::multiply_2_UChar_Mat(Mat &Mat1, Mat &Mat2, Mat &result) {
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



void Fuzzy_C_Means_FLIcm::multiplyOpenCVMat2(Mat &Mat1, Mat &Mat2, Mat &result) {
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

void Fuzzy_C_Means_FLIcm::multiplyMat(std::vector<std::vector<int> > &Mat1, std::vector<std::vector<int> > &Mat2,
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