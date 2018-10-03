//
// Created by tmondal on 24/08/2018.
//

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_M1/hdr/Fuzzy_C_Means_M1.h"

Fuzzy_C_Means_M1 *Fuzzy_C_Means_M1::instance = 0;

Fuzzy_C_Means_M1 *Fuzzy_C_Means_M1::getInstance() {
    if (!instance)
        instance = new Fuzzy_C_Means_M1();

    return instance;
}

Fuzzy_C_Means_M1::Fuzzy_C_Means_M1() {
}

Fuzzy_C_Means_M1::~Fuzzy_C_Means_M1() {
}

void Fuzzy_C_Means_M1::padMatrix(Mat &m, int n, Mat &out) {
    Mat oldOut = m; // For the first time to initialize out matrix
    for (int ii = 0; ii < n; ii++) {
        out = Mat::zeros((oldOut.rows + 2), (oldOut.cols + 2), m.type());
        padMatrix_1(oldOut, out);
        oldOut = out;
    }
}

void Fuzzy_C_Means_M1::printFloatMatrix(cv::Mat imgMat) {
    for (int i = 0; i < imgMat.rows; i++) {
        for (int j = 0; j < imgMat.cols; j++) {

            cout << (float) imgMat.at<float>(i, j) << " ";
        }
        cout << endl;
    }
}

void Fuzzy_C_Means_M1::padMatrix_1(Mat &m, Mat &outModified) {

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


void Fuzzy_C_Means_M1::udDist_FCM_M1(Mat &center, Mat &data, Mat &out) {
    out = Mat::zeros(center.rows, data.rows, CV_32F);
    if (center.cols > 1) {
        for (int ii = 0; ii < center.rows; ii++) {
            Mat oneMatrix = Mat::ones(data.rows, 1, CV_32F);
            Mat calcVal = (data - oneMatrix);
            Mat multipliedVals = calcVal * center.row(ii);
            calcVal.release();
            calcVal.release();
            cv::pow(multipliedVals, 2, multipliedVals);
            multipliedVals = multipliedVals.t();
            double obtainedRes = sqrt(cv::sum(multipliedVals.row(ii))[0]);
            out.row(ii).setTo(obtainedRes);
        }
    } else{
        for (int ii = 0; ii < center.rows; ii++) {
            float tempCenter = center.at<float>(ii,0);
            Mat obtainedRes =  abs((tempCenter - data).t());
            obtainedRes.copyTo(out.row(ii));
        }
    }
}


void Fuzzy_C_Means_M1::ud_FCM_M1(Mat &origFloatImg, int cluster_n, Mat &center, Mat &U, vector<float>& obj_fcn) {
    Mat data = origFloatImg.clone();
    Mat data1 = data;
    data.release();
    // data.setTo(0); // replacing all the values by 0
    Mat data_1_Trans = data1.t();
    data = data_1_Trans.reshape(1, (data_1_Trans.rows * data_1_Trans.cols));
    data_1_Trans.release();

    int data_n = data.rows;
    float default_options[4] = {2, 100, 1e-5, 1};

    float expo = default_options[0];         // Exponent for U
    float max_iter = default_options[1];    // Max. iteration
    float min_impro = default_options[2]; // Min. improvement
    float display = default_options[3]; // Display info or not


    U = Mat::zeros(cluster_n, data_n, CV_32FC1);
    //obj_fcn = Mat::zeros(max_iter, 1, CV_32FC1);    // Array for objective function
    obj_fcn.reserve(max_iter);
    udInit_FCM_M1(cluster_n, data_n, U);     // Initial fuzzy partition
    int ii = 0;

    for (ii = 0; ii < max_iter; ii++) {
        float tempObjectiveFuncVal;
        Mat U_New;
        center = Mat::zeros(cluster_n, 1, CV_8U);

        udStepFCM_M1(data, U, cluster_n, expo, center, tempObjectiveFuncVal);
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


void Fuzzy_C_Means_M1::udStepFCM_M1(Mat &data, Mat &U, int &clusterN, int expo, Mat &center,
                                    float& obj_func) {
    Mat mf;
    cv::pow(U, expo, mf);

    Mat mfTranspose = mf.t();
    Mat columnSum = Mat::zeros(1, mfTranspose.cols, CV_32F);
    for (int iyt = 0; iyt < mfTranspose.cols; iyt++)
        columnSum.at<float>(0, iyt) = (cv::sum(mfTranspose.col(iyt))[0]);

    Mat numera;
    multiplyTwoFloatMat(mf, data, numera); // this is 2 float matrix

    Mat denomina_2;
    Mat oneMatrix = Mat::ones(data.cols, 1, CV_8U);
    multiplyOpenCVMat2(oneMatrix, columnSum, denomina_2);

    Mat denomina = (denomina_2.t());
    cv::divide(numera, denomina, center);

    Mat dist;
    udDist_FCM_M1(center, data, dist);
    Mat powDist;
    cv::pow(dist, 2, powDist);

    obj_func = (cv::sum(powDist.mul(mf))[0]);


    Mat tmpLac;
    cv::pow(dist, (-2 / (expo - 1)), tmpLac);

    Mat makeOnes = Mat::ones(clusterN, 1, CV_8U);
    Mat tempColSum = Mat::zeros(1,tmpLac.cols, CV_32F);
    for (int iCol = 0; iCol<tmpLac.cols; iCol++) {
        float sumCol;
        sumCol = (float)cv::sum(tmpLac.col(iCol))[0]; // actually I am doing here sum of each column but
        tempColSum.at<float>(0,iCol) = sumCol;
    }

    Mat multiplyRes;
    multiplyOpenCVMat2(makeOnes,tempColSum, multiplyRes);

    // Mat divResult = tmp / multiplyRes
    Mat U_New;
    divide(tmpLac, multiplyRes, U_New);

    U.release();
    U = U_New.clone();
    U_New.release();
}

void Fuzzy_C_Means_M1::udInit_FCM_M1(int cluster_n, int data_n, Mat &U) {
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

void Fuzzy_C_Means_M1::neighborCalculation(Mat &dm1, std::vector<std::vector<int>> &result, int ii1) {
    int r = dm1.rows;
    int c = dm1.cols;

    int r1 = ii1 %
             r; // The rule of mod i.e. remainder after division is always equal to ii1 while i < r but when i==r, r1=0
    int c1 = 0;
    if ((r1 == 0) && (ii1 > 0)) { // that means r1 == 0 but ii1 > 0
        r1 = dm1.rows;
        c1 = floor(ii1 / r);
    } else {
        c1 = floor(ii1 / r)+1;
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
    /*
    std::vector<std::vector<int> > copyTemp = refinedTemp;
    for (int ii = 0; ii < copyTemp.size(); ii++){
            copyTemp[ii][1] = refinedTemp[ii][1] - 1;
    }*/
    std::vector<std::vector<int> > mulMe{{0}, {r-1}};
    std::vector<std::vector<int> > out(refinedTemp.size(), std::vector<int>(mulMe[0].size(), 0));;
    multiplyMat(refinedTemp, mulMe, out);

    //gemm(copyTemp, mulMe, 1, noArray(), 0, out);
    result = out;
}

void Fuzzy_C_Means_M1::multiplyMat(std::vector<std::vector<int> > &Mat1, std::vector<std::vector<int> > &Mat2,
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

void Fuzzy_C_Means_M1::multiplyTwoFloatMat(Mat &Mat1, Mat &Mat2, Mat &result) {
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

void Fuzzy_C_Means_M1::multiplyOpenCVMat2(Mat &Mat1, Mat &Mat2, Mat &result) {
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


void Fuzzy_C_Means_M1::applyAlgoOnImage(Mat &imgOrig, Mat &matOut) {
    if (imgOrig.channels() >= 3)
        cv::cvtColor(imgOrig, imgOrig, CV_BGR2GRAY);

    Mat center;
    Mat U;
    vector<float> obj_fcn;

    Mat imgOrigFloat;
    imgOrig.convertTo(imgOrigFloat, CV_32F);
    //imgOrigFloat = imgOrigFloat / 255.0;
    // printFloatMatrix(imgOrigFloat);
    ud_FCM_M1(imgOrigFloat, 4, center, U, obj_fcn);

    double minVal = 0.0;
    double maxVal = 0.0;
    Point minLoc;
    Point maxLoc;
    Mat eachColMax = Mat::zeros(1,U.cols, CV_32F );
    for (int jCol = 0; jCol < U.cols; jCol++)
    {
        Mat eachCol = U.col(jCol);
        minMaxLoc(eachCol, &minVal, &maxVal, &minLoc, &maxLoc);
        eachColMax.at<float>(0, jCol) = (float) maxVal;
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
    normImag.release();
    // BasicAlgo::getInstance()->showImage(normImag);
}

void Fuzzy_C_Means_M1::compareTwoMatrixSimilarity(Mat& src1, Mat& src2, vector<int>& keepIndexes){
    // These two matrices are row matrix, having exactly the same number of columns
    keepIndexes.reserve(1);
    if(src1.cols != src2.cols)
        assert("The dimension of these 2 matrices does not match");
    for (int jj = 0; jj < src1.cols; jj++){
        if(src1.at<float>(0,jj) == src2.at<float>(0,jj))
            keepIndexes.push_back(jj);
    }
}

void Fuzzy_C_Means_M1:: matlab_reshape(const Mat &m, Mat& result, int new_row, int new_col, int new_ch)
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
