//
// Created by tmondal on 24/08/2018.
//

#include "PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/hdr/Fuzzy_C_Means_S2.h"

Fuzzy_C_Means_S2 *Fuzzy_C_Means_S2::instance = 0;

Fuzzy_C_Means_S2 *Fuzzy_C_Means_S2::getInstance() {
    if (!instance)
        instance = new Fuzzy_C_Means_S2();

    return instance;
}

Fuzzy_C_Means_S2::Fuzzy_C_Means_S2() {
}

Fuzzy_C_Means_S2::~Fuzzy_C_Means_S2() {
}

void Fuzzy_C_Means_S2::padMatrix(Mat &m, int n, Mat &out) {
    Mat oldOut = m; // For the first time to initialize out matrix
    for (int ii = 0; ii < n; ii++) {
        out = Mat::zeros((oldOut.rows + 2), (oldOut.cols + 2), m.type());
        padMatrix_1(oldOut, out);
        oldOut = out;
    }
}

void Fuzzy_C_Means_S2::printFloatMatrix(cv::Mat imgMat) {
    for (int i = 0; i < imgMat.rows; i++) {
        for (int j = 0; j < imgMat.cols; j++) {

            cout << (float) imgMat.at<float>(i, j) << " ";
        }
        cout << endl;
    }
}

void Fuzzy_C_Means_S2::padMatrix_1(Mat &m, Mat &outModified) {

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


void Fuzzy_C_Means_S2::udDist_FCM_S2(Mat &center, Mat &data, Mat &dm1, float a, Mat &out) {

    /* DISTFCM Distance measure in fuzzy c-mean clustering. OUT = DISTFCM(CENTER, DATA) calculates the Euclidean
     * distance between each row in CENTER and each row in DATA, and returns a distance matrix OUT of size M by N,
     * where M and N are row dimensions of CENTER and DATA, respectively, and OUT(I, J) is the distance
     * between CENTER(I,:) and DATA(J,:) */


    for (int ii = 0; ii < center.rows; ii++) {
        Mat val1 = abs(center.at<float>(ii, 0) - data);
        pow(val1, 2, val1);

        Mat val2 = abs(center.at<float>(ii, 0) - dm1);
        pow(val2, 2, val2);
        val2 = a*val2;
        Mat val3 = val1 + val2;
        val3 = val3.t();
        out.push_back(val3);
    }
}

void Fuzzy_C_Means_S2::ud_FCM_S2(Mat &origFloatImg, int cluster_n, float a, Mat &center, Mat &U, vector<float>& obj_fcn) {
    Mat data = origFloatImg.clone();
    Mat data1 = data;
    data.release();
    // data.setTo(0); // replacing all the values by 0
     Mat data_1_Transpose = data1.t();
    data = data_1_Transpose.reshape(1, (data_1_Transpose.rows * data_1_Transpose.cols));
    data_1_Transpose.release();
    Mat data2;
    padMatrix(data1, 1, data2);

    Mat dataMean = Mat::zeros(data1.rows, data1.cols, data1.type());
    for (int i = 1; i < (data1.rows + 1); i++) {
        for (int j = 1; j < (data1.cols + 1); j++) {
            float tempVal;
            dm(i, j, data2, tempVal);
            dataMean.at<float>(i - 1, j - 1) = tempVal;
        }
    }
    int data_n = data.rows;
    float default_options[4] = {2, 100, 1e-4, 1};

    float expo = default_options[0];         // Exponent for U
    float max_iter = default_options[1];    // Max. iteration
    float min_impro = default_options[2]; // Min. improvement
    float display = default_options[3]; // Display info or not


    U = Mat::zeros(cluster_n, data_n, CV_32FC1);
    //obj_fcn = Mat::zeros(max_iter, 1, CV_32FC1);    // Array for objective function
    obj_fcn.reserve(max_iter);
    udInit_FCM_S2(cluster_n, data_n, U);     // Initial fuzzy partition
    int ii = 0;

    for (ii = 0; ii < max_iter; ii++) {
        float tempObjectiveFuncVal;
        Mat U_New;
        center = Mat::zeros(cluster_n, 1, CV_8U);

        udStepFCM_S2(data, dataMean, U, cluster_n, a, expo, center, tempObjectiveFuncVal);
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

void Fuzzy_C_Means_S2::udStepFCM_S2(Mat &data, Mat &dataMean, Mat &U, int &clusterN, double a, int expo, Mat &center,
                                    float& obj_func) {

    // data  : It is a float matrix
    // dataMean  : It is also a float matrix
    /*
    STEPFCM One step in fuzzy c-mean clustering. [U_NEW, CENTER, ERR] = STEPFCM(DATA, U, CLUSTER_N, EXPO)
    performs one iteration of fuzzy c-mean clustering, where DATA: matrix of data to be clustered.
    (Each row is a data point.)U: partition matrix. (U(i,j) is the MF value of data j in cluster j.)
    CLUSTER_N: number of clusters.
            EXPO: exponent (> 1) for the partition matrix.
            U_NEW: new partition matrix.
            CENTER: center of clusters. (Each row is a center.)
    ERR: objective function for partition U.
    Note that the situation of "singularity" (one of the data points is exactly the same as one of the
    cluster centers) is not checked. However, it hardly occurs in practice. */
    Mat mf;
    cv::pow(U, expo, mf);
    Mat dataMeanTrans = dataMean.t();
    Mat dm1 = dataMeanTrans.reshape(1, (dataMeanTrans.rows * dataMeanTrans.cols));
    dataMeanTrans.release();

    Mat data1;
    cv::multiply(dm1, a, data1);
    cv::add(data, data1, data1);

    Mat mfTranspose = mf.t();
    Mat columnSum = Mat::zeros(1, mfTranspose.cols, CV_32FC1);
    for (int iyt = 0; iyt < mfTranspose.cols; iyt++)
        columnSum.at<float>(0, iyt) = (cv::sum(mfTranspose.col(iyt))[0]);

    Mat numera;
    multiplyTwoFloatMat(mf, data1, numera); // this is 2 float matrix
    // gemm(mf, data1, 1, noArray(), 0, numera);


    Mat denomina;
    Mat denomina_2;
    Mat oneMatrix = Mat::ones(data1.cols, 1, CV_8UC1);
    multiplyOpenCVMat2(oneMatrix, columnSum, denomina_2);
    // gemm((Mat::ones(data1.cols,1,CV_8UC1)), columnSum, 1, noArray(), 0, denomina_2);

    cv::multiply((denomina_2.t()), (1 + a), denomina);
    cv::divide(numera, denomina, center);

    Mat t1;
    Mat t2 = Mat::zeros(center.rows, data.rows, CV_32FC1);

    obj_mat(center, data, dataMean, U, expo, t1, t2);
    pow(t1, 2.0, t1); // squaring t1 matrix and saving it into t1 matrix again
/*    float sum_1 = 0.0;
    float sum_2 = 0.0;
    for (int ii = 0; ii < t1.rows; ii++) {
        for (int jj = 0; jj < t2.cols; jj++) {
            sum_1 = sum_1 + pow((t1.at<uchar>(ii, jj)), 2) * mf.at<uchar>(ii, jj);
            sum_2 = sum_2 + a * t2.at<uchar>(ii, jj);
        }
    }*/

    // obj_func = sum_1 + sum_2;
    obj_func = (cv::sum(t1.mul(mf))[0]) + (a * cv::sum(t2)[0]);
    Mat dist;
    udDist_FCM_S2(center, data, dm1, a, dist);

    Mat tmpLac;
    cv::pow(dist, (-1 / (expo - 1)), tmpLac);

    Mat makeOnes = Mat::ones(clusterN, 1, CV_8UC1);

    Mat tempColSum = Mat::zeros(1,tmpLac.cols, CV_32FC1);
    for (int iCol = 0; iCol<tmpLac.cols; iCol++) {
        float sumCol;
        // for (int iRw = 0; iRw < tmpLac.rows; iRw++) {
        //     sumCol = sumCol + tmpLac.at<float>(iRw, iCol);
        sumCol = cv::sum(tmpLac.col(iCol))[0]; // actually I am doing here sum of each column but
        // }
        tempColSum.at<float>(0,iCol) = sumCol;
    }
    //  printFloatMatrix(tempColSum);
    //  tempColSum = tempColSum.t(); // as I am putting it in Mat as vector push_up style, so it is put in row order

    Mat multiplyRes;
    multiplyOpenCVMat2(makeOnes,tempColSum, multiplyRes);

    // Mat divResult = tmp / multiplyRes
    Mat U_New;
    divide(tmpLac, multiplyRes, U_New);

    U.release();
    U = U_New.clone();
    U_New.release();
}

void Fuzzy_C_Means_S2::udInit_FCM_S2(int cluster_n, int data_n, Mat &U) {
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

void Fuzzy_C_Means_S2::obj_mat(Mat &center, Mat &data, Mat &dm1, Mat U, int expo, Mat &out1, Mat &out2) {
    Mat dm_1_Trans = dm1.t();
    Mat dm2 = dm_1_Trans.reshape(1, (dm_1_Trans.rows * dm_1_Trans.cols));
    dm_1_Trans.release();

    for (int kk = 0; kk < center.rows; kk++) {
        Mat tempVector = abs((center.at<float>(kk, 0) - data).t());
        out1.push_back(tempVector);
    }
    for (int jj = 0; jj < center.rows; jj++) {
        for (int ii = 0; ii < data.rows; ii++) {
            std::vector<std::vector<int> > neigh;
            neighborCalculation(dm1, neigh, ii);
            float sumMe = 0.0;
            for (int uu = 0; uu < neigh.size(); uu++) {
                int getNeighIndex = neigh[uu][0];
                float part_1 = pow((U.at<float>(jj, getNeighIndex)), expo);
                float part_2 = pow((dm2.at<float>(getNeighIndex, 0) - center.at<float>(jj, 0)), 2);
                sumMe = sumMe + (part_1 * part_2);
            }
            out2.at<float>(jj, ii) = sumMe;
            // BasicAlgo::getInstance()->printMatrix(U);
        }
    }
}

void Fuzzy_C_Means_S2::neighborCalculation(Mat &dm1, std::vector<std::vector<int>> &result, int ii1) {
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

void Fuzzy_C_Means_S2::multiplyMat(std::vector<std::vector<int> > &Mat1, std::vector<std::vector<int> > &Mat2,
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

void Fuzzy_C_Means_S2::multiplyTwoFloatMat(Mat &Mat1, Mat &Mat2, Mat &result) {
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

void Fuzzy_C_Means_S2::multiplyOpenCVMat2(Mat &Mat1, Mat &Mat2, Mat &result) {
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


void Fuzzy_C_Means_S2::applyAlgoOnImage(Mat &imgOrig, Mat &matOut) {
    if (imgOrig.channels() >= 3)
        cv::cvtColor(imgOrig, imgOrig, CV_BGR2GRAY);

    Mat center;
    Mat U;
    vector<float> obj_fcn;

    Mat imgOrigFloat;
    imgOrig.convertTo(imgOrigFloat, CV_32FC1);
    imgOrigFloat = imgOrigFloat / 255.0;
    // printFloatMatrix(imgOrigFloat);
    ud_FCM_S2(imgOrigFloat, 4, 4.2, center, U, obj_fcn);


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

    Mat normImag = d6*255 ;
    normImag.convertTo(normImag, CV_8U);
    d6.release();
    matOut = normImag.clone();
    normImag.release();
    // BasicAlgo::getInstance()->showImage(normImag);
}

void Fuzzy_C_Means_S2::compareTwoMatrixSimilarity(Mat& src1, Mat& src2, vector<int>& keepIndexes){
    // These two matrices are row matrix, having exactly the same number of columns
    keepIndexes.reserve(1);
    if(src1.cols != src2.cols)
        assert("The dimension of these 2 matrices does not match");
    for (int jj = 0; jj < src1.cols; jj++){
        if(src1.at<float>(0,jj) == src2.at<float>(0,jj))
            keepIndexes.push_back(jj);
    }
}

void Fuzzy_C_Means_S2:: matlab_reshape(const Mat &m, Mat& result, int new_row, int new_col, int new_ch)
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
