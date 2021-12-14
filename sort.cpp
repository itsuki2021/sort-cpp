#include "sort.h"

using namespace sort;

Sort::Sort(int maxAge, int minHits, float iouThresh)
    : maxAge(maxAge), minHits(minHits), iouThresh(iouThresh)
{
    hg = HungarianAlgorithmPtr(new HungarianAlgorithm());
}


Sort::~Sort()
{
}


cv::Mat Sort::update(const cv::Mat &bboxesDet)
{
    assert(bboxesDet.rows >= 0 && bboxesDet.cols == 6); // detections, [xc, yc, w, h, score, class_id]

    cv::Mat bboxesPred(0, 6, CV_32F, cv::Scalar(0));  // predictions used in data association, [xc, yc, w, h, ...]
    cv::Mat bboxesPost(0, 7, CV_32F, cv::Scalar(0));  // bounding boxes estimate, [xc, yc, w, h, score, class_id, tracker_id]

    // kalman bbox tracker predict
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        cv::Mat bboxPred = (*it)->predict();   // Mat(1, 4)
        if (isAnyNan<float>(bboxPred))
            trackers.erase(it);     // remove the NAN value and corresponding tracker
        else{
            cv::hconcat(bboxPred, cv::Mat(1, 2, CV_32F,cv::Scalar(0)), bboxPred);   // Mat(1, 6)
            cv::vconcat(bboxesPred, bboxPred, bboxesPred);  // Mat(N, 6)
            ++it;
        }
    }

    TypeAssociate asTuple = dataAssociate(bboxesDet, bboxesPred);
    TypeMatchedPairs matchedDetPred = std::get<0>(asTuple);
    TypeLostDets lostDets = std::get<1>(asTuple);
    TypeLostPreds lostPreds = std::get<2>(asTuple);

    // update matched trackers with assigned detections
    for (auto pair : matchedDetPred)
    {
        int detInd = pair.first;
        int predInd = pair.second;
        cv::Mat bboxPost = trackers[predInd]->update(bboxesDet.rowRange(detInd, detInd + 1));

        if (trackers[predInd]->getHitStreak() >= minHits)
        {
            float score = bboxesDet.at<float>(detInd, 4);
            int classId = bboxesDet.at<float>(detInd, 5);
            int trackerId = trackers[predInd]->getFilterId();
            cv::hconcat(bboxPost, cv::Mat(1, 1, CV_32F, cv::Scalar(score)), bboxPost);      // score
            cv::hconcat(bboxPost, cv::Mat(1, 1, CV_32F, cv::Scalar(classId)), bboxPost);    // classId
            cv::hconcat(bboxPost, cv::Mat(1, 1, CV_32F, cv::Scalar(trackerId)), bboxPost);  // tracker id
            cv::vconcat(bboxesPost, bboxPost, bboxesPost);  // Mat(N, 7)
        }
    }

    // remove dead trackers
    for (auto it = trackers.begin(); it != trackers.end(); )
    {
        if ((*it)->getTimeSinceUpdate() > maxAge)
            trackers.erase(it);
        else
            ++it;
    }

    // create and initialize new trackers for unmatched detections
    for (int lostInd : lostDets)
    {
        cv::Mat lostBbox = bboxesDet.rowRange(lostInd, lostInd + 1);
        trackers.push_back(KalmanBoxTrackerPtr(new KalmanBoxTracker(lostBbox)));
    }

    return bboxesPost;
}


TypeAssociate Sort::dataAssociate(const cv::Mat& bboxesDet, const cv::Mat& bboxesPred)
{
    TypeMatchedPairs matchedDetPred;
    TypeLostDets lostDets;
    TypeLostPreds lostPreds;

    // initialize
    for (int i = 0; i < bboxesDet.rows; ++i)
        lostDets.push_back(i);  // size M
    for (int j = 0; j < bboxesPred.rows; ++j)
        lostPreds.push_back(j); // size N

    // nothing detected or predicted
    if (bboxesDet.rows == 0 || bboxesPred.rows == 0)
        return make_tuple(matchedDetPred, lostDets, lostPreds);

    // compute distance (or cost) matrix
    cv::Mat iouMat = getIouMatrix(bboxesDet, bboxesPred);   // Mat(M, N)
    cv::Mat distMat = 1.0 - iouMat;

    // Hungarian combinatorial optimization algorithm 
    vector<vector<double>> distVec2D;
    for (int i = 0; i < distMat.rows; ++i)
    {
        vector<double> tmp;
        for (int j = 0; j < distMat.cols; ++j)
            tmp.push_back(double(distMat.at<float>(i, j)));
        distVec2D.push_back(tmp);
    }
    vector<int> assignment;
    hg->Solve(distVec2D, assignment);   // assignment, length=M

    // find matched pairs and lost detect and predict
    for (int detInd = 0; detInd < assignment.size(); ++detInd)
    {
        int predInd = assignment[detInd];
        if (predInd != -1 && iouMat.at<float>(detInd, predInd) >= iouThresh)
        {
            matchedDetPred.push_back(pair<int, int>(detInd, predInd));
            lostDets.erase(remove(lostDets.begin(), lostDets.end(), detInd), lostDets.end());
            lostPreds.erase(remove(lostPreds.begin(), lostPreds.end(), predInd), lostPreds.end());
        }
    }

    return make_tuple(matchedDetPred, lostDets, lostPreds);
}


cv::Mat Sort::getIouMatrix(const cv::Mat& bboxesA, const cv::Mat& bboxesB)
{
    assert(bboxesA.cols >= 4 && bboxesB.cols >= 4);
    int numA = bboxesA.rows;
    int numB = bboxesB.rows;
    cv::Mat iouMat(numA, numB, CV_32F, cv::Scalar(0.0));

    cv::Rect re1, re2;
    for (int i = 0; i < numA; ++i)
    {
        for (int j = 0; j < numB; ++j)
        {
            re1.x = bboxesA.at<float>(i, 0) - bboxesA.at<float>(i, 2) / 2.0;
            re1.y = bboxesA.at<float>(i, 1) - bboxesA.at<float>(i, 3) / 2.0;
            re1.width = bboxesA.at<float>(i, 2);
            re1.height = bboxesA.at<float>(i, 3);
            re2.x = bboxesB.at<float>(j, 0) - bboxesB.at<float>(j, 2) / 2.0;
            re2.y = bboxesB.at<float>(j, 1) - bboxesB.at<float>(j, 3) / 2.0;
            re2.width = bboxesB.at<float>(j, 2);
            re2.height = bboxesB.at<float>(j, 3);

            iouMat.at<float>(i, j) = (re1 & re2).area() / ((re1 | re2).area() + FLT_EPSILON);
        }
    }

    return iouMat;
}