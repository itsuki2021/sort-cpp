#include <iostream>
#include <fstream>
#include "sort.h"

using namespace std;
using namespace sort;


// read detections from filestream
vector<cv::Mat> getDetInFrames(string fileName)
{
    vector<cv::Mat> allDet;

    ifstream ifs(fileName);
    if (!ifs.is_open())
    {
        cerr << "can not open the file " << fileName << endl;
        return allDet;
    }

    // read detections in frames
    string line;
    istringstream iss;
    int crtFrame = -1, frame, classId;
    char dummy;
    float x, y, w, h, score;
    cv::Mat bboxesDet = cv::Mat::zeros(0, 5, CV_32F);   // detections in one frame
    while (getline(ifs, line))
    {
        
        iss.str(line);
        iss >> frame >> dummy >> classId >> dummy;
        iss >> x >> dummy >> y >> dummy >> w >> dummy >> h >> dummy >> score;
        iss.str("");

        cv::Mat bbox = (cv::Mat_<float>(1, 5) << x + w/2, y + h/2, w, h, score);

        if (crtFrame == -1)
        {
            crtFrame = frame;
        }
        else if (frame != crtFrame)
        {
            allDet.push_back(bboxesDet.clone());
            bboxesDet = cv::Mat::zeros(0, 5, CV_32F);
            crtFrame = frame;
        }
        
        cv::vconcat(bboxesDet, bbox, bboxesDet);
    }
    allDet.push_back(bboxesDet.clone());
    ifs.close();

    return allDet;
}


int main(int argc, char** argv)
{
    cout << "SORT demo" << endl;
    if (argc != 3)
    {
        cout << "usage: ./demo_sort [input txt] [output txt]" << endl;
        return -1;
    }
    string inputFile(argv[1]);
    string outputFile(argv[2]);
    // string inputFile = "../data/train/ADL-Rundle-6//det/det.txt";
    // string outputFile = "./ADL-Rundle-6.txt";

    vector<cv::Mat> allDet = getDetInFrames(inputFile);

    ofstream ofs(outputFile);
    if (!ofs.is_open())
    {
        cerr << "can not create the file " << outputFile << endl;
        return -1;
    }

    // SORT
    Sort sort(1, 3, 0.3);
    for (int i = 0; i < allDet.size(); ++i)
    {
        cv::Mat bboxesDet = allDet[i];
        cv::Mat bboxesPost = sort.update(bboxesDet);
        for (int j = 0; j < bboxesPost.rows; ++j)
        {
            float xc = bboxesPost.at<float>(j, 0);
            float yc = bboxesPost.at<float>(j, 1);
            float w = bboxesPost.at<float>(j, 2);
            float h = bboxesPost.at<float>(j, 3);
            float score = bboxesPost.at<float>(j, 4);
            float trackerId = bboxesPost.at<float>(j, 5);

            ofs << (i + 1) << ", " << (xc - w / 2) << ", " << (yc - h / 2 ) << ", " 
                << w << ", " << h << ", " << score << ", \t" << trackerId << " --- " << KalmanBoxTracker::getFilterCount() << endl;
        }
    }

    ofs.close();
    cout << "save tracking result to " << outputFile << endl;
    cout << "done." << endl;
    return 0;
}