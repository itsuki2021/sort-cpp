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
    cv::Mat bboxesDet = cv::Mat::zeros(0, 6, CV_32F);   // detections in one frame
    while (getline(ifs, line))
    {
        
        iss.str(line);
        iss >> frame >> dummy >> classId >> dummy;
        iss >> x >> dummy >> y >> dummy >> w >> dummy >> h >> dummy >> score;
        iss.str("");

        cv::Mat bbox = (cv::Mat_<float>(1, 6) << x + w/2, y + h/2, w, h, score, classId);

        if (crtFrame == -1)
        {
            crtFrame = frame;
        }
        else if (frame != crtFrame)
        {
            allDet.push_back(bboxesDet.clone());
            bboxesDet = cv::Mat::zeros(0, 6, CV_32F);
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
    // string inputFile = "../data/train/ADL-Rundle-6/det/det.txt";
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
    for (int frame = 0; frame < allDet.size(); ++frame)
    {
        cv::Mat bboxesDet = allDet[frame];
        cv::Mat bboxesPost = sort.update(bboxesDet);
        for (int cnt = 0; cnt < bboxesPost.rows; ++cnt)
        {
            float xc = bboxesPost.at<float>(cnt, 0);
            float yc = bboxesPost.at<float>(cnt, 1);
            float w = bboxesPost.at<float>(cnt, 2);
            float h = bboxesPost.at<float>(cnt, 3);
            float score = bboxesPost.at<float>(cnt, 4);
            int classId = bboxesPost.at<float>(cnt, 5);
            float dx = bboxesPost.at<float>(cnt, 6);
            float dy = bboxesPost.at<float>(cnt, 7);
            int trackerId = bboxesPost.at<float>(cnt, 8);

            ofs << (frame + 1) << "," << (xc - w / 2) << "," << (yc - h / 2 ) << "," 
                << w << "," << h << "," << score << "," << classId << "," << trackerId << endl;
        }
    }

    ofs.close();
    cout << "save tracking result to " << outputFile << endl;
    cout << "done." << endl;
    return 0;
}