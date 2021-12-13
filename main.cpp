#include <iostream>
#include <fstream>
#include "sort.h"

using namespace std;
using namespace sort;

int main(int argc, char** argv)
{
    // if (argc != 3)
    // {
    //     cout << "usage: ./demo_sort [input txt] [output txt]" << endl;
    //     return -1;
    // }
    // string inputFile(argv[1]);
    // string outputFile(argv[2]);
    string inputFile = "../data/train/ADL-Rundle-6//det/det.txt";
    string outputFile = "../output/ADL-Rundle-6.txt";

    cout << "SORT demo" << endl;
    vector<cv::Mat> vecInputDets;   // detections in all frames

    ifstream ifs(inputFile);
    if (!ifs.is_open())
    {
        cerr << "Can not open input file " << inputFile << endl;
        return -1;
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
            vecInputDets.push_back(bboxesDet.clone());
            bboxesDet = cv::Mat::zeros(0, 5, CV_32F);
            crtFrame = frame;
        }
        
        cv::vconcat(bboxesDet, bbox, bboxesDet);
    }
    vecInputDets.push_back(bboxesDet.clone());
    ifs.close();
    cout << "Input detection loaded." << endl;

    ofstream ofs(outputFile);
    if (!ofs.is_open())
    {
        cerr << "Can not open output file " << outputFile << endl;
        return -1;
    }

    // SORT
    shared_ptr<Sort> sort = shared_ptr<Sort>(new Sort(1, 3, 0.3));
    for (int i = 0; i < vecInputDets.size(); ++i)
    {
        cv::Mat bboxesDet = vecInputDets[i];
        cv::Mat bboxesPost = sort->update(bboxesDet);
        for (int j = 0; j < bboxesPost.rows; ++j)
        {
            float xc = bboxesPost.at<float>(j, 0);
            float yc = bboxesPost.at<float>(j, 1);
            float w = bboxesPost.at<float>(j, 2);
            float h = bboxesPost.at<float>(j, 3);
            float score = bboxesPost.at<float>(j, 4);
            float trackerId = bboxesPost.at<float>(j, 5);

            ofs << (i + 1) << ",-1,"
                << (xc - w / 2) << "," << (yc - h / 2 ) << "," 
                << w << "," << h << "," 
                << "1,-1,-1,-1" << endl;
        }
    }
    ofs.close();

    cout << "Done." << endl;
    return 0;
}