#include <iostream>
#include <fstream>
#include <filesystem>
#include <assert.h>
#include <map>
#include "sort.h"

namespace fs = std::filesystem;

using std::cout;
using std::endl;
using std::ifstream;
using std::stoi;
using std::stof;
using std::vector;
using std::pair;
using std::string;
using std::map;
using std::tuple;

using cv::Mat;
using cv::Mat_;
using cv::Rect;
using cv::Scalar;
using cv::RNG;
using cv::Point;

using sort::Sort;

auto constexpr MAX_COLORS = 2022;
vector<Scalar> COLORS;

vector<string> split(const string& s, char delim) {
    std::istringstream iss(s);
    vector<string> ret;
    string item;
    while (getline(iss, item, delim))
        ret.push_back(item);

    return ret;
}

// (seq info, [(image, detection), ...])
tuple<map<string, string>, vector<pair<Mat, Mat>>> getInputData(string dataFolder) {
    if (*dataFolder.end() != '/') dataFolder += '/';

    ifstream ifs;
    ifs.open(dataFolder + "seqinfo.ini");
    assert(ifs.is_open());
    map<string, string> mp;
    string s;
    while (getline(ifs, s)) {
        size_t pos = s.find('=');
        if (pos != string::npos) {
            string key = s.substr(0, pos);
            string val = s.substr(pos + 1, s.size() - pos);
            mp[key] = val;
        }
    }
    ifs.close();
    assert(mp.find("imDir") != mp.end());
    assert(mp.find("frameRate") != mp.end());
    assert(mp.find("seqLength") != mp.end());

    // get file list
    vector<string> imgPaths;
    for (const auto& entry : fs::directory_iterator(dataFolder + mp["imDir"]))
        imgPaths.push_back(entry.path());
    std::sort(imgPaths.begin(), imgPaths.end());
    assert(imgPaths.size() == std::stoi(mp["seqLength"]));

    vector<pair<Mat, Mat>> pairs(imgPaths.size(), {Mat(0, 0, CV_32F), Mat(0, 6, CV_32F)});

    // read images
    for (int i = 0; i < imgPaths.size(); ++i)
        pairs[i].first = cv::imread(imgPaths[i]);

    // read detections
    ifs.open(dataFolder + "/det/det.txt");
    assert (ifs.is_open());
    float x0, y0, w, h, score;
    int frameId, objId;
    while (getline(ifs, s)) {
        vector<string> ss = split(s, ',');
        frameId = stoi(ss[0]);
        objId = stoi(ss[1]);
        x0 = stof(ss[2]);
        y0 = stof(ss[3]);
        w = stof(ss[4]);
        h = stof(ss[5]);
        score = stof(ss[6]);
        Mat bbox = (Mat_<float>(1, 6) << x0 + w/2, y0 + h/2, w, h, score, 0);
        cv::vconcat(pairs[frameId-1].second, bbox, pairs[frameId-1].second);
    }
    
    return std::make_tuple(mp, pairs);
}

void draw(Mat& img, const Mat& bboxes) {
    float xc, yc, w, h, score, dx, dy;
    int trackerId;
    string sScore;
    for (int i = 0; i < bboxes.rows; ++i) {
        xc = bboxes.at<float>(i, 0);
        yc = bboxes.at<float>(i, 1);
        w = bboxes.at<float>(i, 2);
        h = bboxes.at<float>(i, 3);
        dx = bboxes.at<float>(i, 6);
        dy = bboxes.at<float>(i, 7);
        trackerId = int(bboxes.at<float>(i, 8));

        cv::rectangle(img, Rect(xc - w/2, yc - h/2, w, h), COLORS[trackerId % MAX_COLORS], 2);
        cv::putText(img, std::to_string(trackerId), Point(xc - w/2, yc - h/2 - 4),
                    cv::FONT_HERSHEY_PLAIN, 1.5, COLORS[trackerId % MAX_COLORS], 2);
        cv::arrowedLine(img, Point(xc, yc), Point(xc + 5 * dx, yc + 5 * dy),
                        COLORS[trackerId % MAX_COLORS], 4);
    }
}

int main(int argc, char** argv)
{   
    // generate colors
    RNG rng(MAX_COLORS);
    for (size_t i = 0; i < MAX_COLORS; ++i) {
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        COLORS.push_back(color);
    }

    cout << "SORT demo" << endl;
    if (argc != 2) {
        cout << "usage: ./demo_sort [data folder], e.g. ./demo_sort ../data/TUD-Campus/" << endl;
        return -1;
    }
    string dataFolder = argv[1];
    // string dataFolder = "../data/TUD-Stadtmitte/";
    
    // read image and detections
    cout << "Read image and detections..." << endl;    
    auto [seqInfo, motPairs] = getInputData(dataFolder);
    float fps = std::stof(seqInfo["frameRate"]);

    // tracking
    cout << "Tracking..." << endl;
    Sort::Ptr mot = std::make_shared<Sort>(1, 3, 0.3f);
    cv::namedWindow("SORT", cv::WindowFlags::WINDOW_NORMAL);
    for (auto [img, bboxesDet] : motPairs) {
        Mat bboxesPost = mot->update(bboxesDet);
        
        // show result
        draw(img, bboxesPost);
        cv::imshow("SORT", img);
        cv::waitKey(1000.0 / fps);
    }
    
    cout << "Done" << endl;

    return 0;
}