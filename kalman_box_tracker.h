/**
 * @desc:   kalmanfilter for boundary box tracking.
 *          opencv kalmanfilter documents:
 *              https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html
 * 
 * @author: lst
 * @date:   12/10/2021
 */
#pragma once

#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <assert.h>
#include <math.h>

#define KF_DIM_X 7      // xc, yc, s, r, dxc/dt, dyc/dt, ds/dt
#define KF_DIM_Z 4      // xc, yc, s, r

namespace sort
{
    class KalmanBoxTracker
    {
    // variables
    public:
    private:
        static int count;
        int id;
        int timeSinceUpdate = 0;
        int hitStreak = 0;
        cv::KalmanFilter *kf = nullptr;
        cv::Mat xPost;
    
    // methods
    public:
        /**
         * @brief Kalman filter for bbox tracking
         * @param bbox Mat(4+, 1), [xc; yc; w; h; ...]
         */
        explicit KalmanBoxTracker(const cv::Mat &bbox);

        virtual ~KalmanBoxTracker();
        KalmanBoxTracker(const KalmanBoxTracker&);
        void operator=(const KalmanBoxTracker&);

        /**
         * @brief updates the state vector with observed bbox. 
         * @param bbox  boundary box (x center, y center, width, height)
         * @return corrected bounding box estimate
         */
        cv::Mat update(const cv::Mat &bbox);

        /**
         * @brief advances the state vector and returns the predicted bounding box estimate. 
         * @return predicted bounding box estimate, NOT THE CORRECTED ONE!
         */
        cv::Mat predict();

        inline int getId()
        {
            return id;
        }

        inline int getTimeSinceUpdate()
        {
            return timeSinceUpdate;
        }

        inline int getHitStreak()
        {
            return hitStreak;
        }

        inline cv::Mat getState()
        {
            return xPost.clone();
        }

    private:
        /**
         * @brief convert boundary box to measurement.
         * @param bbox boundary box (x center, y center, width, height)
         * @return measurement vector (x center, y center, scale/area, aspect ratio)
         */
        static inline cv::Mat convertBBoxToZ(const cv::Mat &bbox)
        {
            assert(bbox.rows == 1 && bbox.cols >= 1);
            float x = bbox.at<float>(0, 0);
            float y = bbox.at<float>(0, 1);
            float s = bbox.at<float>(0, 2) * bbox.at<float>(0, 3);
            float r = bbox.at<float>(0, 2) / bbox.at<float>(0, 3);

            return (cv::Mat_<float>(4, 1) << x, y, s, r);
        }

        /**
         * @brief convert state vector to boundary box.
         * @param state state vector (x center, y center, scale/area, aspect ratio, ...)
         * @return boundary box (x center, y center, width, height)
         */
        static inline cv::Mat convertXToBBox(const cv::Mat &state)
        {
            assert(state.rows >= 4 && state.cols == 1);
            float x = state.at<float>(0, 0);
            float y = state.at<float>(0, 1);
            float w = sqrt(state.at<float>(0, 2) * state.at<float>(0, 3));
            float h = state.at<float>(0, 2) / w;

            return (cv::Mat_<float>(4, 1) << x, y, w, h);
        }

        /**
         * @brief convert state vector to boundary box with score.
         * @param state state vector (x center, y center, scale/area, aspect ratio, ...)
         * @return boundary box with score (x center, y center, width, height, score)
         */
        static inline cv::Mat convertXToBBox(const cv::Mat &state, float score)
        {
            cv::Mat bbox = KalmanBoxTracker::convertXToBBox(state);
            cv::vconcat(bbox, cv::Mat(1, 1, CV_32F, cv::Scalar(score)), bbox);
            return bbox;
        }
    };
}

