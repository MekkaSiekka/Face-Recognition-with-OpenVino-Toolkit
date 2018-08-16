/*
 * Copyright 2017 Intel Corporation All Rights Reserved. 
 * 
 * The source code contained or described herein and all documents related to the 
 * source code ("Material") are owned by Intel Corporation or its suppliers or 
 * licensors. Title to the Material remains with Intel Corporation or its suppliers 
 * and licensors. The Material contains trade secrets and proprietary and 
 * confidential information of Intel or its suppliers and licensors. The Material 
 * is protected by worldwide copyright and trade secret laws and treaty provisions. 
 * No part of the Material may be used, copied, reproduced, modified, published, 
 * uploaded, posted, transmitted, distributed, or disclosed in any way without 
 * Intel's prior express written permission.
 * 
 * No license under any patent, copyright, trade secret or other intellectual 
 * property right is granted to or conferred upon you by disclosure or delivery of 
 * the Materials, either expressly, by implication, inducement, estoppel or 
 * otherwise. Any license under such intellectual property rights must be express 
 * and approved by Intel in writing.
 */
#include "face_detection.cpp"
#include "face_recognition.cpp"
#include "person_detection.cpp"
#include "thread_func.h"
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <functional>
#include <fstream>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <inference_engine.hpp>

#include <common.hpp>
#include <slog.hpp>
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <ext_list.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <sys/stat.h>
#include <thread>
#include <mutex>

using namespace InferenceEngine;
using namespace cv;
using namespace std;
using namespace InferenceEngine::details;

std::vector<std::pair<std::string, std::vector<float> > > data;
std::mutex mtx;

FaceDetectionClass FaceDetection;
FaceRecognitionClass FaceRecognition;
PersonDetectionClass PersonDetection;

cv::Mat mGlob(1920,1080,CV_8UC3);
vector<recog_result> global_recog_results;
volatile bool need_process=false;

/**returns: the cropped square image of crop_size having same center of rectangle**/
/**style=0 : original face detection results with largest possible square
   style>0 : larger square resized **/
Mat get_cropped(Mat input, Rect rectangle , int crop_size , int style){
    int center_x= rectangle.x + rectangle.width/2;
    int center_y= rectangle.y + rectangle.height/2;

    int max_crop_size= min(rectangle.width , rectangle.height);

    int adjusted= max_crop_size * 3 / 2;

    std::vector<int> good_to_crop;
    good_to_crop.push_back(adjusted / 2);
    good_to_crop.push_back(input.size().height - center_y);
    good_to_crop.push_back(input.size().width - center_x);
    good_to_crop.push_back(center_x);
    good_to_crop.push_back(center_y);

    int final_crop= * (min_element(good_to_crop.begin(), good_to_crop.end()));
  
    Rect pre (center_x - max_crop_size /2, center_y - max_crop_size /2 , max_crop_size , max_crop_size  );
    Rect pre2 (center_x - final_crop, center_y - final_crop , final_crop *2  , final_crop *2  );
    Mat r;
    if (style==0)   r = input (pre);
    else  r = input (pre2);

    resize(r,r,Size(crop_size,crop_size));
    return r;
}

vector<pair<string, vector<float> > > read_text(){
    cout<< "Reading vectors of saved person"<<endl;
    vector<pair<string, vector<float> > > data;
    string file="/home/xiaojiang/Desktop/webrtc-mcu-analytics-sdk/plugin/samples/face_recognition_plugin/vectors.txt";
    string line;
    std::ifstream infile(file);
    int count;
    float value ;
    vector <float> vec;
    string name;
    while (std::getline(infile, line))
    {
        name=string(line);
        cout<< name ;
        getline(infile , line);
        count= atoi(line.c_str());
        cout<<" : " << count <<endl;
        for (int i = 0; i < count; ++i)
        {
            vec.clear();
            for (int j = 0; j < 512; ++j)
            {
                getline(infile , line);
                value=strtof(line.c_str(),0);
                vec.push_back(value);
            }
            data.push_back ( pair< string,vector<float> > (name, vec) );
        }
    }
    return data;
}

float calculate_dist(std::vector<float> v1, std::vector<float> v2, int len){
    float sum=0.0;
    float subtract=0.0;
    for (int i = 0; i < 512; ++i)
    {
        subtract=v1[i]-v2[i];
        sum=sum+abs(subtract*subtract);
    }
    return sqrt(sum);
}

//requires : data of name-feature set / person to be recognized /threshold 
pair<float , string> recognize(vector<pair<string, vector<float> > > *data , vector <float> target, float threshold){
    float dist=3.0;
    float min_value=3.0;
    string best_match="Unknown";
    for (auto entry : *(data)){
        dist=calculate_dist(target , entry.second , 512);
        if (dist < min_value){
            min_value=dist;
            best_match=entry.first;
        } 
    }
    if (min_value > threshold) best_match= "Unknown";
    return pair<float, string> (min_value, best_match);
}


   
threading_class::threading_class(){}

void threading_class::make_thread(){
    std::thread T=std::thread(&threading_class::threading_func, this);
    T.detach();
}


//------------calling function for the new thread-------------------
void threading_class::threading_func(){
    std::cout<<"creating new thread for inferecne async"<<std::endl;
    while (true){
        if (need_process){
            mtx.lock();
            Mat frame= mGlob.clone();
            mtx.unlock();
            vector<float> v;
            vector<cv::Rect> PersonDetection_rect;
            PersonDetection.load_frame(frame);
            v= PersonDetection.do_infer();

            int x,y,width,height;
            float confidence;
            PersonDetection_rect.clear();
            for (int i = 0; i < v.size(); i=i+7)
            {
                //cout<< i <<endl;
                confidence= v[i+2];
                x=frame.cols*v[i+3];
                y=frame.rows*v[i+4];
                width= ( frame.cols * (v[i+5]-v[i+3]) );
                height=  (frame.rows * (v[i+6] - v[i+4]) );
                Rect rect (x ,y , width ,height);
                rect.height=rect.width;
                if (confidence > 0.2) {
                        PersonDetection_rect.push_back(rect);
                }
            }
            //end of persondetection
            
            Mat a_person;
            vector<Mat> persons;
            vector<Rect> FaceDetection_rect;
                
            vector<recog_result> local_recog_results;
            local_recog_results.clear();
            //cout<<"begining for loop"<<endl;
            for (auto rect: PersonDetection_rect) {
                //cases when the output from persondetection is not good for cropping out the person
                if ((rect.width + rect.x) > frame.cols) {
                    rect.width = frame.cols- abs(rect.x);
                }  
                if ((rect.height  + rect.y) > frame.rows) {
                    rect.height = frame.rows- abs(rect.y);
                }
                if (rect.x <0) {
                    rect.x=0;
                }
                if (rect.y <0){
                    rect.y=0;
                }
                //int overflow/bad output handeling
                if ( (abs(rect.x) > 10000 || abs(rect.y) > 10000 || abs(rect.height) > 10000 || abs(rect.width) > 10000) )
                        continue;
                
                //cout << rect.x << "-"<<rect.y << "-"<<rect.width <<" -"<<rect.height<< endl;
                a_person = frame(rect);
                FaceDetection.enqueue(a_person);
                FaceDetection.submitRequest();
                FaceDetection.wait();
                FaceDetection.fetchResults();
                //cout<< FaceDetection.results.size() <<endl;

                //among all face detection results, do face recog. 
                for (auto r : FaceDetection.results){
                    //cout << r.confidence <<endl;
                    if (r.confidence>0.2) {
                        std::vector<float> output_vector;
                        pair<float, string> compare_result;
                        //--------------for this frame, output the charactiristic vector ------------------            
                        Mat cropped=get_cropped(a_person, r.location, 160 , 2);
                        FaceRecognition.load_frame(cropped);
                        output_vector= FaceRecognition.do_infer();
                        //-------------compare with the dataset, return the most possible person if any---
                        compare_result=recognize(&data , output_vector, 0.6);

                        if (compare_result.second!="Unknown")   {
                            //The output position of facedetection is related to a_person's local corrdinate, 
                                //so need to conver to global
                            recog_result a_result;
                            a_result.name=compare_result.second;
                            a_result.value=compare_result.first;
                            a_result.face_rect=Rect(r.location.x+rect.x, r.location.y+rect.y,
                                                    r.location.width,r.location.height);
                            a_result.body_rect=rect;
                            local_recog_results.push_back(a_result);
                        }
                    }
                }
            }

            mtx.lock();
            global_recog_results.clear();
            for (auto result : local_recog_results){
                global_recog_results.push_back(result);
            }   
            mtx.unlock();
            //cout<<"a frame"<<endl;
        }
        //else std::cout<<"no need"<<std::endl; 
    }
}


int main(){
//-----------------------Read data from "vectors.txt"----------------------
    data=read_text();

//-----------------------Initialize Detections -----------------------------
    FaceRecognition.initialize("/home/xiaojiang/Desktop/webrtc-mcu-analytics-sdk/plugin/samples/face_recognition_plugin/model/20180402-114759.xml");
    FaceDetection.initialize();
    PersonDetection.initialize("/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-person-detection-retail-0002/FP32/face-person-detection-retail-0002.xml");

    threading_class t_c;
    t_c.make_thread();

//----------------------------------Get video source -----------------------------
    VideoCapture cap(0); 
    cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
          // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    int fps=0;
    Mat mBGR;
    while (1){
        fps++;
        cout<<fps<<endl;
        cap.read(mBGR);
        //mBGR= imread("/home/xiaojiang/Desktop/1.jpg");

        if (mBGR.empty()){
            continue;
        }
// --------------------------split--face--recog----style ------------------
        mtx.lock();
        mGlob=mBGR.clone();
        need_process=true;
        mtx.unlock();

        mtx.lock();
        for(auto result : global_recog_results){
            //rectangle(mBGR, result.body_rect, Scalar(255,255,18),2);
            Rect background_rect(result.face_rect.x,
                    min(result.face_rect.y  + result.face_rect.height*2, mBGR.rows-50),
                    200,60
            );
            rectangle(mBGR, background_rect, Scalar(216,206,199),-2);
            
            putText(mBGR, 
                        result.name , 
                        cv::Point2f(background_rect.x,background_rect.y+background_rect.height/2), 
                        FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(198,76,0), 1, CV_AA); 
            /*
            putText(mBGR, 
                        "Value : " + to_string(result.value) , 
                        cv::Point2f(result.body_rect.x, result.body_rect.y ), 
                        FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(73,255,167), 1, CV_AA); */
        }
        mtx.unlock();
        imshow (" asdf", mBGR);
        waitKey(1);
    }
      return 0;
}
