/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/*
// brief This tutorial replaces OpenCV DNN with IE
*/

#include <string>
#include <chrono>
#include <cmath>
#include <inference_engine.hpp>
#include <cpp/ie_plugin_cpp.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <ext_list.hpp>

#include <iomanip>

#include "face_recognition.h"
using namespace std;
using namespace cv;
using namespace InferenceEngine::details;
using namespace InferenceEngine;

cv::Mat prewhitten(cv::Mat input  ){     
    float sum=0.0;
    float sum_stddev=sum;
    //image read checked. correct. 
    //cout<<"input matix before"<<endl<<input<<endl;
    Mat buffer;
    for (size_t pixelnum = 0, imgIdx = 0; pixelnum < 25600; ++pixelnum) {
        for (size_t ch = 0; ch < 3; ++ch) {
            sum += input.at<cv::Vec3b>(pixelnum)[ch];                    
        }
    } 
    //std::cout<<"finished sum "<<endl;
    float mean=sum/25600.0/3.0;
    //cout<<mean<<endl;
    for (size_t pixelnum = 0, imgIdx = 0; pixelnum < 25600; ++pixelnum) {
        for (size_t ch = 0; ch < 3; ++ch) {
            sum_stddev += pow ( (input.at<cv::Vec3b>(pixelnum)[ch] - mean) , 2 );                    
        }
    }
    float stddev= sqrt(sum_stddev/25600.0/3.0);
 
 
    input.convertTo(buffer,CV_32FC3);
    //cout<<"mean="<<mean<<endl;
    Mat M_sub(160,160, CV_32FC3, Scalar(mean,mean,mean));
    Mat M_div(160,160, CV_32FC3, Scalar(stddev,stddev,stddev));
    cv::subtract(buffer,M_sub,buffer);
    Mat output= buffer.mul(1/stddev);
    
    //cout<< "prehitten matix after----"<<endl<<output<<endl<<"em"<<endl;
    return output;
}

FaceRecognitionClass::~FaceRecognitionClass(){
    delete [] output_frames;
}

int FaceRecognitionClass::initialize(std::string modelfile){
    InferenceEngine::PluginDispatcher dispatcher({""});
    //InferencePlugin plugin(dispatcher.getPluginByDevice("CPU")); 
    plugin=dispatcher.getPluginByDevice("CPU"); 
    cout<< "============Initialize FaceRecognition ================="<<endl;
    try {
        networkReader.ReadNetwork(modelfile);
    }
    catch (InferenceEngineException ex) {
        cerr << "Failed to load network: "  << endl;
        return 1;
    }

    cout << "Network loaded." << endl;
    auto pos=modelfile.rfind('.');
    if (pos !=string::npos) {
        string binFileName=modelfile.substr(0,pos)+".bin";
        std::cout<<"binFileName="<<binFileName<<std::endl;
        networkReader.ReadWeights(binFileName.c_str());
    }
    else {
        cerr << "Failed to load weights: " << endl;
        return 1;
    }

    auto network = networkReader.getNetwork();

    // --------------------
    // Set batch size
    // --------------------
    networkReader.getNetwork().setBatchSize(1);
    size_t batchSize = network.getBatchSize();

    cout << "Batch size = " << batchSize << endl;

    //----------------------------------------------------------------------------
    //  Inference engine input setup
    //----------------------------------------------------------------------------

    cout << "Setting-up input, output blobs..." << endl;

    // ---------------
    // set input configuration
    // ---------------
    //InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    input_info=network.getInputsInfo();
    InferenceEngine::SizeVector inputDims;
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

    if (input_info.size() != 1) {
        cout << "This sample accepts networks having only one input." << endl;
        return 1;
    }

    for (auto &item : input_info) {
        auto input_data = item.second;
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        inputDims=input_data->getDims();
    }
    cout << "inputDims=";
    for (int i=0; i<inputDims.size(); i++) {
        cout << (int)inputDims[i] << " ";
    }
    cout << endl;

    const int infer_width=inputDims[0];
    const int infer_height=inputDims[1];
    const int num_channels=inputDims[2];
    const int channel_size=infer_width*infer_height;
    const int full_image_size=channel_size*num_channels;

    /** Get information about topology outputs **/
    output_info=network.getOutputsInfo();
    InferenceEngine::SizeVector outputDims;
    for (auto &item : output_info) {
        auto output_data = item.second;
        output_data->setPrecision(Precision::FP32);
        output_data->setLayout(Layout::NCHW);
        outputDims=output_data->getDims();
    }
    cout << "outputDims=";
    for (int i=0; i<outputDims.size(); i++) {
        cout << (int)outputDims[i] << " ";
    }
    cout << endl;

    const int output_data_size=outputDims[1]*outputDims[2]*outputDims[3];

    // --------------------------------------------------------------------------
    // Load model into plugin
    // --------------------------------------------------------------------------
    cout << "Loading model to plugin..." << endl;

    executable_network = plugin.LoadNetwork(network, {});

    // --------------------------------------------------------------------------
    // Create infer request
    // --------------------------------------------------------------------------
    cout << "Create infer request..." << endl;

    async_infer_request = executable_network.CreateInferRequestPtr();

    if (async_infer_request == nullptr) {
       cout << "____________________________Failed to create async infer req." << std::endl;
    }

    //----------------------------------------------------------------------------
    //  Inference engine output setup
    //----------------------------------------------------------------------------

    Mat frame,frameInfer, frameInfer_prewhitten;

    // get the input blob buffer pointer location
   // free(input_buffer);
    input_buffer = NULL;
    for (auto &item : input_info) {
        auto input_name = item.first;
        auto input_data = item.second;
        auto input = async_infer_request->GetBlob(input_name);
        input_buffer = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    }

    // get the output blob pointer location

    output_buffer = NULL;
    for (auto &item : output_info) {
        auto output_name = item.first;
        auto output = async_infer_request->GetBlob(output_name);
        output_buffer = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    }

    output_frames = new Mat[1];
    cout<< "==========Sucessfully initialized face_recognition plugin==============="<<endl;
    return 0;
}

/*load a frame into the network, given that the network is already initialized*/
void FaceRecognitionClass::load_frame(cv::Mat frame){
    const size_t input_width   = 160;
    const size_t input_height  = 160;
    const size_t output_width  = 1;
    const size_t output_height = 512;
    const size_t infer_width = 160;
    const size_t infer_height= 160;
    const size_t num_channels= 3;

    int pipeline_type = 3;  //default.  Input frame is resized to output dims as well as infer dims
    if ((output_width==infer_width) && (output_height==infer_height)) pipeline_type=1; // Input frame is resized to infer resolution
    else if ((output_width==input_width) && (output_height==input_height)) pipeline_type=2; //output is input resolution

    //open video output
    switch (pipeline_type)
        {
        case 1: // Input frame is resized to infer resolution
            resize(frame,output_frames[0], Size(infer_width, infer_height));
            frameInfer=output_frames[0];
            break;
        case 2: //output is input resolution
            output_frames[0]=frame;
            resize(frame, frameInfer, Size(infer_width, infer_height));
            break;
        default: //other cases -- resize for output and infer
            //resize(frame,output_frames[0], Size(output_width, output_height));
            resize(frame, frameInfer, Size(infer_width, infer_height));
        }

    //std::cout<<"output size"<< output_frames[0].rows<<std::endl;
    //std::cout<<"frameInfer input"<< frameInfer.rows<< "|"<<frameInfer.cols<<std::endl;
    auto input_channels = 3;  // channels for color format.  RGB=4   
    int input_size=76800; 
    size_t framesize = frameInfer.rows * frameInfer.step1();
    if (framesize != 76800) {
        cout << "input pixels mismatch, expecting " << input_size
                  << " bytes, got: " << framesize << endl;
    }
    //prewhitten the frame, required, as the model is trained with prewhittened images
    this->frameInfer_prewhitten=prewhitten(frameInfer);
    int channel_size=25600;

    for (size_t pixelnum = 0, imgIdx = 0; pixelnum < 25600; ++pixelnum) {
        for (size_t ch = 0; ch < num_channels; ++ch) {
            if (input_buffer==NULL){
                cout<< "input_buffer problem"<<endl;
            }
            input_buffer[(ch * channel_size) + pixelnum] = (float)frameInfer_prewhitten.at<cv::Vec3f>(pixelnum)[ch];                    
        }
    }  
}

std::vector<float> FaceRecognitionClass::do_infer(){
    int num_channels=3;
    int channel_size=25600;

    async_infer_request->StartAsync();
            

    async_infer_request->Wait(IInferRequest::WaitMode::RESULT_READY);
    //async_infer_request.Wait();

    std::vector<float> v1;
    for (int i = 0; i < 512; ++i)
    {
        float *localbox=&output_buffer[i];
        //std::cout<<(float)localbox[0]<<" ";
        /*!!!!!!!!! chage back to localbox0*/
        v1.push_back(localbox[0]);
    }
    return v1;
}


