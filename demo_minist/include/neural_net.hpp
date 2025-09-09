#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class NeuralNet {
public:
    NeuralNet(int input_size, int hidden_size, int output_size);
    std::vector<float> forward(const cv::Mat& input);
    void train(const std::vector<cv::Mat>& images, const std::vector<int>& labels, int epochs);

private:
    cv::Mat weights_ih, weights_ho, bias_h, bias_o;
    float learning_rate = 0.01f;

    cv::Mat relu(const cv::Mat& x);
    cv::Mat softmax(const cv::Mat& x);
};