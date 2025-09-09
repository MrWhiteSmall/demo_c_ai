#include "neural_net.hpp"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// 构造函数：初始化神经网络的权重和偏置
NeuralNet::NeuralNet(int input_size, int hidden_size, int output_size) {
    // 初始化权重和偏置为零矩阵
    weights_ih = Mat::zeros(input_size, hidden_size, CV_32FC1); // 输入层到隐藏层权重
    weights_ho = Mat::zeros(hidden_size, output_size, CV_32FC1); // 隐藏层到输出层权重
    bias_h = Mat::zeros(1, hidden_size, CV_32FC1); // 隐藏层偏置
    bias_o = Mat::zeros(1, output_size, CV_32FC1); // 输出层偏置

    // 随机初始化权重和偏置（高斯分布）
    randn(weights_ih, 0, 1.0/sqrt(input_size));
    randn(weights_ho, 0, 1.0/sqrt(hidden_size));
    randn(bias_h, 0, 0.1);
    randn(bias_o, 0, 0.1);
}

// ReLU激活函数
Mat NeuralNet::relu(const Mat& x) {
    Mat result = x.clone();
    // 对每个元素进行ReLU操作
    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j) {
            if (result.at<float>(i,j) < 0) {
                result.at<float>(i,j) = 0;
            }
        }
    }
    return result;
}

// Softmax激活函数
Mat NeuralNet::softmax(const Mat& x) {
    Mat exp_x;
    exp(x, exp_x); // 对每个元素取指数
    float sum_exp = sum(exp_x)[0]; // 求和
    return exp_x / sum_exp; // 归一化
}

// 前向传播函数
vector<float> NeuralNet::forward(const Mat& input) {
    // 展平输入图像为一行
    Mat flat_input = input.reshape(1, 1);
    
    // 输入层到隐藏层的线性变换并加偏置
    Mat hidden = flat_input * weights_ih + bias_h;
    hidden = relu(hidden); // 应用ReLU激活函数
    
    // 隐藏层到输出层的线性变换并加偏置
    Mat output = hidden * weights_ho + bias_o;
    output = softmax(output); // 应用Softmax激活函数
    
    // 将输出转换为vector<float>
    vector<float> result(output.cols);
    for (int i = 0; i < output.cols; ++i) {
        result[i] = output.at<float>(0, i);
    }
    return result;
}

// 训练函数
void NeuralNet::train(const vector<Mat>& images, const vector<int>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0; // 累计损失
        int correct = 0;      // 正确预测数
        
        // 遍历所有训练样本
        for (size_t i = 0; i < images.size(); ++i) {
            // 前向传播
            Mat input = images[i].reshape(1, 1); // 展平输入
            Mat hidden = input * weights_ih + bias_h; // 输入到隐藏层
            Mat hidden_act = relu(hidden); // 隐藏层激活
            Mat output = hidden_act * weights_ho + bias_o; // 隐藏层到输出层
            Mat output_act = softmax(output); // 输出层激活
            
            // 计算损失（交叉熵）和准确率
            int true_label = labels[i];
            total_loss += -log(output_act.at<float>(0, true_label)); // 交叉熵损失
            int predicted = max_element(output_act.begin<float>(), output_act.end<float>()) - output_act.begin<float>();
            if (predicted == true_label) correct++; // 预测正确计数
            
            // 反向传播
            Mat grad_output = output_act.clone(); // 输出层梯度
            grad_output.at<float>(0, true_label) -= 1; // softmax+交叉熵的梯度
            
            // 计算隐藏层到输出层的权重和偏置梯度
            Mat grad_weights_ho = hidden_act.t() * grad_output; // 权重梯度
            Mat grad_bias_o = grad_output.clone(); // 偏置梯度
            
            // 计算隐藏层的梯度
            Mat grad_hidden = grad_output * weights_ho.t();
            // ReLU的梯度
            for (int j = 0; j < hidden_act.cols; ++j) {
                if (hidden_act.at<float>(0, j) <= 0) {
                    grad_hidden.at<float>(0, j) = 0;
                }
            }
            
            // 计算输入层到隐藏层的权重和偏置梯度
            Mat grad_weights_ih = input.t() * grad_hidden; // 权重梯度
            Mat grad_bias_h = grad_hidden.clone(); // 偏置梯度
            
            // 更新权重和偏置（梯度下降）
            weights_ih -= learning_rate * grad_weights_ih;
            bias_h -= learning_rate * grad_bias_h;
            weights_ho -= learning_rate * grad_weights_ho;
            bias_o -= learning_rate * grad_bias_o;
        }
        
        // 输出每个epoch的损失和准确率
        cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / images.size() 
             << ", Accuracy: " << 100.0 * correct / images.size() << "%" << endl;
    }
}