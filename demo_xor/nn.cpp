#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn.h"

// 激活函数
double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}

// 激活函数的导数
double sigmoid_derivative(double x){
    return x*(1-x);
}

// 创建神经网络
NeuralNetwork* create_nn(int input_size, int hidden_size, int output_size){
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->weight_ih = (double*)malloc(input_size*hidden_size*sizeof(double));
    nn->weight_ho = (double*)malloc(hidden_size*output_size*sizeof(double));
    nn->bias_h = (double*)malloc(hidden_size*sizeof(double));
    nn->bias_o = (double*)malloc(output_size*sizeof(double));

    for (int i = 0; i < input_size*hidden_size; i++){
        nn->weight_ih[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int i=0 ; i<hidden_size*output_size;i++){
        nn->weight_ho[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int i=0; i<hidden_size; i++){
        nn->bias_h[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int i=0; i<output_size; i++){
        nn->bias_o[i] = (double)rand() / RAND_MAX - 0.5;
    }

    return nn;
}

// 前向传播预测
double* predict(NeuralNetwork *nn , double *input){
    double *hidden = (double*)calloc(nn->hidden_size , sizeof(double));
    double *output = (double*)calloc(nn->output_size , sizeof(double));

    // 输入层 -> 隐藏层
    for (int i = 0; i < nn->hidden_size; i++){
        for (int j =0; j < nn->input_size;j++){
            hidden[i] += input[j] * nn->weight_ih[j*nn->hidden_size + i];
        }
        hidden[i] = sigmoid(hidden[i] + nn->bias_h[i]);
    }
    // 隐藏层 -> 输出层
    for (int i = 0; i < nn->output_size; i++){
        for (int j = 0;j < nn->hidden_size; j++){
            output[i] += hidden[j] * nn->weight_ho[j*nn->output_size + i];
        }
        output[i] = sigmoid(output[i] + nn->bias_o[i]);
    }

    free(hidden);
    return output;
}

// 释放神经网络
void free_nn(NeuralNetwork *nn){
    free(nn->weight_ih);
    free(nn->weight_ho);
    free(nn->bias_h);
    free(nn->bias_o);
    free(nn);
}

// 反向传播训练
void train(NeuralNetwork *nn, double *input, double *target, double learning_rate) {
    // 前向传播
    double *hidden = (double*)calloc(nn->hidden_size, sizeof(double));
    double *output = (double*)calloc(nn->output_size, sizeof(double));

    // 输入 → 隐藏层
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            hidden[i] += input[j] * nn->weight_ih[j * nn->hidden_size + i];
        }
        hidden[i] = sigmoid(hidden[i] + nn->bias_h[i]);
    }

    // 隐藏层 → 输出层
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            output[i] += hidden[j] * nn->weight_ho[j * nn->output_size + i];
        }
        output[i] = sigmoid(output[i] + nn->bias_o[i]);
    }

    // 计算输出层误差和梯度
    double *output_errors = (double*)malloc(nn->output_size * sizeof(double));
    double *output_gradients = (double*)malloc(nn->output_size * sizeof(double));
    for (int i = 0; i < nn->output_size; i++) {
        output_errors[i] = target[i] - output[i];
        output_gradients[i] = output_errors[i] * sigmoid_derivative(output[i]);
    }

    // 计算隐藏层误差和梯度
    double *hidden_errors = (double*)malloc(nn->hidden_size * sizeof(double));
    double *hidden_gradients = (double*)malloc(nn->hidden_size * sizeof(double));
    for (int i = 0; i < nn->hidden_size; i++) {
        hidden_errors[i] = 0.0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_errors[i] += output_gradients[j] * nn->weight_ho[i * nn->output_size + j];
        }
        hidden_gradients[i] = hidden_errors[i] * sigmoid_derivative(hidden[i]);
    }

    // 更新权重和偏置 (隐藏层 → 输出层)
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            nn->weight_ho[i * nn->output_size + j] += learning_rate * output_gradients[j] * hidden[i];
        }
    }
    for (int i = 0; i < nn->output_size; i++) {
        nn->bias_o[i] += learning_rate * output_gradients[i];
    }

    // 更新权重和偏置 (输入层 → 隐藏层)
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->weight_ih[i * nn->hidden_size + j] += learning_rate * hidden_gradients[j] * input[i];
        }
    }
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->bias_h[i] += learning_rate * hidden_gradients[i];
    }

    // 释放临时内存
    free(hidden);
    free(output);
    free(output_errors);
    free(output_gradients);
    free(hidden_errors);
    free(hidden_gradients);
}