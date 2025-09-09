#ifndef NN_H
#define NN_H

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *weight_ih;
    double *weight_ho;
    double *bias_h;
    double *bias_o;
} NeuralNetwork;

NeuralNetwork* create_nn(int input_size, int hidden_size, int output_size);
void free_nn(NeuralNetwork *nn);
double* predict(NeuralNetwork *nn, double *input); 
void train(NeuralNetwork *nn, double *input, double *target, double learning_rate);

#endif