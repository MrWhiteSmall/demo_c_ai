#include <stdlib.h>
#include <cstdio>
#include "nn.h"

int main(){
    NeuralNetwork *nn = create_nn(2, 6, 1);

    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets[4] = {0, 1, 1, 0};

    for (int epoch = 0; epoch < 10000; epoch++){
        for (int i = 0; i < 4; i++){
            train(nn, inputs[i], &targets[i], 0.1);
        }
    }

    for (int i = 0; i < 4; i++){
        double *output = predict(nn, inputs[i]);
        printf("Input [%f, %f], output %f \n", inputs[i][0], inputs[i][1], output[0]);
        free(output);
    }

    free_nn(nn);
    return 0;
}