#include "mnist_reader.hpp"
#include "neural_net.hpp"
#include <iostream>

int main() {
    // 加载数据
    auto train_data = MNISTReader::load_mnist("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
    
    // 初始化网络 (784输入, 128隐藏层, 10输出)
    NeuralNet net(28 * 28, 128, 10);
    
    // 训练
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    for (const auto& [img, label] : train_data) {
        images.push_back(img);
        labels.push_back(label);
    }
    net.train(images, labels, 10);
    
    // 测试
    auto test_data = MNISTReader::load_mnist("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");
    int correct = 0;
    for (const auto& [img, label] : test_data) {
        // 前向传播，获取网络对当前图片的输出（概率分布）
        auto output = net.forward(img);
        // 找到概率最大的类别作为预测结果
        int predicted = std::max_element(output.begin(), output.end()) - output.begin();
        // 如果预测正确，则计数加一
        if (predicted == label) correct++;
    }
    std::cout << "Accuracy: " << 100.0 * correct / test_data.size() << "%\n";
    return 0;
}