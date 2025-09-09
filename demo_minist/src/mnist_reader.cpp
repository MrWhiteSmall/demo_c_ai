#include "mnist_reader.hpp"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

// 加载MNIST数据集的函数实现
vector<pair<Mat, int>> MNISTReader::load_mnist(const string& image_path, const string& label_path) {
    // 打开MNIST图像和标签文件（二进制模式）
    ifstream images(image_path, ios::binary);
    ifstream labels(label_path, ios::binary);
    
    // 检查文件是否成功打开
    if (!images.is_open() || !labels.is_open()) {
        cerr << "Error opening MNIST files!" << endl;
        exit(1);
    }

    // 读取图像文件头部信息
    int magic_number, num_images, rows, cols;
    images.read((char*)&magic_number, sizeof(magic_number)); // 魔数
    images.read((char*)&num_images, sizeof(num_images));     // 图像数量
    images.read((char*)&rows, sizeof(rows));                 // 行数
    images.read((char*)&cols, sizeof(cols));                 // 列数
    
    // MNIST文件为大端格式，需要转换为主机字节序
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    // 读取标签文件头部信息
    int label_magic, num_labels;
    labels.read((char*)&label_magic, sizeof(label_magic)); // 标签魔数
    labels.read((char*)&num_labels, sizeof(num_labels));   // 标签数量
    label_magic = __builtin_bswap32(label_magic);
    num_labels = __builtin_bswap32(num_labels);

    vector<pair<Mat, int>> dataset; // 用于存储图像和标签对

    // 逐个读取图像和对应的标签
    for (int i = 0; i < num_images; ++i) {
        // 创建一个单通道浮点型图像用于存储当前图片
        Mat image(rows, cols, CV_32FC1);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel;
                images.read((char*)&pixel, sizeof(pixel)); // 读取一个像素
                image.at<float>(r, c) = pixel / 255.0f;    // 归一化到[0,1]
            }
        }

        // 读取当前图像对应的标签
        unsigned char label;
        labels.read((char*)&label, sizeof(label));
        
        // 将图像和标签组成pair加入数据集
        dataset.emplace_back(image, label);
    }

    // 返回完整的数据集
    return dataset;
}