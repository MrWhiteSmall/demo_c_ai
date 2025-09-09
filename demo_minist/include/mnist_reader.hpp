#pragma once  // 防止头文件被多次包含
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class MNISTReader {
public:
    static std::vector<std::pair<cv::Mat, int>> load_mnist( const std::string& image_path, 
                                                            const std::string& label_path);
};