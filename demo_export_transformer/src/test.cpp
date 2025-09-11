#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3}).cuda();  // 测试CUDA张量
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << tensor << std::endl;
    return 0;
}