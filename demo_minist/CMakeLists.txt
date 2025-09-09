# 最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(demo_minist)

# 设置C++标准为C++11
set(CMAKE_CXX_STANDARD 17)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(include)

# 搜索src目录下所有cpp源文件
file(GLOB SOURCES "src/*.cpp")

# 生成可执行文件
add_executable(mnist_cpp ${SOURCES})

# 链接OpenCV库
target_link_libraries(mnist_cpp ${OpenCV_LIBS})