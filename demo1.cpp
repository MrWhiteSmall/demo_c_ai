#include <iostream>
#include <csignal>  // 用于触发调试器

// void debug_breakpoint() {
//     #if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
//         __builtin_trap();  // GCC/Clang内置函数
//     #else
//         raise(SIGTRAP);
//     #endif
// }

int main() {
    int x = 10;
    std::string message = "Hello Debug";
     
    std::cout << "Before breakpoint" << std::endl;
    
    // 在这里设置断点
    // debug_breakpoint();  // ← 程序运行到这里会暂停
    
    std::cout << "After breakpoint" << std::endl;
    std::cout << "x = " << x << ", message = " << message << std::endl;
    
    return 0;
}