#include <torch/script.h>
#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>  // 添加这行
#include <torch/cuda.h>  // 必须包含这个头文件才能使用 torch::cuda::synchronize()

// 解析命令行参数，支持 -g <gpu_id> 设置 CUDA_VISIBLE_DEVICES
void set_cuda_visible_device_from_args(int argc, char* argv[]) {
    int gpu_id = -1;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "-g") {
            gpu_id = std::stoi(argv[i + 1]);
            break;
        }
    }
    if (gpu_id >= 0) {
        std::string cuda_env = std::to_string(gpu_id);
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", cuda_env.c_str());
#else
        setenv("CUDA_VISIBLE_DEVICES", cuda_env.c_str(), 1);
#endif
    }
}

// 辅助函数处理模块输出
torch::Tensor safe_to_tensor(const torch::jit::IValue& ivalue, const std::string& module_name) {
    if (ivalue.isTensor()) {
        return ivalue.toTensor();
    } else if (ivalue.isTuple()) {
        auto tuple = ivalue.toTuple();
        if (tuple->elements().size() > 0 && tuple->elements()[0].isTensor()) {
            return tuple->elements()[0].toTensor();
        }
    }
    throw std::runtime_error(module_name + " did not return a Tensor");
}

class FluxTransformer {
private:
    torch::jit::script::Module pos_embed;
    torch::jit::script::Module time_text_embed;
    torch::jit::script::Module context_embedder;
    torch::jit::script::Module x_embedder;
    std::vector<torch::jit::script::Module> transformer_blocks;
    std::vector<torch::jit::script::Module> single_transformer_blocks;
    torch::jit::script::Module norm_out;
    torch::jit::script::Module proj_out;
    
public:
    FluxTransformer(const std::string& parts_dir) {
        // 加载所有组件
        try {
            pos_embed = torch::jit::load(parts_dir + "/pos_embed.pt");
            time_text_embed = torch::jit::load(parts_dir + "/time_text_embed.pt");
            context_embedder = torch::jit::load(parts_dir + "/context_embedder.pt");
            x_embedder = torch::jit::load(parts_dir + "/x_embedder.pt");
            
            // 加载transformer blocks
            int i = 0;
            while (true) {
                std::string path = parts_dir + "/transformer_block_" + std::to_string(i) + ".pt";
                if (access(path.c_str(), F_OK) == -1) break;
                transformer_blocks.push_back(torch::jit::load(path));
                i++;
            }
            
            // 加载single transformer blocks
            i = 0;
            while (true) {
                std::string path = parts_dir + "/single_block_" + std::to_string(i) + ".pt";
                if (access(path.c_str(), F_OK) == -1) break;
                single_transformer_blocks.push_back(torch::jit::load(path));
                i++;
            }
            
            // 加载输出部分
            norm_out = torch::jit::load(parts_dir + "/norm.pt");
            proj_out = torch::jit::load(parts_dir + "/proj.pt");
            
            // 将所有模块移到GPU
            pos_embed.to(torch::kCUDA);
            time_text_embed.to(torch::kCUDA);
            context_embedder.to(torch::kCUDA);
            x_embedder.to(torch::kCUDA);
            for (auto& block : transformer_blocks) block.to(torch::kCUDA);
            for (auto& block : single_transformer_blocks) block.to(torch::kCUDA);
            norm_out.to(torch::kCUDA);
            proj_out.to(torch::kCUDA);
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }
    
    torch::Tensor forward(
        torch::Tensor hidden_states,
        torch::Tensor encoder_hidden_states,
        torch::Tensor pooled_projections,
        torch::Tensor timestep,
        torch::Tensor img_ids,
        torch::Tensor txt_ids,
        torch::Tensor guidance = torch::Tensor()
    ) {
        try {
            // 确保所有输入在GPU上
            hidden_states = hidden_states.to(torch::kCUDA);
            encoder_hidden_states = encoder_hidden_states.to(torch::kCUDA);
            pooled_projections = pooled_projections.to(torch::kCUDA);
            timestep = timestep.to(torch::kCUDA);
            img_ids = img_ids.to(torch::kCUDA);
            txt_ids = txt_ids.to(torch::kCUDA);
            guidance = guidance.to(torch::kCUDA);
            
            // 前向传播各组件
            // hidden_states = x_embedder.forward({hidden_states}).toTensor();
            hidden_states = safe_to_tensor(x_embedder.forward({hidden_states}), "x_embedder");
            
            timestep = timestep.to(hidden_states.dtype()) * 1000;
            guidance = guidance.to(hidden_states.dtype()) * 1000;
            
            torch::Tensor temb;
            // if (guidance.defined()) {
            //     temb = time_text_embed.forward({timestep, guidance, pooled_projections}).toTensor();
            // } else {
            //     throw std::runtime_error("Guidance is required in this implementation");
            // }
            temb = safe_to_tensor(time_text_embed.forward({timestep, guidance, pooled_projections}), "time_text_embed");
            
            // encoder_hidden_states = context_embedder.forward({encoder_hidden_states}).toTensor();
            encoder_hidden_states = safe_to_tensor(context_embedder.forward({encoder_hidden_states}), "context_embedder");
            
            // 处理ids
            auto ids = torch::cat({txt_ids, img_ids}, 0);
            auto image_rotary_emb = pos_embed.forward({ids});
            
            // transformer blocks
            for (auto& block : transformer_blocks) {
                auto block_output = block.forward({
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb
                }).toTuple();
                
                encoder_hidden_states = block_output->elements()[0].toTensor();
                hidden_states = block_output->elements()[1].toTensor();
            }
            
            // single transformer blocks
            for (auto& block : single_transformer_blocks) {
                auto block_output = block.forward({
                    hidden_states, encoder_hidden_states, temb, image_rotary_emb
                }).toTuple();
                
                encoder_hidden_states = block_output->elements()[0].toTensor();
                hidden_states = block_output->elements()[1].toTensor();
            }
            
            // 输出部分
            hidden_states = norm_out.forward({hidden_states, temb}).toTensor();
            auto output = proj_out.forward({hidden_states}).toTensor();
            
            return output;
            
        } catch (const std::exception& e) {
            std::cerr << "Error during forward pass: " << e.what() << std::endl;
            throw;
        }
    }
};

// set_cuda_visible_device_from_args 检查 正确
int main(int argc, char* argv[]) {
    set_cuda_visible_device_from_args(argc, argv);

    try {
        // 初始化CUDA
        torch::Device device(torch::kCUDA);
        
        // 创建transformer
        FluxTransformer transformer("../chunks_exported_transformers");
        
        // 创建测试输入 (形状与Python端一致)
        auto options_float16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
        
        auto hidden_states = torch::randn({1, 4096, 128}, options_float16);
        auto encoder_hidden_states = torch::randn({1, 1241, 4096}, options_float16);
        auto pooled_projections = torch::randn({1, 768}, options_float16);
        auto timestep = torch::tensor({0.5}, options_float16).to(torch::kLong);
        auto img_ids = torch::randint(0, 100, {4096, 3}, options_long);
        auto txt_ids = torch::randint(0, 100, {1241, 3}, options_long);
        auto guidance = torch::tensor({3.5}, options_float16);
        
        // 运行前向传播
        std::cout << torch::GradMode::is_enabled() << std::endl;  // 输出 true 1
        torch::NoGradGuard no_grad;
        std::cout << torch::GradMode::is_enabled() << std::endl;  // 输出 false 0
        
        // 循环10次计算平均每次所需时间
        torch::Tensor output;
        auto start = std::chrono::high_resolution_clock::now();
        int num_iters = 10;
        for (int i = 0; i < num_iters; ++i) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            output = transformer.forward(
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance
            );
            // 如果在CUDA上，确保同步以获得准确计时
            torch::cuda::synchronize();
            auto iter_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> iter_elapsed = iter_end - iter_start;
            std::cout << "Time for iteration " << i << ": " << iter_elapsed.count() << " seconds" << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // Average time per forward: 2.64119 seconds
        std::cout << "Average time per forward: " << (elapsed.count() / num_iters) << " seconds" << std::endl;
        
        std::cout << "Output shape: "; // Output shape: 1 4096 64 
        for (auto s : output.sizes()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}