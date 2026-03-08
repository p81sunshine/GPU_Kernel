# GPU Kernel 项目

一个结构化的 CUDA kernel 开发项目，每个 kernel 都有独立的目录。

## 项目结构

```
GPU_Kernel/
├── include/              # 共享头文件
│   └── utils.hpp        # CUDA 工具函数和宏
├── vector_add/          # 向量加法 kernel
│   └── vector_add.cu
├── build/               # 构建输出目录（由 CMake 生成）
├── CMakeLists.txt       # CMake 构建配置
├── .clangd              # clangd 静态分析配置
├── .gitignore          # Git 忽略文件
└── README.md           # 本文件
```

## 添加新的 Kernel

1. 在项目根目录创建新的文件夹，例如 `matrix_multiply/`
2. 在该文件夹中创建对应的 `.cu` 文件，例如 `matrix_multiply.cu`
3. 在 `CMakeLists.txt` 中添加：
   ```cmake
   add_kernel_executable(matrix_multiply)
   ```

## 构建项目

```bash
mkdir -p build
cd build
cmake ..
make
```

可执行文件将生成在 `build/bin/` 目录下。

## 运行

```bash
cd build/bin
./vector_add
```

## clangd 配置（VSCode 跳转支持）

项目已配置 `.clangd` 文件以支持 CUDA 代码的静态分析和跳转功能。clangd 会自动：
- 识别 CUDA 语法
- 提供代码补全
- 进行错误检查
- **支持跳转到定义**（F12 或 Ctrl+点击）
- **支持查找所有引用**（Shift+F12）
- **支持符号搜索**（Ctrl+Shift+O）

### 使用跳转功能

在 VSCode 中：
1. **跳转到定义**：将光标放在符号上（如 `CUDA_CHECK`），按 `F12` 或 `Ctrl+点击`
2. **查找引用**：按 `Shift+F12` 查看所有使用该符号的地方
3. **返回**：按 `Alt+←` 返回上一个位置

### 如果跳转不工作

1. **最佳方式**：运行 CMake 生成 `compile_commands.json`（clangd 会优先使用）
   ```bash
   mkdir -p build && cd build && cmake ..
   ```
   这会在 `build/` 目录生成 `compile_commands.json`，clangd 会自动找到它

2. **备选方式**：项目已包含 `compile_flags.txt` 作为备选配置

3. **确保 CUDA 路径正确**：如果 CUDA 不在 `/usr/local/cuda`，需要修改 `.clangd` 中的 `--cuda-path` 参数

4. **重启 clangd**：在 VSCode 中按 `Ctrl+Shift+P`，输入 "clangd: Restart language server"

## 依赖

- CMake 3.18+
- CUDA Toolkit
- C++17 兼容的编译器

