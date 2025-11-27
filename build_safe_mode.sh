#!/bin/bash

# 构建安全模式测试程序的脚本

echo "=== Building robin-map Safe Mode Test ==="

# 创建构建目录
mkdir -p build
cd build

# 运行 CMake 配置
echo "Running CMake configuration..."
cmake .. -DTSL_ROBIN_MAP_BUILD_TESTS=ON

# 编译测试程序
echo "Compiling test program..."
make -j$(nproc)

# 运行测试程序
echo "Running test program..."
./test_safe_mode

# 检查测试结果
if [ $? -eq 0 ]; then
    echo "\n=== All tests passed successfully ==="
else
    echo "\n=== Tests failed ==="
    exit 1
fi

# 清理构建目录（可选）
# echo "Cleaning up build directory..."
# cd ..
# rm -rf build

exit 0
