@echo off
REM 构建安全模式测试程序的批处理脚本

echo === Building robin-map Safe Mode Test ===

REM 创建构建目录
if not exist build mkdir build
cd build

REM 运行 CMake 配置
echo Running CMake configuration...
cmake .. -DTSL_ROBIN_MAP_BUILD_TESTS=ON

REM 编译测试程序
echo Compiling test program...
cmake --build . --config Release

REM 运行测试程序
echo Running test program...
Release\test_safe_mode.exe

REM 检查测试结果
if %errorlevel% equ 0 (
    echo.
    echo === All tests passed successfully ===
) else (
    echo.
    echo === Tests failed ===
    exit /b 1
)

REM 清理构建目录（可选）
REM echo Cleaning up build directory...
REM cd ..
REM rmdir /s /q build

exit /b 0
