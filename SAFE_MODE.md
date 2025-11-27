# robin-map 安全模式使用指南

## 概述

robin-map 库提供了一个安全模式，可以在运行时检测和防止常见的错误，如空指针访问、边界越界和数据完整性问题。安全模式通过宏 `TSL_ROBIN_MAP_SAFE_MODE` 启用。

## 功能特性

### 1. 空指针安全防护
- 在哈希表核心操作（插入、查找、删除）中增加关键指针的空值校验
- 对迭代器添加空指针安全访问机制
- 对自定义分配器返回的指针增加空值检查

### 2. 边界安全增强
- 在哈希计算和桶索引映射过程中增加范围检查
- 为哈希表扩容/缩容操作添加边界校验
- 在批量插入/删除操作中增加元素数量的边界检查

### 3. 序列化安全机制
- 在序列化过程中添加数据长度、校验和计算
- 对反序列化输入流增加边界检查
- 为反序列化过程添加类型兼容性检查

### 4. 异常处理优化
- 为内存分配失败场景添加异常处理
- 实现哈希表操作的事务性语义
- 为关键操作添加详细的错误信息记录

## 使用方法

### 启用安全模式

要启用安全模式，只需在编译时定义宏 `TSL_ROBIN_MAP_SAFE_MODE`：

```cmake
# 在 CMake 中
add_definitions(-DTSL_ROBIN_MAP_SAFE_MODE)

# 或者在目标级别
target_compile_definitions(your_target PRIVATE TSL_ROBIN_MAP_SAFE_MODE)
```

```cpp
// 在代码中直接定义
#define TSL_ROBIN_MAP_SAFE_MODE
#include "tsl/robin_map.h"
```

### 编译测试程序

```bash
mkdir build
cd build
cmake .. -DTSL_ROBIN_MAP_BUILD_TESTS=ON
make
./test_safe_mode
```

## 异常类型

安全模式下可能抛出以下异常：

- `tsl::rh::allocation_error`：内存分配失败
- `tsl::rh::serialization_error`：序列化/反序列化错误
- `tsl::rh::invalid_iterator_error`：无效迭代器访问
- `std::runtime_error`：其他运行时错误

## 性能影响

启用安全模式会带来一定的性能开销，主要来自：
- 额外的空指针和边界检查
- 数据完整性校验
- 异常处理机制

在性能敏感的场景中，可以通过不定义 `TSL_ROBIN_MAP_SAFE_MODE` 宏来关闭安全模式。

## 测试覆盖

安全模式提供了全面的测试覆盖，包括：
- 空指针访问测试
- 迭代器安全测试
- 边界越界测试
- 序列化完整性测试
- 异常处理测试

## 兼容性

安全模式确保：
- 与 C++17 及以上标准兼容
- 不破坏现有 API
- 可以与现有代码无缝集成

## 示例代码

```cpp
#include <iostream>
#include <tsl/robin_map.h>

#define TSL_ROBIN_MAP_SAFE_MODE

int main() {
    try {
        tsl::robin_map<int, std::string> map;
        
        // 插入元素
        map.insert({1, "one"});
        map.insert({2, "two"});
        map.insert({3, "three"});
        
        // 查找元素
        auto it = map.find(2);
        if (it != map.end()) {
            std::cout << "Found: " << it->second << std::endl;
        }
        
        // 删除元素
        map.erase(1);
        
        // 序列化
        std::string serialized;
        map.serialize(serialized);
        
        // 反序列化
        tsl::robin_map<int, std::string> map2;
        map2.deserialize(serialized);
        
        std::cout << "Map size after deserialization: " << map2.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```
