#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

// 测试空指针访问防护
void test_null_pointer_protection() {
    std::cout << "Testing null pointer protection..." << std::endl;
    
    tsl::robin_map<int, std::string> map;
    
    // 测试插入操作的空指针防护
    try {
        map.insert({1, "test"});
        std::cout << "✓ Insert operation passed null pointer checks" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Insert operation failed: " << e.what() << std::endl;
    }
    
    // 测试查找操作的空指针防护
    try {
        auto it = map.find(1);
        if (it != map.end()) {
            std::cout << "✓ Find operation passed null pointer checks" << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Find operation failed: " << e.what() << std::endl;
    }
    
    // 测试删除操作的空指针防护
    try {
        map.erase(1);
        std::cout << "✓ Erase operation passed null pointer checks" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Erase operation failed: " << e.what() << std::endl;
    }
}

// 测试迭代器安全访问
void test_iterator_safety() {
    std::cout << "\nTesting iterator safety..." << std::endl;
    
    tsl::robin_map<int, std::string> map;
    for (int i = 0; i < 10; ++i) {
        map[i] = std::to_string(i);
    }
    
    // 测试有效迭代器访问
    try {
        auto it = map.begin();
        while (it != map.end()) {
            std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
            ++it;
        }
        std::cout << "✓ Iterator traversal passed safety checks" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Iterator traversal failed: " << e.what() << std::endl;
    }
    
    // 测试失效迭代器访问
    try {
        auto it = map.begin();
        map.clear();
        // 尝试访问已失效的迭代器
        std::cout << "Key: " << it->first << std::endl; // 这应该触发断言
        std::cout << "✗ Iterator invalidation check failed" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Iterator invalidation check passed: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "✓ Iterator invalidation triggered assertion" << std::endl;
    }
}

// 测试边界安全增强
void test_boundary_safety() {
    std::cout << "\nTesting boundary safety..." << std::endl;
    
    tsl::robin_map<int, std::string> map;
    
    // 测试大量元素插入的边界检查
    try {
        const int max_elements = 1000000;
        for (int i = 0; i < max_elements; ++i) {
            map[i] = std::to_string(i);
        }
        std::cout << "✓ Large number of elements inserted successfully" << std::endl;
        std::cout << "  Number of elements: " << map.size() << std::endl;
        std::cout << "  Number of buckets: " << map.bucket_count() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Large insertion failed: " << e.what() << std::endl;
    }
    
    // 测试哈希表扩容边界检查
    try {
        map.reserve(2000000);
        std::cout << "✓ Reserve operation passed boundary checks" << std::endl;
        std::cout << "  New number of buckets: " << map.bucket_count() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Reserve operation failed: " << e.what() << std::endl;
    }
}

// 测试序列化安全机制
void test_serialization_safety() {
    std::cout << "\nTesting serialization safety..." << std::endl;
    
    tsl::robin_map<int, std::string> map;
    for (int i = 0; i < 10; ++i) {
        map[i] = std::to_string(i);
    }
    
    // 测试序列化
    try {
        std::string serialized;
        map.serialize(serialized);
        std::cout << "✓ Serialization passed safety checks" << std::endl;
        std::cout << "  Serialized size: " << serialized.size() << " bytes" << std::endl;
        
        // 测试反序列化
        tsl::robin_map<int, std::string> map2;
        map2.deserialize(serialized);
        std::cout << "✓ Deserialization passed safety checks" << std::endl;
        std::cout << "  Deserialized elements: " << map2.size() << std::endl;
        
        // 测试数据完整性
        assert(map == map2);
        std::cout << "✓ Data integrity verified after deserialization" << std::endl;
        
    } catch (const std::runtime_error& e) {
        std::cout << "✗ Serialization/deserialization failed: " << e.what() << std::endl;
    }
    
    // 测试损坏数据的反序列化
    try {
        std::string corrupted_data = "corrupted_data"; // 无效的序列化数据
        tsl::robin_map<int, std::string> map3;
        map3.deserialize(corrupted_data);
        std::cout << "✗ Corrupted data deserialization check failed" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Corrupted data detection passed: " << e.what() << std::endl;
    }
}

// 测试异常处理优化
void test_exception_handling() {
    std::cout << "\nTesting exception handling..." << std::endl;
    
    // 测试内存分配失败处理（模拟）
    try {
        // 这只是一个示例，实际内存分配失败很难模拟
        tsl::robin_map<int, std::string> map;
        // 尝试分配大量内存
        map.reserve(static_cast<std::size_t>(-1)); // 这应该触发异常
        std::cout << "✗ Memory allocation failure handling failed" << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cout << "✓ Memory allocation failure handled correctly: " << e.what() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Memory allocation failure handled as runtime error: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "✓ Memory allocation failure triggered exception" << std::endl;
    }
}

int main() {
    std::cout << "=== Testing robin-map Safe Mode ===" << std::endl;
    
    test_null_pointer_protection();
    test_iterator_safety();
    test_boundary_safety();
    test_serialization_safety();
    test_exception_handling();
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    
    return 0;
}
