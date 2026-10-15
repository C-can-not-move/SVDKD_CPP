# ==================== 全局设置（必须在 add_executable 之前）====================

# ---------------- 推荐cmake版本 ----------------
cmake_minimum_required (VERSION 3.10)

# ---------------- 项目信息 ----------------
project ("SVDKD_C++")

# ---------------- C++ 标准 ----------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- 依赖库 ----------------
# libtorch
# 找到包
set(LIBTORCH_ROOT "D:/DATA/C++_lib/libtorch_cu126")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)
message(STATUS "libtorch: ${TORCH_INSTALL_PREFIX}   (${Torch_VERSION})")

# ==================== 目标 ====================

# ---------------- 源文件 ----------------
# 加新的 .cpp 就在这里加一行。CMake 不会自动扫目录。
set(APP_SOURCES
    "src/SVDKD_C++.h"
    "src/SVDKD_C++.cpp"
    "src/config.cpp" 
    "src/config.h"
    "src/data/cifar10.h"
    "src/data/cifar10.cpp"  
    "src/model/resnet.h" 
    "src/model/resnet.cpp" 
)
add_executable(SVDKD_C++ ${APP_SOURCES})

# ==================== target 属性（必须在 add_executable 之后）====================

# ---------------- 链接库 ----------------
target_include_directories(SVDKD_C++ PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(SVDKD_C++ PRIVATE torch)

# ---------------- 编译器选项 ----------------

if(MSVC)
  target_compile_options(SVDKD_C++ PRIVATE /bigobj /EHsc /wd4267 /wd4244 /utf-8)
  target_compile_definitions(SVDKD_C++ PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------- 运行期依赖：拷 DLL ---------------- 
if(WIN32)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET SVDKD_C++ POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} "$<TARGET_FILE_DIR:SVDKD_C++>")
endif()


