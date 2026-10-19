# ==================== 全局设置（必须在 add_executable 之前）====================

# ---------------- 推荐cmake版本 ----------------
cmake_minimum_required (VERSION 4.1)


# ---------------- 项目信息 ----------------
project ("SVDKD_C++")

# ---------------- C++ 标准 ----------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- 依赖库 ----------------
# libtorch
# 找到包

if(WIN32)
    set(LIBTORCH_ROOT "D:/DATA/C++_lib/libtorch_cu126")
else()
    set(LIBTORCH_ROOT "/home/chenzhijiang/userFile/C++_lib/libtorch")
    set(CUDA_TOOLKIT_ROOT_DIR "/home/chenzhijiang/userFile/miniConda3/envs/cuda126")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)
message(STATUS "libtorch: ${TORCH_INSTALL_PREFIX}   (${Torch_VERSION})")
message(STATUS "TORCH_CXX_FLAGS = [${TORCH_CXX_FLAGS}]")


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
    "src/data/transforms.h"
    "src/data/transforms.cpp"
    "src/model/resnet.h" 
    "src/model/resnet.cpp"
    "src/model/checkpoint.cpp" 
    "src/model/checkpoint.h"
    "src/kd/my_svd.h"
    "src/kd/my_svd.cpp"
    "src/kd/loss.h"
    "src/kd/loss.cpp"
    "src/train/trainer.h"
    "src/train/trainer.cpp")
add_executable(SVDKD_C++ ${APP_SOURCES})

# ==================== target 属性（必须在 add_executable 之后）====================

# ---------------- libtorch 带来的编译参数（仅 Linux）----------------
# 不要写进 CMAKE_CXX_FLAGS —— 那是【全局】变量，会连带污染 CMake 内部的
#   功能探测 try-compile，报错出现在 CMakeFiles/CMakeTmp/CMakeLists.txt 里，
#   跟你的代码毫无关系。
#   用 target_compile_options：只影响这一个目标。
if(NOT WIN32)
    separate_arguments(_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(SVDKD_C++ PRIVATE ${_TORCH_CXX_FLAGS})
endif()

# ---------------- 链接库 ----------------
target_include_directories(SVDKD_C++ PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(SVDKD_C++ PRIVATE torch)

# ---------------- 编译器选项 ----------------
# 仅 MSVC。Linux（GCC/Clang）下整块自动跳过 —— 上面五项都不是必需的：
#   /bigobj  ELF 没有段数限制 ／ /EHsc  GCC 默认 -fexceptions
#   /wd4267 /wd4244  对应的 -Wconversion 默认不开
#   /utf-8  GCC/Clang 默认按 UTF-8 读源文件（这条是 MSVC 的问题，不是平台问题）
#   NOMINMAX / WIN32_LEAN_AND_MEAN  都是 <windows.h> 的
if(MSVC)
  target_compile_options(SVDKD_C++ PRIVATE /bigobj /EHsc /wd4267 /wd4244 /utf-8)
  target_compile_definitions(SVDKD_C++ PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------- 运行期依赖：WIN32 拷 DLL ／ Linux 写 RPATH ----------------
if(WIN32)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET SVDKD_C++ POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} "$<TARGET_FILE_DIR:SVDKD_C++>")
else()
  # Linux：不拷（libtorch 的 .so 有几个 GB），直接写进 RPATH
  set_target_properties(SVDKD_C++ PROPERTIES
    BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")

endif()


