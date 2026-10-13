# CMakeList.txt: SVDKD_C++ 的 CMake 项目，在此处包括源代码并定义项目特定的逻辑。

# 推荐版本
cmake_minimum_required (VERSION 3.10)

# ---------------- C++ 标准（必须在 add_executable 之前）----------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
# 
if(MSVC)
  target_compile_options(SVDKD_C++ PRIVATE /utf-8)
endif()


project ("SVDKD_C++")

# ---------------- 源文件 ----------------
# 加新的 .cpp 就在这里加一行。CMake 不会自动扫目录。
set(APP_SOURCES
    "src/SVDKD_C++.cpp"
    "src/SVDKD_C++.h"
)
add_executable(SVDKD_C++ ${APP_SOURCES})



# 库

# ---------------- libtorch ----------------
# 找到包
set(LIBTORCH_ROOT "D:/DATA/C++_lib/libtorch_cu126")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)

message(STATUS "libtorch: ${TORCH_INSTALL_PREFIX}   (${Torch_VERSION})")

# 链接 
target_include_directories(SVDKD_C++ PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(SVDKD_C++ PRIVATE torch)

if(MSVC)
  target_compile_options(SVDKD_C++ PRIVATE /bigobj /EHsc /wd4267 /wd4244)
  target_compile_definitions(SVDKD_C++ PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

#  Windows: 运行期把 DLL 拷到 exe 旁边 
if(WIN32)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET SVDKD_C++ POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} "$<TARGET_FILE_DIR:SVDKD_C++>")
endif()