// window 控制台显示utf-8字符
#ifdef _WIN32
#include <windows.h>
#endif

#include "SVDKD_C++.h"

int  main(int argc, char** argv)
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);   // 65001，输出
	SetConsoleCP(CP_UTF8);         // 输入（读中文时要）
#endif

	std::cout << "SVD-KD C++ demo" << std::endl;
	return 0;
}