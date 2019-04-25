// model_in_C++.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <Python.h>
#include <windows.h>
using namespace std;

void testImage(char *path)
{
	try {
		//Py_SetPath(L"D:/software/Anoconda/Lib");

		Py_SetPythonHome(L"D:/software/Anoconda");
		Py_Initialize();
		PyEval_InitThreads();

		if (!Py_IsInitialized())
		{
			cout << "no" << endl;
		}

		PyObject *pFunc = NULL;
		PyObject *pArg = NULL;
		PyObject *module = NULL;

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('./')");

		module = PyImport_ImportModule("model_test");      // 用于测试模型的 Python 文件
		if (!module)             // 如果不能正确导入该模块
		{
			cout << "Can't open module!" << endl;
			Py_Finalize();
			return;
		}

		// 从 module 模块中导入函数名为 recognize 的函数
		pFunc = PyObject_GetAttrString(module, "recognize");
		if (!pFunc)
		{
			cout << "Can't open FUNC!" << endl;
			Py_Finalize();
		}

		pArg = Py_BuildValue("(s)", path);

		if (module != NULL)
		{
			PyGILState_STATE gstate;
			gstate = PyGILState_Ensure();
			PyEval_CallObject(pFunc, pArg);
			PyGILState_Release(gstate);
		}
	}
	catch (exception& e)
	{
		cout << "Standard exception:" << e.what() << endl;
	}
}

int main()
{
	testImage("test.jpg");        // 这里的路径应该是相对于python脚本的路径
	system("pause");
    return 0;
}

