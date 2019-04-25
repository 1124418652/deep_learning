// model_in_C++.cpp : �������̨Ӧ�ó������ڵ㡣
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

		module = PyImport_ImportModule("model_test");      // ���ڲ���ģ�͵� Python �ļ�
		if (!module)             // ���������ȷ�����ģ��
		{
			cout << "Can't open module!" << endl;
			Py_Finalize();
			return;
		}

		// �� module ģ���е��뺯����Ϊ recognize �ĺ���
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
	testImage("test.jpg");        // �����·��Ӧ���������python�ű���·��
	system("pause");
    return 0;
}

