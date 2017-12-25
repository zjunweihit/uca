#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const int ARRAY_SIZE = 16;
const int NUM_MEM_OBJ = 2;

inline int checkErr(cl_int err, const char *msg)
{
	if (err != CL_SUCCESS) {
		std::cout << "Error(" << err << "): " << msg << std::endl;
		return err;
	}
	return CL_SUCCESS;
}

void Cleanup(cl_context context,  cl_command_queue commandQueue,
		cl_program program, cl_kernel kernel,
		cl_mem memObjects[NUM_MEM_OBJ])
{
	if (kernel != NULL)
		clReleaseKernel(kernel);

	if (program != NULL)
		clReleaseProgram(program);

	for (int i = 0; i < NUM_MEM_OBJ; i++) {
		if (memObjects[i] != NULL)
			clReleaseMemObject(memObjects[i]);
	}

	if (commandQueue != NULL)
		clReleaseCommandQueue(commandQueue);

	if (context != NULL)
		clReleaseContext(context);
}

int main(int argc, char *argv[])
{
	cl_int err;
	cl_uint plat_num;
	cl_platform_id *plat_ids; // alloca()

	cl_uint dev_num;
	cl_device_id *dev_ids; // alloca()

	cl_context ctx = NULL;
	cl_program prog = NULL;
	cl_kernel kernel = NULL;

	cl_mem mem_obj[NUM_MEM_OBJ] = {0, 0};
	cl_command_queue cmdq = NULL;

	err = clGetPlatformIDs(0, NULL, &plat_num);
	err = checkErr((err != CL_SUCCESS) ? err : (plat_num < 1 ? -1 : CL_SUCCESS),
			"failed to get valid platform number!");
	if (err != CL_SUCCESS)
		return err;

	plat_ids = (cl_platform_id *)alloca(sizeof(cl_platform_id) * plat_num);
	err = clGetPlatformIDs(plat_num, plat_ids, NULL);
	err = checkErr(err, "failed to get platform IDs!");
	if (err != CL_SUCCESS)
		return err;

	std::cout << "Platform Number: " << plat_num << std::endl;

	/* Use the first available platform */
	err = clGetDeviceIDs(plat_ids[0], CL_DEVICE_TYPE_ALL, 0, NULL, &dev_num);
	err = checkErr((err != CL_SUCCESS) ? err : (dev_num < 1 ? -1 : CL_SUCCESS),
			"failed to get valid ALL device number!");
	if (err != CL_SUCCESS)
		return err;
	std::cout << "Device Number: " << dev_num << std::endl;

	err = clGetDeviceIDs(plat_ids[0], CL_DEVICE_TYPE_GPU, 0, NULL, &dev_num);
	err = checkErr((err != CL_SUCCESS) ? err : (dev_num < 1 ? -1 : CL_SUCCESS),
			"failed to get valid GPU device number!");
	if (err != CL_SUCCESS)
		return err;
	std::cout << "Device(GPU) Number: " << dev_num << std::endl;

	/* Get the first GPU device */
	dev_ids = (cl_device_id *)alloca(sizeof(cl_device_id) * dev_num);
	err = clGetDeviceIDs(plat_ids[0], CL_DEVICE_TYPE_GPU, 1, dev_ids, NULL);
	if (checkErr(err, "failed to get device IDs!") != CL_SUCCESS)
		return err;

	cl_context_properties ctx_pro[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)plat_ids[0],
		0
	};
	ctx = clCreateContext(ctx_pro, dev_num, dev_ids, NULL, NULL, &err);
	if (checkErr(err, "failed to create context!") != CL_SUCCESS)
		return err;

	/* Deprecated in OCL 2.0 */
	cmdq = clCreateCommandQueue(ctx, dev_ids[0], 0, &err);
	if (checkErr(err, "failed to create command queue!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	int result[ARRAY_SIZE];
	int input[ARRAY_SIZE];

	for (int i = 0; i < ARRAY_SIZE; i++)
		input[i] = i;

	mem_obj[0] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(int) * ARRAY_SIZE, input, &err);
	if (checkErr(err, "failed to create buffer 0!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		std::cout << "error value: " << err << std::endl;
		return err;
	}
	mem_obj[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			sizeof(int) * ARRAY_SIZE, NULL, &err);
	if (checkErr(err, "failed to create buffer 1!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	std::ifstream kern_file("memory.cl", std::ios::in);
	if (!kern_file.is_open()) {
		std::cerr << "failed to open file: memory.cl" << std::endl;
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return -1;
	}
	std::ostringstream oss;
	oss << kern_file.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();

	/* only one string for program source code */
	prog = clCreateProgramWithSource(ctx, 1, (const char**)&srcStr, NULL, &err);
	if (checkErr(err, "failed to create program!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		char buildLog[16384];
		clGetProgramBuildInfo(prog, dev_ids[0], CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	kernel = clCreateKernel(prog, "memory_obj", &err);
	if (checkErr(err, "failed to create kernel!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj[0]);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_obj[1]);
	if (checkErr(err, "failed to set kernel arguments!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	const size_t globalWorkSize[1] = { ARRAY_SIZE };
	const size_t localWorkSize[1] = { 1 };

	err = clEnqueueNDRangeKernel(cmdq, kernel, 1, NULL, globalWorkSize,
			localWorkSize, 0, NULL, NULL);
	if (checkErr(err, "failed to enqueue kernel!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	err = clEnqueueReadBuffer(cmdq, mem_obj[1], CL_TRUE, 0,
			ARRAY_SIZE * sizeof(int), result, 0, NULL, NULL);
	if (checkErr(err, "failed to enqueue kernel!") != CL_SUCCESS) {
		Cleanup(ctx, cmdq, prog, kernel, mem_obj);
		return err;
	}

	for (int i = 0; i < ARRAY_SIZE; i++)
		std::cout << result[i] << " ";

	std::cout << std::endl;
	std::cout << "Execution is sucessful." << std::endl;

	Cleanup(ctx, cmdq, prog, kernel, mem_obj);
	return 0;
}
