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

inline void checkErr(cl_int err, const char *msg, bool ex)
{
	if (err != CL_SUCCESS) {
		std::cout << "Error(" << err << "): " << msg << std::endl;
		if (ex)
			exit(EXIT_FAILURE);
	}
}

int main(int argc, char *argv[])
{
	cl_int err;
	cl_uint plat_num;
	cl_platform_id *plat_ids; // alloca()

	cl_uint dev_num;
	cl_device_id *dev_ids; // alloca()

	cl_context ctx;

	cl_command_queue cmdq;

	err = clGetPlatformIDs(0, NULL, &plat_num);
	checkErr((err != CL_SUCCESS) ? err : (plat_num < 1 ? -1 : CL_SUCCESS),
			"failed to get valid platform number!", true);

	plat_ids = (cl_platform_id *)alloca(sizeof(cl_platform_id) * plat_num);
	err = clGetPlatformIDs(plat_num, plat_ids, NULL);
	checkErr(err, "failed to get platform IDs!", true);

	std::cout << "Platform Number: " << plat_num << std::endl;

	/* Use the first available platform */
	err = clGetDeviceIDs(plat_ids[0], CL_DEVICE_TYPE_ALL, 0, NULL, &dev_num);
	checkErr((err != CL_SUCCESS) ? err : (dev_num < 1 ? -1 : CL_SUCCESS),
			"failed to get valid ALL device number!", true);
	std::cout << "Device Number: " << dev_num << std::endl;

	err = clGetDeviceIDs(plat_ids[0], CL_DEVICE_TYPE_GPU, 0, NULL, &dev_num);
	checkErr((err != CL_SUCCESS) ? err : (dev_num < 1 ? -1 : CL_SUCCESS),
			"failed to get valid GPU device number!", true);
	std::cout << "Device(GPU) Number: " << dev_num << std::endl;

	/* Get the first GPU device */
	dev_ids = (cl_device_id *)alloca(sizeof(cl_device_id) * dev_num);
	err = clGetDeviceIDs(plat_ids[0], CL_DEVICE_TYPE_GPU, 1, dev_ids, NULL);
	checkErr(err, "failed to get device IDs!", true);

	cl_context_properties ctx_pro[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)plat_ids[0],
		0
	};
	ctx = clCreateContext(ctx_pro, dev_num, dev_ids, NULL, NULL, &err);
	checkErr(err, "failed to create context!", true);

	/* Deprecated in OCL 2.0 */
	cmdq = clCreateCommandQueue(ctx, dev_ids[0], 0, &err);
	checkErr(err, "failed to create command queue!", true);

	int result[ARRAY_SIZE];
	int input[ARRAY_SIZE];
	cl_mem mem_obj[2];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		input[i] = i;
	}

	mem_obj[0] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_PTR,
			sizeof(int) * ARRAY_SIZE, input, &err);
	checkErr(err, "failed to create buffer 0!", true);
	mem_obj[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			sizeof(int) * ARRAY_SIZE, NULL, &err);
	checkErr(err, "failed to create buffer 1!", true);

	cl_program prog;

	std::ifstream kern_file("memory.cl", std::ios::in);
	if (!kern_file.is_open()) {
		std::cerr << "failed to open file: memory.cl" << std::endl;
		return -1;
	}
	std::ostringstream oss;
	oss << kern_file.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();

	// only one string for program source code
	prog = clCreateProgramWithSource(ctx, 1, (const char**)&srcStr, NULL, &err);
	checkErr(err, "failed to create program!", true);
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		char buildLog[16384];
		clGetProgramBuildInfo(prog, dev_ids[0], CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(prog);
		return -1;
	}

	cl_kernel kernel;
	kernel = clCreateKernel(prog, "mem_obj", &err);
	checkErr(err, "failed to create kernel!", false);
	// do cleanup

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_obj[0]);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_obj[1]);
	checkErr(err, "failed to set kernel arguments!", false);
	// do cleanup

	const size_t globalWorkSize[1] = { ARRAY_SIZE };
	const size_t localWorkSize[1] = { 1 };

	err = clEnqueueNDRangeKernel(cmdq, kernel, 1, NULL, globalWorkSize,
			localWorkSize, 0, NULL, NULL);
	checkErr(err, "failed to enqueue kernel!", false);
	// do cleanup

	return 0;
}
