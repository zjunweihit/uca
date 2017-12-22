#include <iostream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

inline void checkErr(cl_int err, const char *msg)
{
	if (err != CL_SUCCESS) {
		std::cout << "Error(" << err << "): " << msg << std::endl;
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char *argv[])
{
	cl_int ret;
	cl_uint plat_num;
	cl_platform_id *plat_ids;

	ret = clGetPlatformIDs(0, NULL, &plat_num);
	checkErr((ret != CL_SUCCESS) ? ret : (plat_num <= 0 ? -1 : CL_SUCCESS),
			"failed to get valid platform number!");

	plat_ids = (cl_platform_id *)alloca(sizeof(cl_platform_id) * plat_num);
	ret = clGetPlatformIDs(plat_num, plat_ids, NULL);
	checkErr(ret, "failed to get platform IDs!");
	std::cout << "Platform Number: " << plat_num << std::endl;

	return 0;
}
