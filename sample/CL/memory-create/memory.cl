__kernel void memory_obj(__global int *input, __global int *result)
{
	int gid = get_global_id(0);

	result[gid] = input[gid] * input[gid];
	input[gid] = input[gid] * input[gid];
}
