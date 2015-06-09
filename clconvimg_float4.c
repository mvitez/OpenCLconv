#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/time.h>
#include <math.h>
#include <CL/opencl.h>

char *source =
"const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"__kernel void conv9x9(int w, __global image2d_t in, __constant float *ck, __global write_only image2d_t out)\n"
"{\n"
"	int i, x, y, y1, i9;\n"
"	float4 sum;\n"
"\n"
"	x = get_global_id(0);\n"
"	y = get_global_id(1);\n"
"	sum = 0.0f;\n"		
"	for(i = 0; i < 9; i++) {\n"
"		i9 = 9*i;\n"
"		y1 = y+i;\n"
"		sum += read_imagef(in, sampler, (int2)(x, y1)) * ck[i9];\n"
"		sum += read_imagef(in, sampler, (int2)(x+1 ,y1)) * ck[i9+1];\n"
"		sum += read_imagef(in, sampler, (int2)(x+2 ,y1)) * ck[i9+2];\n"
"		sum += read_imagef(in, sampler, (int2)(x+3 ,y1)) * ck[i9+3];\n"
"		sum += read_imagef(in, sampler, (int2)(x+4 ,y1)) * ck[i9+4];\n"
"		sum += read_imagef(in, sampler, (int2)(x+5 ,y1)) * ck[i9+5];\n"
"		sum += read_imagef(in, sampler, (int2)(x+6 ,y1)) * ck[i9+6];\n"
"		sum += read_imagef(in, sampler, (int2)(x+7 ,y1)) * ck[i9+7];\n"
"		sum += read_imagef(in, sampler, (int2)(x+8 ,y1)) * ck[i9+8];\n"
"	}\n"
"	write_imagef(out, (int2)(x,y), sum);\n"
"}\n";
 
int main()
{
	int i,j,k;
	// nb of operations:
	const int dsize = 512;
	int nthreads = 1;
	int nbOfAverages = 1e2;
	int opsMAC = 2; // operations per MAC
	cl_float4 *in, *out;
	cl_float *ck;
	double tops; //total ops

#define NQUEUES 1
	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queues[NQUEUES];
	cl_mem bufin, bufck, bufout;
	cl_event event = NULL;
	cl_program program;
	cl_kernel kernel;
	size_t global[2], local[2];
	size_t param[5];
	char version[300];
  
	// allocate matrices
	
	in = (cl_float4 *) calloc(dsize*dsize, sizeof(*in));
	out = (cl_float4 *) calloc(dsize*dsize, sizeof(*out));
	ck = (cl_float *) calloc(9*9, sizeof(*ck));
	in[0].x = 2.0f;
	in[1].x = 3.0f;
	in[dsize].x = 1.0;
	ck[0] = 1.0f;
	ck[1] = 0.5f;
	ck[9] = 0.001f;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    for(i = 0; i < NQUEUES; i++)
    	queues[i] = clCreateCommandQueue( ctx, device, 0, &err );

	// Print some info about the system
	clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
	printf("CL_DEVICE_VERSION=%s\n", version);
	clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(version), version, NULL);
	printf("CL_DRIVER_VERSION=%s\n", version);
	program = clCreateProgramWithSource(ctx, 1, (const char **)&source, NULL, &err);
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(param[0]), param, NULL);
	printf("CL_DEVICE_LOCAL_MEM_SIZE=%d\n", (int)param[0]);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(param[0]), param, NULL);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE=%d\n", (int)param[0]);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(param[0]), param, NULL);
	printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS=%d\n", (int)param[0]);
	j = param[0];
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(param[0])*j, param, NULL);
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES=");
	for(i = 0; i < j; i++)
		printf("%d ", (int)param[i]);
	printf("\n");
        clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(param[0]), param, NULL);
        printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE=%d\n", (int)param[0]);
		
		
	program = clCreateProgramWithSource(ctx, 1, (const char **)&source, NULL, &err);
	if(!program)
	{
		printf("Error creating program\n");
		return -1;
	}
	err = clBuildProgram(program, 0, 0, 0, 0, 0);
	if(err != CL_SUCCESS)
	{
		char buffer[20000];
		size_t len;
		
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		puts(buffer);
		return -1;
	}
	kernel = clCreateKernel(program, "conv9x9", &err);
	if(!kernel || err != CL_SUCCESS)
	{
		printf("Error creating kernel\n");
		return -1;
	}

    /* Prepare OpenCL memory objects and place matrices inside them. */
	cl_image_format fmt = {CL_RGBA, CL_FLOAT};
	cl_int rc;
	bufin = clCreateImage2D(ctx, CL_MEM_READ_ONLY, &fmt, dsize, dsize, 0, 0, &rc);
	bufout = clCreateImage2D(ctx, CL_MEM_WRITE_ONLY, &fmt, dsize, dsize, 0, 0, &rc);
    bufck = clCreateBuffer( ctx, CL_MEM_READ_ONLY, 9 * 9 * sizeof(*ck),
                          NULL, &err );

	size_t origin[3] = {0,0,0};
	size_t region[3] = {dsize, dsize, 1};
    err = clEnqueueWriteImage(queues[0], bufin, CL_TRUE, origin, region, dsize * sizeof(*in), 0, in, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queues[0], bufck, CL_TRUE, 0, 9 * 9 * sizeof( *ck ), ck, 0, NULL, NULL );
	clSetKernelArg(kernel, 0, sizeof(int), &dsize);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufin);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufck);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufout);
	local[0] = 8;
	local[1] = 8;
	global[0] = global[1] = dsize-32;
    usleep(100000);

	struct timeval start,end;
	gettimeofday(&start, NULL);

	for (k=0; k<nthreads; k++) {
		//printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
		for(i=0;i<nbOfAverages;i++) {
		// do the 2D convolution
			err = clEnqueueNDRangeKernel(queues[0], kernel, 2, NULL, global, local, 0, NULL, NULL);
			if(err != CL_SUCCESS)
			{
				printf("clEnqueueNDRangeKernel error %d\n", err);
				return -1;
			}
		}
	}

	clFinish(queues[0]);
	gettimeofday(&end, NULL);
	double t = ((double) (end.tv_sec - start.tv_sec))
	+ ((double) (end.tv_usec - start.tv_usec)) / 1e6; //reports time in [s] - verified!

    /* Wait for calculations to be finished. */

    /* Fetch results of calculations from GPU memory. */
    err = clEnqueueReadImage(queues[0], bufout, CL_TRUE, origin, region, dsize * sizeof(*out), 0, out, 0, NULL, NULL );
	clFinish(queues[0]);
	
	printf("%f %f %f %f\n", out[0].x, out[1].x, out[dsize].x, out[dsize+1].x);

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufin );
    clReleaseMemObject( bufck );
    clReleaseMemObject( bufout );

    /* Release OpenCL working objects. */
    for(i = 0; i < NQUEUES; i++)
    	clReleaseCommandQueue( queues[i] );
    clReleaseContext( ctx );
	
	// report performance:
	tops = 4 * nthreads * opsMAC * (dsize-32)*(dsize-32)*9*9; // total ops
	printf("Total M ops = %.0lf, # of threads = %d", nbOfAverages*tops*1e-6, nthreads);
	printf("\nTime in s: %lf:", t);
	printf("\nTest performance [G OP/s] %lf:", tops*nbOfAverages/t*1e-9);
	printf("\n");
	return(0);
}
