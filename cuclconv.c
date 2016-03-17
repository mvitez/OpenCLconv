#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fcntl.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <CL/opencl.h>

void subst(char *buf, const char *from, const char *to)
{
	char *p = buf;
	int fromlen = strlen(from);
	int tolen = strlen(to);
	
	while(p = strstr(p, from))
	{
		if(isalnum(p[fromlen]))
		{
			p++;
			continue;
		}
		memmove(p + tolen, p + fromlen, strlen(p + fromlen) + 1);
		memcpy(p, to, tolen);
	}
}

char *loadcusource(const char *path, ...)
{
	const char *vars = "\nstruct { int x, y; } blockIdx = {get_group_id(0), get_group_id(1)};\n"
		"struct { int x, y; } threadIdx = {get_local_id(0), get_local_id(1)};\n";
	FILE *fp = fopen(path, "r");
	if(!fp)
		return 0;
	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char *buf = malloc(2 * size);
	fread(buf, size, 1, fp);
	fclose(fp);
	buf[size] = 0;
	char *p = strstr(buf, "template");
	if(p)
	{
		char *q = strchr(p, '>');
		if(q)
			memmove(p, q+1, strlen(q+1) + 1);
	}
	p = strchr(buf, '{');
	if(p)
	{
		p++;
		memmove(p + strlen(vars), p, strlen(p) + 1);
		memcpy(p, vars, strlen(vars));
	}
	subst(buf, "bool", "int");
	subst(buf, "__global__", "__kernel");
	subst(buf, "__shared__", "__local");
	subst(buf, "__syncthreads()", "barrier(CLK_LOCAL_MEM_FENCE)");
	va_list ap;

	va_start(ap, path);
	char *from, to[20];
	while(from = va_arg(ap, char *))
	{
		sprintf(to, "%d", va_arg(ap, int));		
		subst(buf, from, to);
	}
	va_end(ap);
	return buf;
}

#define DIVUP(a,b) (((a)+(b-1))/(b))

int main()
{
	int i,j,k;
	// nb of operations:
	int nthreads = 1;
	int nbOfAverages = 1;//1e2;
	int opsMAC = 2; // operations per MAC
	cl_float *in, *out;
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
  
    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
	if(err)
		printf("clGetPlatformIDs failed, err=%d\n", err);
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if(err)
		printf("clGetDeviceIDs failed, err=%d\n", err);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	if(err)
		printf("clCreateContext failed, err=%d\n", err);
    for(i = 0; i < NQUEUES; i++)
    	queues[i] = clCreateCommandQueue( ctx, device, 0, &err );

	// Print some info about the system
	clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
	printf("CL_DEVICE_VERSION=%s\n", version);
	clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(version), version, NULL);
	printf("CL_DRIVER_VERSION=%s\n", version);
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

	int numImgColors = 3;
	int numImages = 128;
	int numFilters = 32;
	int imgSizeX = 256;
	int imgSizeY = 256;
	int filterSize = 9;
	int padding = 0;
	int paddingStart = -floor(padding/2);
	int moduleStride = 1;
	int numModulesY = (padding + imgSizeY - filterSize) / moduleStride + 1;
	int numModulesX = (padding + imgSizeX - filterSize) / moduleStride + 1;
	int imgStride = numImages;
	float scaleTargets = 0;
	float scaleOutputs = 1;
	int conv = 1;
	int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
	int numModules = numModulesY * numModulesX;
	int checkImgBounds = numImages % (32*imgsPerThread) != 0;

	// allocate matrices
	in = (cl_float *) calloc(numImages * numImgColors * imgSizeX * imgSizeY, sizeof(*in));
	out = (cl_float *) calloc(numImages * numFilters * numModulesX * numModulesY, sizeof(*out));
	ck = (cl_float *) calloc(numFilters * numImgColors * filterSize * filterSize, sizeof(*ck));
	in[0] = 2.0f;
	in[1] = 3.0f;
	in[imgSizeX] = 1.0;
	ck[0] = 2.0f;
	ck[numFilters] = 0.5f;
	
	char *src = loadcusource("filterActs_YxX_color.cu",
		"B_Y", 4,
		"B_X", 32,
		"imgsPerThread", 1,
		"filtersPerThread", (numFilters % 32 == 0 ? 8 : 4),
		"numColors", numImgColors,
		"scale", 0,
		"checkImgBounds", checkImgBounds,
		0);

	/*cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, false, true >, cudaFuncCachePreferShared);
	filterActs_YxX_color < 4, 32, 1, 8, 3, false, true > <<<blocks, threads>>>(images, filters, targets,
		numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
*/
		
	program = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err);
	if(!program)
	{
		printf("Error creating program, err = %d\n", err);
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
	kernel = clCreateKernel(program, "filterActs_YxX_color", &err);
	if(!kernel || err != CL_SUCCESS)
	{
		printf("Error creating kernel\n");
		return -1;
	}
	
	int elem_in = imgSizeX * imgSizeY * numImgColors * numImages;
	int elem_filt = filterSize * filterSize * numImgColors * numFilters;
	int elem_out = numImages * numFilters * numModulesY * numModulesX;
    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufin = clCreateBuffer( ctx, CL_MEM_READ_ONLY, elem_in * sizeof(*in),
                          NULL, &err );
    bufck = clCreateBuffer( ctx, CL_MEM_READ_ONLY, elem_filt * sizeof(*ck),
                          NULL, &err );
    bufout = clCreateBuffer( ctx, CL_MEM_READ_WRITE, elem_out * sizeof(*out),
                          NULL, &err );

    err = clEnqueueWriteBuffer( queues[0], bufin, CL_TRUE, 0,
        elem_in * sizeof( *in ), in, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queues[0], bufck, CL_TRUE, 0,
        elem_filt * sizeof( *ck ), ck, 0, NULL, NULL );
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufin);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufck);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufout);
	clSetKernelArg(kernel, 3, sizeof(int), &numImages);
	clSetKernelArg(kernel, 4, sizeof(int), &numFilters);
	clSetKernelArg(kernel, 5, sizeof(int), &imgSizeY);
	clSetKernelArg(kernel, 6, sizeof(int), &imgSizeX);
	clSetKernelArg(kernel, 7, sizeof(int), &filterSize);
	clSetKernelArg(kernel, 8, sizeof(int), &paddingStart);
	clSetKernelArg(kernel, 9, sizeof(int), &moduleStride);
	clSetKernelArg(kernel, 10, sizeof(int), &numModulesY);
	clSetKernelArg(kernel, 11, sizeof(int), &numModulesX);
	clSetKernelArg(kernel, 12, sizeof(int), &imgStride);
	clSetKernelArg(kernel, 13, sizeof(float), &scaleTargets);
	clSetKernelArg(kernel, 14, sizeof(float), &scaleOutputs);
	clSetKernelArg(kernel, 15, sizeof(int), &conv);

	if(numFilters % 32 == 0)
	{
		global[0] = DIVUP(numImages, 32 * imgsPerThread);
		global[1] = (numModules * numFilters) / (4 * 8);
	} else {
		global[0] = DIVUP(numImages, 32 * imgsPerThread);
		global[1] = (numModules * numFilters) / (4 * 4);
	}
	local[0] = 32;
	local[1] = 4;
	global[0] *= local[0];
	global[1] *= local[1];
	
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
    err = clEnqueueReadBuffer( queues[0], bufout, CL_TRUE, 0,
                                elem_out * sizeof(*out),
                                out, 0, NULL, NULL );
	clFinish(queues[0]);
	
	printf("%f %f %f %f\n", out[0], out[1], out[imgSizeX], out[imgSizeX+1]);

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufin );
    clReleaseMemObject( bufck );
    clReleaseMemObject( bufout );

    /* Release OpenCL working objects. */
    for(i = 0; i < NQUEUES; i++)
    	clReleaseCommandQueue( queues[i] );
    clReleaseContext( ctx );
	
	// report performance:
	tops = 1.0 * nthreads * opsMAC * numModulesX * numModulesY * numFilters * numImages * numImgColors * filterSize * filterSize;
	printf("Total M ops = %.0lf, # of threads = %d", nbOfAverages*tops*1e-6, nthreads);
	printf("\nTime in s: %lf:", t);
	printf("\nTest performance [G OP/s] %lf:", tops*nbOfAverages/t*1e-9);
	printf("\n");
	return(0);
}
