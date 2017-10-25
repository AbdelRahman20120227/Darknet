#include "opencl_platform.h"

cl_context clContext;
cl_kernel mmKernel;
cl_kernel mpKernel;
cl_kernel icKernel;
cl_kernel roKernel;
cl_command_queue queue;
cl_mem clCurrentImage;
int realHxW;

float *readOpenCLCurrentImage(int c, int h, int w, float * out){

	float *temp = malloc(realHxW * c * sizeof(float));
	clEnqueueReadBuffer(queue, clCurrentImage, CL_TRUE, 0, realHxW * c * sizeof(float), temp, 0, NULL, NULL);

	for(int i = 0; i < c; i++){
		for(int j = 0; j < h; j++){
			for(int k = 0; k < w; k++){
				out[(i * h * w) + (j * w) + k] = temp[(i * realHxW) + (j * w) + k];
			}
		}
	}

}
int initOpenCL(){

	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, NULL, &platformIdCount);

	cl_platform_id *platformIds = malloc(platformIdCount * sizeof(cl_platform_id));
	clGetPlatformIDs (platformIdCount, platformIds, NULL);

	fprintf(stderr, "\nThere are %d platforms,\nEnter your platform number:\n", platformIdCount);
	int platformNumber;

	FILE *inputStream = fopen("input.txt", "r");

	fscanf(inputStream, "%d", &platformNumber);

	cl_uint deviceIdCountCPU = 0;
	cl_uint deviceIdCountGPU = 0;

	clGetDeviceIDs(platformIds[platformNumber], CL_DEVICE_TYPE_CPU, 0, NULL
		, &deviceIdCountCPU);

	clGetDeviceIDs(platformIds[platformNumber], CL_DEVICE_TYPE_GPU, 0, NULL
		, &deviceIdCountGPU);

	fprintf(stderr, "\nThere are %d CPU & %d GPU,\nEnter 1 for CPU or 2 for GPU followed by device number:\n", deviceIdCountCPU, deviceIdCountGPU);
	int deviceType;
	int deviceNumber;

	fscanf(inputStream,"%d", &deviceType);

	fscanf(inputStream,"%d", &deviceNumber);

	fclose(inputStream);    
	cl_uint deviceIdCount = 0;
	cl_device_id * deviceIds = NULL;

	if(deviceType == 1){
		deviceIdCount = deviceIdCountCPU;
		deviceIds = malloc(deviceIdCount * sizeof(cl_device_id));
		clGetDeviceIDs(platformIds[platformNumber], CL_DEVICE_TYPE_CPU, deviceIdCountCPU, deviceIds
		, NULL);

	}
	else{	
		deviceIdCount = deviceIdCountGPU;
		deviceIds = malloc(deviceIdCount * sizeof(cl_device_id));
		clGetDeviceIDs(platformIds[platformNumber], CL_DEVICE_TYPE_GPU, deviceIdCountGPU, deviceIds
		, NULL);

	}

	const cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM,
		platformIds[platformNumber],
		0,
		0
	};


	cl_uint error;

	clContext = clCreateContext(contextProperties, deviceIdCount, deviceIds, NULL, NULL, &error);

	const char *source = "  __kernel void matrixMultiply (__global float* filter, __global float* image, __global float* mean, __global float* variance, __global float* bias, __global float* scale_bias, __global float* result, __constant int* dims)\
		            {\
		                const int BLOCKSIZE = 16;\
		                const int i = get_group_id (0);\
		                const int j = get_group_id (1);\
		                const int idy = get_local_id (0);\
		                const int idx = get_local_id (1);\
		                const int y = get_global_id (0);\
		                const int x = get_global_id (1);\
		                const int h1 = dims[0];\
		                const int w2 = dims[1];\
		                const int common = dims[2];\
		                const int batch_normalize = dims[3];\
		                const int leaky = dims[4];\
		                const int nSubMatrix = common / BLOCKSIZE;\
		                float8 tempVector = (float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);\
		                __local float local_filter[16][16];\
		                __local float local_image[16][16];\
		                int k;\
		                const int filter_pos_in_first_block = (i * common * BLOCKSIZE) + (idy * common) + idx;\
		                const int image_pos_in_first_block = (j * BLOCKSIZE) + (idy * w2) + idx;\
		                for(k = 0; k < nSubMatrix; k++){\
		                    local_filter[idy][idx] = filter[filter_pos_in_first_block + (k * BLOCKSIZE)];\
		                    local_image[idy][idx] = image[image_pos_in_first_block + (w2 * BLOCKSIZE * k)];\
		                    barrier(CLK_LOCAL_MEM_FENCE);\
		                    int m;\
		                    for(m = 0; m < BLOCKSIZE; m += 8){\
		                        float8 filter_temp = (float8)(local_filter[idy][m], local_filter[idy][m + 1], local_filter[idy][m + 2], local_filter[idy][m + 3], local_filter[idy][m + 4], local_filter[idy][m + 5], local_filter[idy][m + 6], local_filter[idy][m + 7]);\
		                        \
		                        float8 image_temp = (float8)(local_image[m][idx], local_image[m + 1][idx], local_image[m + 2][idx],local_image[m + 3][idx], local_image[m + 4][idx], local_image[m + 5][idx], local_image[m + 6][idx],local_image[m + 7][idx]);\
		                        tempVector += filter_temp * image_temp;\
		                    }\
		                    barrier(CLK_LOCAL_MEM_FENCE);\
		                }\
		                float4 first = tempVector.s0123 + tempVector.s4567;\
		                float2 second = first.s01 + first.s23;\
		                float intermediateResult = second.s0 + second.s1;\
		                if(batch_normalize != 0){\
		                    intermediateResult = ((intermediateResult - mean[y]) / (sqrt(variance[y]) + .000001f)) * scale_bias[y];\
		                }\
		                intermediateResult += bias[y];\
		                int b = (intermediateResult <= 0.0f);\
		                intermediateResult = (0.1 * intermediateResult * leaky * b) + ( ( (!leaky) || (!b) ) * intermediateResult);\
		                result[(y * w2) + x] = intermediateResult;\
		            }\
		            __kernel void maxPool (__global float* inImage, __global float* outImage, __constant int* dims)\
		            {\
		                const int i = get_global_id (0);\
		                const int j = get_global_id (1);\
		                const int k = get_global_id (2);\
		                float small = -1000000000.0f;\
		                int m;\
		                int n;\
		                int h = dims[0];\
		                int w = dims[1];\
		                int h_in = dims[2];\
		                int w_in = dims[3];\
		                int size = dims[4];\
		                int stride = dims[5];\
		                int offset = dims[6];\
		                int realHxW = dims[7];\
		                int out_index = (w * ( (i * h) + j)) + k;\
		                float max = small;\
		                for(m = 0; m < size; m++){\
		                    for(n = 0; n < size; n++){\
		                        int x = offset + (k * stride) + n;\
		                        int y = offset + (j * stride) + m;\
		                        bool valid = x >= 0 && x < w_in && y >= 0 && y < h_in;\
		                        float val = (valid * inImage[x + (y * w_in) + (i * realHxW)]) + ( (!valid) * small);\
		                        int b = val > max;\
		                        max = (b * val) + ( (!b) * max);\
		                    }\
		                }\
		                outImage[out_index] = max;\
		            }\
		            __kernel void im2col (__global float* im, __global float* col, __constant int* dims)\
		            {\
		                int i = get_global_id (0);\
		                int j = get_global_id (1);\
		                int h = dims[0];\
		                int w = dims[1];\
		                int c = dims[2];\
		                int out_h = dims[3];\
		                int out_w = dims[4];\
		                int stride = dims[5];\
		                int size = dims[6];\
		                int pad = dims[7];\
		                int new_out_h = dims[8];\
		                int new_out_w = dims[9];\
		                int realHxW = dims[10];\
		                if(i >= out_h || j >= out_w){\
		                    col[(i * new_out_w) + j] = 0;\
		                    return;\
		                }\
		                int offset = -1 * pad;\
		                int convH = ((h + (2 * pad) - size) / stride ) + 1;\
		                int convW = ((w + (2 * pad) - size) / stride ) + 1;\
		                int yFilterShift = j / (convW);\
		                int xFilterShift = j % (convW);\
		                int xStartPos = offset + (xFilterShift * stride);\
		                int yStartPos = offset + (yFilterShift * stride);\
		                int depth = i / (size * size);\
		                int remainder = i % (size * size);\
		                int xShift = remainder % (size);\
		                int yShift = remainder / size;\
		                int xPos = xStartPos + xShift;\
		                int yPos = yStartPos + yShift;\
		                if(xPos >= w || yPos >= h || xPos < 0 || yPos < 0){\
		                    col[(i * new_out_w) + j] = 0;\
		                    return;\
		                }\
		                col[(i * new_out_w) + j] = im[((depth * realHxW) + (yPos * w) + xPos)];\
		            }\
			    __kernel void reorg (__global float* input, __global float* output, __constant int* dims)\
		            {\
		                int i = get_global_id (0);\
		                int j = get_global_id (1);\
		                int k = get_global_id (2);\
		                int c = dims[0];\
		                int h = dims[1];\
		                int w = dims[2];\
				int realHxW = dims[3];\
				int stride = dims[4];\
				int newH = h / stride;\
				int newW = w / stride;\
				int startPosX = stride * k;\
				int startPosY = stride * j;\
				int depth = i / (stride * stride);\
				int iter = i % (stride * stride);\
				int down = iter / stride;\
				int right = iter % (stride);\
				int posX = startPosX + right;\
				int posY = startPosY + down;\
				output[(i * newH * newW) + (j * newW) + k] = input[(depth * realHxW) + (posY * w) + posX];\
		            }";

	cl_program program = createProgram(source, clContext, deviceIdCount, deviceIds);

	mmKernel = createKernel(program, "matrixMultiply");

	mpKernel = createKernel(program, "maxPool");

	icKernel = createKernel(program, "im2col");
	
	roKernel = createKernel(program, "reorg");
	size_t max_group_size;

	cl_int compute_units;

	clGetDeviceInfo(deviceIds[deviceNumber], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &compute_units, NULL);

	clGetDeviceInfo(deviceIds[deviceNumber], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_group_size, NULL);

	fprintf(stderr, "Computing units = %d max size =  %d\n", compute_units, max_group_size);
	
	queue = clCreateCommandQueue(clContext, deviceIds[deviceNumber], 0, &error);

	fprintf(stderr, "%s %d\n", "queue error = ", error);

}

int initOpenCLCurrentImage(int w, int h, int c, float* X){

	int error;	
	clCurrentImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * w * h * c, X, &error);
	realHxW = w * h;	
	return error;

}
int initCLLayerOutput(layer *l){
	l->clOutput = (cl_mem *)malloc(sizeof(cl_mem));
	l->clRealHxW = (int *)malloc(sizeof(int));
}
int initCLConvLayerData(layer *l){

	cl_int error;

	int m = l->n;
	int k = l->size * l->size * l->c;

	int power = pow(2, 4);
	int newM = m + ((power - (m % power)) % power);
	int newK = k + ((power - (k % power)) % power);
	
	l->clOutput = (cl_mem *)malloc(sizeof(cl_mem));	

	float *newWeights = calloc(newM * newK, sizeof(float));

	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++){
		    	newWeights[(i * newK) + j] = l->weights[(i * k) + j];
		}
	}

	float *newMean = calloc(newM, sizeof(float));
	float *newVariance = calloc(newM, sizeof(float));
	float *newBias = calloc(newM, sizeof(float));
	float *newScaleBias = calloc(newM, sizeof(float));

	for(int i = 0; i < m; i++){
		newBias[i] = l->biases[i];
		if(l->batch_normalize){
			newMean[i] = l->rolling_mean[i];
			newVariance[i] = l->rolling_variance[i];
			newScaleBias[i] = l->scales[i];
		}
	}

	l->clWeights = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * newM * newK, newWeights, &error);
	l->clMean = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * newM, newMean, &error);
	l->clVariance = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * newM, newVariance, &error);
	l->clBias = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * newM, newBias, &error);
	l->clScaleBias = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * newM, newScaleBias, &error);

	free(newWeights);
	free(newMean);
	free(newVariance);
	free(newBias);
	free(newScaleBias);

	return error;

}

cl_program createProgram(const char *source, cl_context context, cl_uint deviceIdCount, cl_device_id *deviceIds){

	int error;

	size_t programSize = strlen(source);

	size_t *lengths = malloc(sizeof(size_t));
	lengths[0] = programSize;

	const char **sources = malloc(sizeof(char *));

	sources[0] = source;

	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);

	fprintf(stderr, "%s %d\n", "error = ", error);

	clBuildProgram(program, deviceIdCount, deviceIds, "", NULL, NULL);

	return program;

}

cl_kernel createKernel(cl_program program, const char *kernelName){
    
	int error;

	cl_kernel kernel = clCreateKernel(program, kernelName, &error);

	fprintf(stderr, "%s %d\n", "kernel error = ", error);

	return kernel;

}

void clMaxPool(int size, int stride, int offset, int h, int w, int c, int h_in, int w_in){

	cl_int error;

	cl_mem clOutImage = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * h * w * c, NULL, &error);

	int *dims = malloc(8 * sizeof(int));
	dims[0] = h;
	dims[1] = w;
	dims[2] = h_in;
	dims[3] = w_in;
	dims[4] = size;
	dims[5] = stride;
	dims[6] = offset;
	dims[7] = realHxW;

	cl_mem clDims = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8 * sizeof(int), dims, &error);

	error = clSetKernelArg(mpKernel, 0, sizeof(cl_mem), &clCurrentImage);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mpKernel, 1, sizeof(cl_mem), &clOutImage);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mpKernel, 2, sizeof(cl_mem), &clDims);

	//printf("%s %d\n", "error = ", error);

	const size_t offsets[3] = {0, 0, 0};
	const size_t sizes[3] = {c, h, w};

	error = clEnqueueNDRangeKernel(queue, mpKernel, 3, offsets, sizes, NULL, 0, NULL, NULL);

	realHxW = h * w;
	clCurrentImage = clOutImage;
	
	return error;

}

void assignCLOutput(cl_mem *output, int *clRealHxW){
	
	*output = clCurrentImage;
	*clRealHxW = realHxW;
}

void matrixMultiply(cl_mem clWeights, cl_mem clMean, cl_mem clVariance, cl_mem clBias, cl_mem clScaleBias, size_t h1, size_t w2, size_t common, int batch_normalize, int leaky){
    
	//fprintf(stderr, "\n\n%s\n\n", "MATRIX_MULTIPLY");

	cl_int error;

	cl_mem clResult = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
		sizeof(float) * h1 * w2, NULL, &error);

	int *dims = malloc(5 * sizeof(int));
	dims[0] = h1;
	dims[1] = w2;
	dims[2] = common;
	dims[3] = batch_normalize;
	dims[4] = leaky;

	cl_mem clDims = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		5 * sizeof(int), dims, &error);

	error = clSetKernelArg(mmKernel, 0, sizeof(cl_mem), &clWeights);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 1, sizeof(cl_mem), &clCurrentImage);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 2, sizeof(cl_mem), &clMean);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 3, sizeof(cl_mem), &clVariance);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 4, sizeof(cl_mem), &clBias);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 5, sizeof(cl_mem), &clScaleBias);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 6, sizeof(cl_mem), &clResult);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(mmKernel, 7, sizeof(cl_mem), &clDims);

	//printf("%s %d\n", "error = ", error);

	const size_t offsets[3] = {0, 0, 0};
	const size_t sizes[3] = {h1, w2, 1};

	const size_t group_sizes[3] = {16, 16, 1};

	error = clEnqueueNDRangeKernel(queue, mmKernel, 2, offsets, sizes, group_sizes, 0, NULL, NULL);
	//clFinish(queue);
	
	clCurrentImage = clResult;
	//fprintf(stderr, "\nEnd Matrix Multiply\n");
}

void clIm2Col(int h, int w, int c, int size, int stride, int pad, int newOut_h, int newOut_w){

	//printf("\n\n%s\n\n", "IMG_2_COL");

	cl_int error;

	int out_h = size * size * c;
	int out_w = (((h + 2*pad - size) / stride) + 1) * (((w + 2*pad - size) / stride) + 1);

	cl_mem clOutImage = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
	sizeof(float) * newOut_h * newOut_w, NULL, &error);

	int *dims = malloc(11 * sizeof(int));
	dims[0] = h;
	dims[1] = w;
	dims[2] = c;
	dims[3] = out_h;
	dims[4] = out_w;
	dims[5] = stride;
	dims[6] = size;
	dims[7] = pad;
	dims[8] = newOut_h;
	dims[9] = newOut_w;
	dims[10] = realHxW;

	cl_mem clDims = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	11 * sizeof(int), dims, &error);

	error = clSetKernelArg(icKernel, 0, sizeof(cl_mem), &clCurrentImage);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(icKernel, 1, sizeof(cl_mem), &clOutImage);

	//printf("%s %d\n", "error = ", error);

	error = clSetKernelArg(icKernel, 2, sizeof(cl_mem), &clDims);

	//printf("%s %d\n", "error = ", error);

	const size_t offsets[3] = {0, 0, 0};
	const size_t sizes[3] = {newOut_h, newOut_w, 1};

	error = clEnqueueNDRangeKernel(queue, icKernel, 2, offsets, sizes, NULL, 0, NULL, NULL);

	clCurrentImage = clOutImage;
	realHxW = newOut_w;

}
void clCopyFromHistory(float *data, int h, int w,int *clRealHxW, int c, cl_mem *clHistory, int flag){
	
	float *temp = malloc(*clRealHxW * c * sizeof(float));
	clEnqueueReadBuffer(queue, *clHistory, CL_TRUE, 0, *clRealHxW * c * sizeof(float), temp, 0, NULL, NULL);
 	for(int i = 0; i < c; i++){		
		for(int j = 0; j < h; j++){
			for(int k = 0; k < w; k++){
				data[(i * h * w) + (j * w) + k] = temp[(i * (*clRealHxW)) + (j * w) + k];			
			}		
		}	
	}

	
	realHxW = h * w;
	
}

void clRoute(float *X, int size){
	int error;
	clCurrentImage = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, X, &error);	
}

void clReorg(int h, int w, int c, int stride){

	cl_int error;

	int *dims = malloc(5 * sizeof(int));
	
	int newC = c * stride * stride;
	int newW = w / stride;
	int newH = h / stride;
	
	cl_mem clResult = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
		sizeof(float) * newW * newH * newC, NULL, &error);
	
	dims[0] = c;
	dims[1] = h;
	dims[2] = w;
	dims[3] = realHxW;
	dims[4] = stride;
	
	cl_mem clDims = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  5 * sizeof(int), dims, &error);
	
	clSetKernelArg(roKernel, 0, sizeof(cl_mem), &clCurrentImage);
	
	clSetKernelArg(roKernel, 1, sizeof(cl_mem), &clResult);

	clSetKernelArg(roKernel, 2, sizeof(cl_mem), &clDims);
	
	const size_t offsets[3] = {0, 0, 0};
	const size_t sizes[3] = {newC, newH, newW};
	
	error = clEnqueueNDRangeKernel(queue, roKernel, 3, offsets, sizes, NULL, 0, NULL, NULL);
	
	clCurrentImage = clResult;
	realHxW = newH * newW;
	
}





