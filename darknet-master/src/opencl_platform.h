#include "CL/cl.h"
#include "stdio.h"
#include "layer.h"

extern cl_context clContext;
extern cl_kernel mmKernel;
extern cl_kernel mpKernel;
extern cl_kernel icKernel;
extern cl_kernel roKernel;
extern cl_command_queue queue;
extern cl_mem clCurrentImage;
extern int realHxW;

void matrixMultiply(cl_mem clWeights, cl_mem clMean, cl_mem clVariance, cl_mem clBias, cl_mem clScaleBias, 
	size_t h1, size_t w2, size_t common, int batch_normalize, int leaky);

void clIm2Col(int h, int w, int c, int size, int stride, int pad, int newOut_h, int newOut_w);
float *readOpenCLCurrentImage(int c, int h, int w, float * out);
int initOpenCL();
int initOpenCLCurrentImage(int w, int h, int c, float* X);
int initCLLayerData(layer *l);
cl_program createProgram(const char *source, cl_context context, cl_uint deviceIdCount, cl_device_id *deviceIds);
cl_kernel createKernel(cl_program program, const char *kernelName);
void clMaxPool(int size, int stride, int offset, int h, int w, int c, int h_in, int w_in);

