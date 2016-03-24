// GPU based vector summation 

//#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>


#define CHECK(call)					\
{							\
	const cudaError_t error = call;			\
	if (error != cudaSuccess)			\
	{						\
		printf("Error: %s: %d", __FILE__, __LINE__);\
		printf("code:%d, reason: %s \n", error, cudaGetErrorString(error)); \
		exit(1);    \
	}		    \
}			     \


/*
void checkResult(float *hostRef, float *gpuRef, const int N){
	double epsilon = 1.0E-8;
	bool match =1;
	for (int i=0; i<N; i++){
		if (abs(hostRef[i] - gpuRef[i])> epsilon){
			match  = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i],i);
			break;
	}
}
	if(match) 
		printf("Arrays match!\n\n");
}
*/
double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialDataX(float *ip, int size){
	time_t t;
	srand((unsigned) time(&t));
	
	for (int i=0; i<size; i++){
		//ip[i] = (float)(rand() & 0xFF)/10.0f;
		ip[i]=i;
	}

}

void initialDataY(float *ip, int size){
	time_t t;
	srand((unsigned) time(&t));
	
	for (int i=0; i<size; i++){
		ip[i] = (float)(rand() & 0xFF)/5.0f;
		//ip[i]=i;
		
	}

}

void linearRegressionOnHost(float *X,float *Y, float predict_value, const int N) {
	long double sum_x=0.0,sum_y=0.0,sum_xy=0.0,sum_xsq = 0.0;
	float beta0, beta1;	
	for (int idx=0; idx<N; idx++){
		sum_x = sum_x + X[idx];
		sum_y = sum_y + Y[idx];
		sum_xy = sum_xy + X[idx]*Y[idx];
		sum_xsq = sum_xsq + X[idx]*X[idx];
		}
	
	printf("sum_x: %Lf \n",sum_x);
	printf("sum_y: %Lf \n",sum_y);
	printf("sum_xy: %Lf \n",sum_xy);
	printf("sum_xsq: %Lf \n",sum_xsq);
	beta0 = (sum_xsq * sum_y - sum_x * sum_xy)/ (N * sum_xsq - sum_x * sum_x);
	printf("Beta0: %f \n",beta0);

	beta1 = (N * sum_xy - sum_x * sum_y)/ (N* sum_xsq - sum_x * sum_x);
	printf("Beta1: %f \n",beta1);
		
	float predict_output = 0.0;
	predict_output = beta0 + beta1 * predict_value;
	printf("Predicted output: %f \n",predict_output);
}

__global__ void kernel_multiply(float *x,float *y,float *p, const int size){

   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if(tid < size) {

	   p[tid] = x[tid] * y[tid];
   }



}

__global__ void kernel_sum_x(float *in,float *partial_sum, const int size){
	
    int tid = threadIdx.x;
    //local pointer of this data block
    float *data = in + blockIdx.x * blockDim.x*4;
    int index = blockIdx.x * blockDim.x*4 + tid;

    //assume size is multiple of 4*blockDim.x
    if(index+blockDim.x*3 < size){
    	float b1 = in[index];
    	float b2 = in[index+blockDim.x];
    	float b3 = in[index+blockDim.x*2];
    	float b4 = in[index+blockDim.x*3];

    	//
    	in[index] = b1 + b2 + b3 + b4;
   }
    __syncthreads();

    //now one block
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    	if(tid < stride){
   			 data[tid] = data[tid] + data[tid+stride];
  		    //printf("B(%d,%d),T(%d,%d)  s=%d ins=%f\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,stride,ins[tid] );
    	} //if
    	__syncthreads();
    }//for

    //at this point all threads are done, thread with id 0 has the thread block sum
    if ( tid == 0 ){
          partial_sum[blockIdx.x] = data[0];
    }
}

__global__ void kernel_sum_y(float *in,float *partial_sum, const int size){
	
    int tid = threadIdx.x;
    //local pointer of this data block
    float *data = in + blockIdx.x * blockDim.x*4;
    int index = blockIdx.x * blockDim.x*4 + tid;

    //assume size is multiple of 4*blockDim.x
    if(index+blockDim.x*3 < size){
    	float b1 = in[index];
    	float b2 = in[index+blockDim.x];
    	float b3 = in[index+blockDim.x*2];
    	float b4 = in[index+blockDim.x*3];

    	//
    	in[index] = b1 + b2 + b3 + b4;
   }
    __syncthreads();

    //now one block
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    	if(tid < stride){
   			 data[tid] = data[tid] + data[tid+stride];
  		    //printf("B(%d,%d),T(%d,%d)  s=%d ins=%f\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,stride,ins[tid] );
    	} //if
    	__syncthreads();
    }//for

    //at this point all threads are done, thread with id 0 has the thread block sum
    if ( tid == 0 ){
          partial_sum[blockIdx.x] = data[0];
          //printf("B(%d,%d),T(%d,%d) %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,partial_sum[blockIdx.x] );
    }
}

__global__ void kernel_sum_xy(float *in,float *partial_sum, const int size){
	
    int tid = threadIdx.x;
    //local pointer of this data block
    float *data = in + blockIdx.x * blockDim.x*4;
    int index = blockIdx.x * blockDim.x*4 + tid;

    //assume size is multiple of 4*blockDim.x
    if(index+blockDim.x*3 < size){
    	float b1 = in[index];
    	float b2 = in[index+blockDim.x];
    	float b3 = in[index+blockDim.x*2];
    	float b4 = in[index+blockDim.x*3];

    	//
    	in[index] = b1 + b2 + b3 + b4;
   }
    __syncthreads();

    //now one block
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    	if(tid < stride){
   			 data[tid] = data[tid] + data[tid+stride];
  		    //printf("B(%d,%d),T(%d,%d)  s=%d ins=%f\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,stride,ins[tid] );
    	} //if
    	__syncthreads();
    }//for

    //at this point all threads are done, thread with id 0 has the thread block sum
    if ( tid == 0 ){
          partial_sum[blockIdx.x] = data[0];
          //printf("B(%d,%d),T(%d,%d) %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,partial_sum[blockIdx.x] );
    }
}

__global__ void kernel_sum_xsq(float *in,float *partial_sum, const int size){
	
    int tid = threadIdx.x;
    //local pointer of this data block
    float *data = in + blockIdx.x * blockDim.x*4;
    int index = blockIdx.x * blockDim.x*4 + tid;

    //assume size is multiple of 4*blockDim.x
    if(index+blockDim.x*3 < size){
    	float b1 = in[index];
    	float b2 = in[index+blockDim.x];
    	float b3 = in[index+blockDim.x*2];
    	float b4 = in[index+blockDim.x*3];

    	//
    	in[index] = b1 + b2 + b3 + b4;
   }
    __syncthreads();

    //now one block
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    	if(tid < stride){
   			 data[tid] = data[tid] + data[tid+stride];
  		    //printf("B(%d,%d),T(%d,%d)  s=%d ins=%f\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,stride,ins[tid] );
    	} //if
    	__syncthreads();
    }//for

    //at this point all threads are done, thread with id 0 has the thread block sum
    if ( tid == 0 ){
          partial_sum[blockIdx.x] = data[0];
          //printf("B(%d,%d),T(%d,%d) %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,partial_sum[blockIdx.x] );
    }
}

int main(int argc, char **argv){
	printf("%s Starting ...\n", argv[0]);
	
	// set up device
	int dev = 0;
	cudaSetDevice(dev);
	
	// set up data size of array 
	int size = 1<<13;
	int blockdimx = 256;
	printf("Array size %d\n", size);

	// Host memory allocation
	size_t nBytes = size * sizeof(float);
	
	float *h_X, *h_Y, predict_value = 30;// *hostRef, *gpuRef;
	float *h_partial_sum_x; 
	float *h_partial_sum_y;
	float *h_partial_sum_xy; 
	float *h_partial_sum_xsq;

	h_X = (float *) malloc(nBytes);
	h_Y = (float *) malloc(nBytes);
	
	//hostRef = (float *) malloc(nBytes);
	//gpuRef = (float *) malloc(nBytes);
	
	// initialialize data at host side
	initialDataX(h_X, size);
	initialDataY(h_Y, size);

	//memset(hostRef, 0, nBytes);
	//memset(gpuRef, 0, nBytes);
	
	double iStart, iElaps;

	// Device memory allocation
	float *d_X, *d_Y;
	float *d_partial_sum_x;
	float *d_partial_sum_y;
	float *d_P;
	float *d_P_xsq;
	float *d_partial_sum_xy;
	float *d_partial_sum_xsq;

	cudaMalloc((float **)&d_X, nBytes);
	cudaMalloc((float **)&d_Y, nBytes);
	cudaMalloc((float **)&d_P, nBytes);
	cudaMalloc((float **)&d_P_xsq, nBytes);
	

	// transfer data from host to device
	CHECK(cudaMemcpy(d_X, h_X, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Y, h_Y, nBytes, cudaMemcpyHostToDevice));
	cudaMemset(d_P, 0, nBytes);
	cudaMemset(d_P_xsq, 0, nBytes);

	 //grid and block dim
	dim3 block(blockdimx, 1, 1);
	dim3 grid((size + block.x - 1) / block.x / 4, 1 , 1);
	dim3 gridm((size + block.x - 1) / block.x , 1 , 1);
        nBytes = grid.x * sizeof(float); 
	
	//host
        h_partial_sum_x = (float*) malloc(nBytes);
	h_partial_sum_y = (float*) malloc(nBytes);
	h_partial_sum_xy = (float *) malloc(nBytes);
	h_partial_sum_xsq = (float *) malloc(nBytes);
	//device	
	cudaMalloc((void**) &d_partial_sum_x, nBytes);
	cudaMalloc((void**) &d_partial_sum_y, nBytes);
	cudaMalloc((void**) &d_partial_sum_xy, nBytes);
	cudaMalloc((void**) &d_partial_sum_xsq, nBytes);

	// kernel_multiply XY
	double startm = cpuSecond();
	kernel_multiply<<<gridm, block>>>(d_X, d_Y, d_P, size);
	cudaDeviceSynchronize();
	double tm = cpuSecond() - startm;

	// kernel_multiply XX
	double startxx = cpuSecond();
	kernel_multiply<<<gridm, block>>>(d_X, d_X, d_P_xsq, size);
	cudaDeviceSynchronize();
	double txx = cpuSecond() - startxx;

	// kernel_sum_x
	double startx = cpuSecond();
	kernel_sum_x<<<grid, block>>>(d_X, d_partial_sum_x, size);
	cudaDeviceSynchronize();
	double tx = cpuSecond() - startx;
	cudaMemcpy(h_partial_sum_x, d_partial_sum_x, nBytes, cudaMemcpyDeviceToHost);
	//printf("Starting of sum_x");
    //final sum of device on the host
	long double device_sum_x = 0;
	int kx;
		 for (kx=0; kx < grid.x; kx++){
			 device_sum_x = device_sum_x + h_partial_sum_x[kx];
                        
		 }
	printf("Sum_x on GPU: %Lf \n",device_sum_x);

	// kernel_sum_y
	double starty = cpuSecond();
	kernel_sum_y<<<grid, block>>>(d_Y, d_partial_sum_y, size);
	cudaDeviceSynchronize();
	double ty = cpuSecond() - starty;
	cudaMemcpy(h_partial_sum_y, d_partial_sum_y, nBytes, cudaMemcpyDeviceToHost);

    //final sum of device on the host
	long double device_sum_y = 0;
	int ky;
		 for (ky=0; ky < grid.x; ky++){
			 device_sum_y = device_sum_y + h_partial_sum_y[ky];
                        
		 }
	printf("Sum_y on GPU: %Lf \n",device_sum_y);

	// kernel_sum_xy
	double startxy = cpuSecond();
	kernel_sum_xy<<<grid, block>>>(d_P, d_partial_sum_xy, size);
	cudaDeviceSynchronize();
	double txy = cpuSecond() - startxy;
	cudaMemcpy(h_partial_sum_xy, d_partial_sum_xy, nBytes, cudaMemcpyDeviceToHost);

    //final sum of device on the host
	long double device_sum_xy = 0;
	int kxy;
		 for (kxy=0; kxy < grid.x; kxy++){
			 device_sum_xy = device_sum_xy + h_partial_sum_xy[kxy];
                        
		 }
	printf("Sum_xy on GPU: %Lf \n",device_sum_xy);
	
	// kernel_sum_xsq
	double startxsq = cpuSecond();
	kernel_sum_xsq<<<grid, block>>>(d_P_xsq, d_partial_sum_xsq, size);
	cudaDeviceSynchronize();
	double txsq = cpuSecond() - startxsq;
	cudaMemcpy(h_partial_sum_xsq, d_partial_sum_xsq, nBytes, cudaMemcpyDeviceToHost);

    //final sum of device on the host
	long double device_sum_xsq = 0;
	int kxx;
		 for (kxx=0; kxx < grid.x; kxx++){
			 device_sum_xsq = device_sum_xsq + h_partial_sum_xsq[kxx];
                        
		 }
	printf("Sum_xsq on GPU: %Lf \n",device_sum_xsq);
	
	double t_total= tm+ txx+tx+ty+txy+ txsq;
	printf("total gpu time: %f\n",t_total);

	// add vector at host side for result checks
	iStart = cpuSecond();
	linearRegressionOnHost(h_X, h_Y, predict_value, size);
	iElaps = cpuSecond() - iStart;
	printf("Time elapsed of LinearRegressionOnHost: %f",iElaps,"sec\n");
	
	// check device results
	//checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_P);
	cudaFree(d_P_xsq);
	cudaFree(d_partial_sum_x);
	cudaFree(d_partial_sum_y);
	cudaFree(d_partial_sum_xy);
	cudaFree(d_partial_sum_xsq);

	// free host memory
	free(h_X);
	free(h_Y);
	free(h_partial_sum_x);
	free(h_partial_sum_y);
	//free(hostRef);
	//free(gpuRef);
	cudaDeviceReset();
	return 0;	
}
