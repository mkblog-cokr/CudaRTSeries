#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

//MK: FB 사이즈
int nx = 1200;
int ny = 600;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
//MK: (코드 1-1) #val은 val 전체를 String으로 Return 함
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


//MK: Error 위치를 파악하기 위해서 사용
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
    	cerr << "MK: CUDA ERROR = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
    	// Make sure we call CUDA Device Reset before exiting
       	cudaDeviceReset();
       	exit(99);
    }
}

//MK: (코드 1-3) Kernel 코드. GPU에서 수행할 함수
__global__ void mkRender(float *fb, int max_x, int max_y) {
	//MK: Pixel 위치 계산을 위해 ThreadId, BlockId를 사용함
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

	//MK: 계산된 Pixel 위치가 FB사이즈 보다 크면 연산을 수행하지 않음
	if((i >= max_x) || (j >= max_y)){
	   return;
	}

	//MK: FB Pixel 값 계산
    int pixel_index = j*max_x*3 + i*3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2f;
}

int main() {
	//MK: Thread Block 사이즈
    int tx = 8;
    int ty = 8;

    cout << "MK: Rendering a " << nx << "x" << ny << " Image ";
    cout << "MK: in " << tx << "x" << ty << " Thread Blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    //MK: (코드 1-2) FB 메모리 할당 (cudaMallocManaged 는 Unitifed Memory를 사용 할 수 있도록 함)
	//MK: 필요에 따라 CPU/GPU에서 GPU/CPU로 데이터를 복사함
    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    //MK: (코드 1-4) GPU (CUDA) 연산을 위해서 Thread Block, Grid 사이즈 결정
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
	//MK: CUDA 함수 호출
    mkRender<<<blocks, threads>>>(fb, nx, ny);
    checkCudaErrors(cudaGetLastError());
	//MK: CUDA 연산이 완료되길 기다림
    checkCudaErrors(cudaDeviceSynchronize());
	//MK: 연산 시간과 끝 부분을 계산하여서 연산 시간을 측정함 

    //MK: (코드 1-5) CPU 코드와 동일하게 결과를 파일에 작성
	string fileName = "Ch1_gpu.ppm";
	ofstream writeFile(fileName.data());
	if(writeFile.is_open()){
		writeFile.flush();
		writeFile << "P3\n" << nx << " " << ny << "\n255\n";
    		for (int j = ny-1; j >= 0; j--) {
        		for (int i = 0; i < nx; i++) {
            			size_t pixel_index = j*3*nx + i*3;
            			float r = fb[pixel_index + 0];
            			float g = fb[pixel_index + 1];
            			float b = fb[pixel_index + 2];
            			int ir = int(255.99*r);
            			int ig = int(255.99*g);
            			int ib = int(255.99*b);
            			writeFile  << ir << " " << ig << " " << ib << "\n";
        		}
    		}
		writeFile.close();
	}

	//MK: 연산 시간과 끝 부분을 계산하여서 연산 시간을 측정함 
	stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << "MK: GPU (CUDA) Took " << timer_seconds << " Seconds.\n";
    //MK: 메모리 Free
    checkCudaErrors(cudaFree(fb));
    return 0;
}
