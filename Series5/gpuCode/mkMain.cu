#include <fstream>
#include "mkRay.h"
#include <time.h>
#include "mkSphere.h"
#include "mkHitablelist.h"
#include <float.h>

using namespace std;

//MK: FB 사이즈
int nx = 1200;
int ny = 600;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
//MK: #val은 val 전체를 String으로 Return 함 (출처 3)
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

//MK: (코드 4-2) Sphere 2개를 추가하는 코드
//MK: 1개의 Thread만 연산을 수행할 수 있도록 if문을 추가함
__global__ void mkCreateWorld(hitable **dList, hitable **dWorld){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        *(dList) = new sphere(vec3(0, 0, -1), 0.5);
	*(dList + 1) = new sphere(vec3(0, -100.5, -1), 100);
	*dWorld = new hitableList(dList, 2);
    }
}

//MK: (코드 4-7) 여러개의 Sphere Hit 여부를 판단하여 색상을 결정하도록 코드 변경
__device__ vec3 color(const ray &r, hitable **dWorld){
    hitRecord rec;
    vec3 ret = vec3(0, 0, 0);
    if((*dWorld)->hit(r, 0.0, FLT_MAX, rec)){
	ret = 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }	
    else{
	vec3 unitDirection = unitVector(r.direction());
    	float t = 0.5f * (unitDirection.y() + 1.0f);
    	ret = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
    return ret;
}

//MK: (코드 4-6) 여러개의 Sphere의 색상을 결정하기 위해서 코드 변경
__global__ void mkRender(vec3 *fb, int max_x, int max_y, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **dWorld) {
    //MK: Pixel 위치 계산을 위해 ThreadId, BlockId를 사용함
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //MK: 계산된 Pixel 위치가 FB사이즈 보다 크면 연산을 수행하지 않음
    if((i >= max_x) || (j >= max_y)){
        return;
    }

    //MK: FB Pixel 값 계산
    int pixel_index = j*max_x + i;
    float u = float(i)/float(max_x);
    float v = float(j)/float(max_y);
    ray r(origin, lowerLeftCorner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r, dWorld);
}

//MK: (코드 4-4) mkCreateWorld에서 생성한 클래스 제거
__global__ void mkFreeWorld(hitable **dList, hitable **dWorld){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        delete *(dList);
	delete *(dList + 1);
	delete *dWorld;
    }
}

int main() {
    //MK: Thread Block 사이즈
    int tx = 8;
    int ty = 8;

    cout << "MK: Rendering a " << nx << "x" << ny << " Image ";
    cout << "MK: in " << tx << "x" << ty << " Thread Blocks.\n";

    clock_t start, stop;
    start = clock();

    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(float);

    //MK: FB 메모리 할당 (cudaMallocManaged 는 Unitifed Memory를 사용 할 수 있도록 함)
    //MK: 필요에 따라 CPU/GPU에서 GPU/CPU로 데이터를 복사함
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    //MK: (코드 4-1) Sphere을 여러개 추가하기 위해서 메모리 할당을 진행
    hitable **dList;
    hitable **dWorld;
    checkCudaErrors(cudaMalloc((void **) &dList, 2 * sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((void **) &dWorld, sizeof(hitable *)));
	
    //MK: (코드 4-3) Sphere를 생성하는 함수를 호출함
    mkCreateWorld<<<1, 1>>>(dList, dWorld);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //MK: GPU (CUDA) 연산을 위해서 Thread Block, Grid 사이즈 결정
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    //MK: CUDA 함수 호출
    mkRender<<<blocks, threads>>>(fb, nx, ny,
					vec3(-2.0, -1.0, -1.0),
					vec3(4.0, 0.0, 0.0),
					vec3(0.0, 2.0, 0.0),
					vec3(0.0, 0.0, 0.0),
					dWorld);
    checkCudaErrors(cudaGetLastError());
    //MK: CUDA 연산이 완료되길 기다림
    checkCudaErrors(cudaDeviceSynchronize());
    //MK: 연산 시간과 끝 부분을 계산하여서 연산 시간을 측정함 

    //MK: CPU 코드와 동일하게 결과를 파일에 작성
    string fileName = "Ch5_gpu.ppm";
    ofstream writeFile(fileName.data());
    if(writeFile.is_open()){
	writeFile.flush();
	writeFile << "P3\n" << nx << " " << ny << "\n255\n";
    	for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j*nx + i;
            	int ir = int(255.99 * fb[pixel_index].r());
            	int ig = int(255.99 * fb[pixel_index].g());
            	int ib = int(255.99 * fb[pixel_index].b());
            	writeFile  << ir << " " << ig << " " << ib << "\n";
            }
    	}
	writeFile.close();
    }

    //MK: (코드 4-5) 사용한 메모리를 제거함
    mkFreeWorld<<<1, 1>>>(dList, dWorld);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(dList));
    checkCudaErrors(cudaFree(dWorld));
    checkCudaErrors(cudaFree(fb));
	
    //MK: 연산 시간과 끝 부분을 계산하여서 연산 시간을 측정함 
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cout << "MK: GPU (CUDA) Took " << timer_seconds << " Seconds.\n";

    return 0;
}