#include <cuda_runtime.h>   // CUDA 런타임 API 헤더, 텐서 메모리 할당/해제/복사를 위한 함수 포함
#include <cufft.h>          // cuFFT 라이브러리 헤더, FFT 연산용 복소수 텐서 처리 함수 포함

// 입력 및 처리할 이미지(텐서)의 크기 정의
#define HEIGHT 1024         // 입력/출력 텐서 높이: HEIGHT
#define WIDTH  1024         // 입력/출력 텐서 너비: WIDTH

// 실수 입력 텐서(HEIGHT x WIDTH)를 복소수(HEIGHT x WIDTH)로 변환하는 커널
__global__ void realToComplex(
    const float* __restrict__ realInput, // [HEIGHT, WIDTH] shape 실수 입력 텐서
    cufftComplex* __restrict__ complexData // [HEIGHT, WIDTH] shape 복소수 출력 텐서
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x 좌표 (0 ≤ x < WIDTH)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y 좌표 (0 ≤ y < HEIGHT)
    if (x < WIDTH && y < HEIGHT) {
        int idx = y * WIDTH + x;                   // 평탄화된 1D 인덱스 (총 HEIGHT*WIDTH 요소)
        complexData[idx].x = realInput[idx];       // 실수부 ← realInput[idx], 텐서 shape 유지
        complexData[idx].y = 0.0f;                  // 허수부 ← 0, 텐서 shape 유지
    }
}

// 노치 필터를 적용하는 커널 (주파수 도메인에서)
__global__ void applyNotchFilter(
    cufftComplex* __restrict__ data, // [HEIGHT, WIDTH] shape 복소수 주파수 텐서
    int u0,                          // 제거할 주파수 중심 u 좌표
    int v0,                          // 제거할 주파수 중심 v 좌표
    int radius                       // 노치 반경 (거리 기준)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x 좌표 (0 ≤ x < WIDTH)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y 좌표 (0 ≤ y < HEIGHT)
    if (x < WIDTH && y < HEIGHT) {
        int idx = y * WIDTH + x;                   // 평탄화된 1D 인덱스
        // 중앙 이동된 주파수 좌표 계산 (범위: -WIDTH/2..WIDTH/2-1, -HEIGHT/2..HEIGHT/2-1)
        int u = (x < WIDTH/2) ? x : x - WIDTH;     
        int v = (y < HEIGHT/2) ? y : y - HEIGHT;
        // 노치 거리 계산 (두 대칭 위치)
        float dist1 = sqrtf((u - u0)*(u - u0) + (v - v0)*(v - v0)); // scalar
        float dist2 = sqrtf((u + u0)*(u + u0) + (v + v0)*(v + v0)); // scalar
        if (dist1 < radius || dist2 < radius) {
            data[idx].x = 0.0f; // 실수부 0으로 설정 (노치 제거)
            data[idx].y = 0.0f; // 허수부 0으로 설정 (노치 제거)
        }
    }
}

// 복소수(HEIGHT x WIDTH) 주파수 텐서를 실수(HEIGHT x WIDTH)로 변환 및 정규화하는 커널
__global__ void complexToReal(
    const cufftComplex* __restrict__ complexData, // [HEIGHT, WIDTH] 복소수 텐서
    float* __restrict__ realOutput                // [HEIGHT, WIDTH] 실수 출력 텐서
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x 좌표 (0 ≤ x < WIDTH)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y 좌표 (0 ≤ y < HEIGHT)
    if (x < WIDTH && y < HEIGHT) {
        int idx = y * WIDTH + x;                   // 평탄화된 1D 인덱스
        // 역 FFT 결과 정규화: 실수부 / (HEIGHT*WIDTH)
        realOutput[idx] = complexData[idx].x / (float)(HEIGHT * WIDTH);
    }
}

int main() {
    // 호스트 메모리 할당: 입력 및 출력 실수 텐서 [HEIGHT, WIDTH]
    float* h_input  = (float*)malloc(sizeof(float) * HEIGHT * WIDTH); // real input tensor
    float* h_output = (float*)malloc(sizeof(float) * HEIGHT * WIDTH); // real output tensor
    // (실제 사용 시 h_input에 이미지 데이터 초기화 필요)

    // 디바이스 메모리 할당: 실수 입력/출력 텐서 [HEIGHT, WIDTH]
    float* d_input;  
    cudaMalloc(&d_input,  sizeof(float) * HEIGHT * WIDTH); // real input tensor on GPU
    float* d_output; 
    cudaMalloc(&d_output, sizeof(float) * HEIGHT * WIDTH); // real output tensor on GPU

    // 디바이스 메모리 할당: 복소수 FFT 텐서 [HEIGHT, WIDTH]
    cufftComplex* d_data;
    cudaMalloc(&d_data, sizeof(cufftComplex) * HEIGHT * WIDTH); // complex tensor for FFT

    // 호스트 → 디바이스로 실수 입력 텐서 복사 (shape: [HEIGHT, WIDTH])
    cudaMemcpy(d_input, h_input, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyHostToDevice);

    // cuFFT 핸들 생성 및 2D C2C FFT 계획 수립 (텐서 shape: [HEIGHT, WIDTH])
    cufftHandle plan;
    cufftPlan2d(&plan, HEIGHT, WIDTH, CUFFT_C2C);

    // 커널 실행 설정: 블록(16×16), 그리드(ceil(WIDTH/16), ceil(HEIGHT/16))
    dim3 block(16, 16);
    dim3 grid((WIDTH + 15)/16, (HEIGHT + 15)/16);

    // 실수 입력 → 복소수 텐서 변환 커널 실행 (입력 [HEIGHT, WIDTH] → 출력 [HEIGHT, WIDTH])
    realToComplex<<<grid, block>>>(d_input, d_data);

    // 순방향 FFT 실행 (in-place): 복소수 텐서 [HEIGHT, WIDTH]
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // 노치 필터 적용 커널 실행 (주파수 도메인에서 텐서 [HEIGHT, WIDTH])
    int u0 = 50, v0 = 50, radius = 5; // 예시 파라미터
    applyNotchFilter<<<grid, block>>>(d_data, u0, v0, radius);

    // 역방향 FFT 실행 (in-place): 복소수 텐서 [HEIGHT, WIDTH]
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    // 복소수 → 실수 변환 및 정규화 커널 실행 (입력 [HEIGHT, WIDTH] → 출력 [HEIGHT, WIDTH])
    complexToReal<<<grid, block>>>(d_data, d_output);

    // 디바이스 → 호스트로 결과 실수 텐서 복사 (shape: [HEIGHT, WIDTH])
    cudaMemcpy(h_output, d_output, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyDeviceToHost);

    // 리소스 해제
    cufftDestroy(plan);                 // FFT 핸들 해제
    cudaFree(d_input);                  // GPU 입력 텐서 해제
    cudaFree(d_output);                 // GPU 출력 텐서 해제
    cudaFree(d_data);                   // GPU 복소수 텐서 해제
    free(h_input);                      // 호스트 입력 텐서 해제
    free(h_output);                     // 호스트 출력 텐서 해제

    return 0;                           // 프로그램 종료
}
