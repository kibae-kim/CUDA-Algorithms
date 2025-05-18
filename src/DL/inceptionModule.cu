#include <cuda_runtime.h> // CUDA 런타임 API 포함
#include <device_launch_parameters.h> // 디바이스 파라미터 정의

// 2D Convolution 커널
__global__ void conv2d_kernel(const float* __restrict__ input,      // 입력 텐서: [N, C, H, W]
                              float* __restrict__ output,           // 출력 텐서: [N, outC, H_out, W_out]
                              const float* __restrict__ weight,     // 가중치 텐서: [outC, C, K, K]
                              const float* __restrict__ bias,       // 바이어스 텐서: [outC]
                              int N, int C, int H, int W,           // 배치 크기 N, 입력 채널 C, 높이 H, 너비 W
                              int K, int outC,                      // 커널 크기 K, 출력 채널 수 outC
                              int pad, int stride) {                // 패딩 pad, 스트라이드 stride
    int n   = blockIdx.z;                  // 배치 인덱스: 0 <= n < N
    int oc  = blockIdx.y;                  // 출력 채널 인덱스: 0 <= oc < outC
    int y   = threadIdx.y + blockIdx.x * blockDim.y; // 출력 높이 좌표: 0 <= y < H_out
    int x   = threadIdx.x + blockIdx.x * blockDim.x; // 출력 너비 좌표: 0 <= x < W_out
    int H_out = (H + 2*pad - K) / stride + 1;   // 출력 높이 계산
    int W_out = (W + 2*pad - K) / stride + 1;   // 출력 너비 계산
    if (n < N && oc < outC && y < H_out && x < W_out) {
        float sum = bias ? bias[oc] : 0.0f;        // 바이어스 더하기: scalar
        for (int ic = 0; ic < C; ++ic) {          // 입력 채널 순회
            for (int ky = 0; ky < K; ++ky) {      // 커널 높이 순회
                for (int kx = 0; kx < K; ++kx) {  // 커널 너비 순회
                    int in_y = y * stride + ky - pad;   // 입력 텐서 높이 좌표
                    int in_x = x * stride + kx - pad;   // 입력 텐서 너비 좌표
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        // input[n, ic, in_y, in_x] * weight[oc, ic, ky, kx]
                        sum += input[((n*C + ic)*H + in_y)*W + in_x] * 
                               weight[((oc*C + ic)*K + ky)*K + kx];
                    }
                }
            }
        }
        // output[n, oc, y, x] = sum
        output[((n*outC + oc)*H_out + y)*W_out + x] = sum;
    }
}

// 2D Max Pooling 커널
__global__ void maxpool2d_kernel(const float* __restrict__ input,   // 입력 텐서: [N, C, H, W]
                                 float* __restrict__ output,        // 출력 텐서: [N, C, H, W]
                                 int N, int C, int H, int W,        // N, C, H, W
                                 int poolK, int pad, int stride) {  // 풀 크기 poolK, 패딩 pad, 스트라이드 stride
    int n = blockIdx.z;                    // 배치 인덱스
    int c = blockIdx.y;                    // 채널 인덱스
    int y = threadIdx.y + blockIdx.x * blockDim.y; // 출력 높이 좌표
    int x = threadIdx.x + blockIdx.x * blockDim.x; // 출력 너비 좌표
    int H_out = (H + 2*pad - poolK) / stride + 1;  // 출력 높이
    int W_out = (W + 2*pad - poolK) / stride + 1;  // 출력 너비
    if (n < N && c < C && y < H_out && x < W_out) {
        float m = -FLT_MAX;                // 최대값 초기화
        for (int ky = 0; ky < poolK; ++ky) { // 풀링 윈도우 높이 순회
            for (int kx = 0; kx < poolK; ++kx) { // 풀링 윈도우 너비 순회
                int in_y = y * stride + ky - pad;  // 입력 높이 좌표
                int in_x = x * stride + kx - pad;  // 입력 너비 좌표
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    // input[n, c, in_y, in_x]
                    m = fmaxf(m, input[((n*C + c)*H + in_y)*W + in_x]);
                }
            }
        }
        // output[n, c, y, x] = m
        output[((n*C + c)*H_out + y)*W_out + x] = m;
    }
}

// Inception 모듈 구현 (호스트 코드)
void inception_module(const float* d_input,      // 입력 디바이스 텐서: [N, C, H, W]
                      float* d_output,           // 출력 디바이스 텐서: [N, F1+F3+F5+FPP, H, W]
                      const float* w1x1,         // 1x1 branch 가중치: [F1, C, 1, 1]
                      const float* b1x1,         // 1x1 branch 바이어스: [F1]
                      const float* w3r,          // 3x3 reduce 가중치: [F3R, C, 1, 1]
                      const float* b3r,          // 3x3 reduce 바이어스: [F3R]
                      const float* w3x3,         // 3x3 branch 가중치: [F3, F3R, 3, 3]
                      const float* b3x3,         // 3x3 branch 바이어스: [F3]
                      const float* w5r,          // 5x5 reduce 가중치: [F5R, C, 1, 1]
                      const float* b5r,          // 5x5 reduce 바이어스: [F5R]
                      const float* w5x5,         // 5x5 branch 가중치: [F5, F5R, 5, 5]
                      const float* b5x5,         // 5x5 branch 바이어스: [F5]
                      const float* wprj,         // pooling branch 가중치: [FPP, C, 1, 1]
                      const float* bprj,         // pooling branch 바이어스: [FPP]
                      int N, int C, int H, int W, // N, C, H, W
                      int F1, int F3R, int F3,    // 각 브랜치 채널 수
                      int F5R, int F5, int FPP) {
    // 브랜치별 출력 디바이스 포인터 할당
    float *d_b1, *d_b3r, *d_b3, *d_b5r, *d_b5, *d_pool, *d_pp;
    size_t size = N * H * W;
    cudaMalloc(&d_b1,  sizeof(float) * F1 * size);   // [N, F1, H, W]
    cudaMalloc(&d_b3r, sizeof(float) * F3R * size);  // [N, F3R, H, W]
    cudaMalloc(&d_b3,  sizeof(float) * F3 * size);   // [N, F3, H, W]
    cudaMalloc(&d_b5r, sizeof(float) * F5R * size);  // [N, F5R, H, W]
    cudaMalloc(&d_b5,  sizeof(float) * F5 * size);   // [N, F5, H, W]
    cudaMalloc(&d_pool, sizeof(float) * C  * size);   // [N, C, H, W] (풀링 후)
    cudaMalloc(&d_pp,   sizeof(float) * FPP* size);  // [N, FPP, H, W]

    dim3 block(16, 16);                            // 블록 크기: 16x16 쓰레드
    dim3 grid((W+15)/16, (H+15)/16, N);            // 그리드 크기: [W_out/16, H_out/16, N]

    // 1x1 분기
    conv2d_kernel<<<grid, block>>>(d_input, d_b1, w1x1, b1x1, N, C, H, W, 1, F1, 0, 1);
    // 3x3 분기: 1x1 축소 후 3x3
    conv2d_kernel<<<grid, block>>>(d_input, d_b3r, w3r, b3r, N, C, H, W, 1, F3R, 0, 1);
    conv2d_kernel<<<grid, block>>>(d_b3r,  d_b3,  w3x3, b3x3, N, F3R, H, W, 3, F3, 1, 1);
    // 5x5 분기: 1x1 축소 후 5x5
    conv2d_kernel<<<grid, block>>>(d_input, d_b5r, w5r, b5r, N, C, H, W, 1, F5R, 0, 1);
    conv2d_kernel<<<grid, block>>>(d_b5r,  d_b5,  w5x5, b5x5, N, F5R, H, W, 5, F5, 2, 1);
    // 풀링 분기: 3x3 풀링 후 1x1
    maxpool2d_kernel<<<grid, block>>>(d_input, d_pool, N, C, H, W, 3, 1, 1);
    conv2d_kernel<<<grid, block>>>(d_pool, d_pp, wprj, bprj, N, C, H, W, 1, FPP, 0, 1);

    // 결과 합치기 (채널 차원 concat)
    int offset = 0;
    // 브랜치 1 복사: [N, F1, H, W] -> [N, offset:offset+F1, H, W]
    cudaMemcpy(d_output + offset*size, d_b1, sizeof(float)*F1*size, cudaMemcpyDeviceToDevice);
    offset += F1;
    // 3x3 브랜치
    cudaMemcpy(d_output + offset*size, d_b3, sizeof(float)*F3*size, cudaMemcpyDeviceToDevice);
    offset += F3;
    // 5x5 브랜치
    cudaMemcpy(d_output + offset*size, d_b5, sizeof(float)*F5*size, cudaMemcpyDeviceToDevice);
    offset += F5;
    // 풀링 브랜치
    cudaMemcpy(d_output + offset*size, d_pp, sizeof(float)*FPP*size, cudaMemcpyDeviceToDevice);

    // 임시 버퍼 해제
    cudaFree(d_b1); cudaFree(d_b3r); cudaFree(d_b3);
    cudaFree(d_b5r); cudaFree(d_b5); cudaFree(d_pool); cudaFree(d_pp);
}
