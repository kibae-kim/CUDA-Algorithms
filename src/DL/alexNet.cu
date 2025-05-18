#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <float.h>

// 2D Convolution 커널 함수
// 입력: input [N, C, H, W]
// 가중치: weight [Kout, C, Kh, Kw]
// 편향: bias [Kout]
// 출력: output [N, Kout, Ho, Wo]
__global__ void conv2d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int N, int C, int H, int W,
                              int Kout, int Kh, int Kw,
                              int pad, int stride) {
    int n = blockIdx.x;           // 배치 인덱스 n ∈ [0, N)
    int k = blockIdx.y;           // 출력 채널 인덱스 k ∈ [0, Kout)
    int out_idx = blockIdx.z;     // Ho*Wo를 flatten한 인덱스
    int Ho = (H + 2*pad - Kh)/stride + 1; // 출력 높이
    int Wo = (W + 2*pad - Kw)/stride + 1; // 출력 너비
    int out_y = out_idx / Wo;     // 출력 y 좌표
    int out_x = out_idx % Wo;     // 출력 x 좌표

    if (n < N && k < Kout && out_y < Ho && out_x < Wo) {
        float sum = bias[k];       // 초기값으로 편향 설정

        // 입력 채널과 커널 공간 순회
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < Kh; ++i) {
                for (int j = 0; j < Kw; ++j) {
                    int in_y = out_y*stride + i - pad; // 입력 y 좌표
                    int in_x = out_x*stride + j - pad; // 입력 x 좌표
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        // input[n, c, in_y, in_x]
                        float val = input[n*C*H*W + c*H*W + in_y*W + in_x];
                        // weight[k, c, i, j]
                        float w = weight[k*C*Kh*Kw + c*Kh*Kw + i*Kw + j];
                        sum += val * w;
                    }
                }
            }
        }
        // output[n, k, out_y, out_x]
        output[n*Kout*Ho*Wo + k*Ho*Wo + out_y*Wo + out_x] = sum;
    }
}

// ReLU 활성화 함수 커널
// 입력/출력: data [size]
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x; // 평탄화된 인덱스
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f); // ReLU 적용
    }
}

// Max Pooling 커널 함수
// 입력: input [N, C, H, W]
// 출력: output [N, C, Ho, Wo]
// poolH, poolW: 풀링 커널 크기, stride: 풀링 스트라이드
__global__ void maxpool_kernel(const float* input, float* output,
                               int N, int C, int H, int W,
                               int poolH, int poolW, int stride) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;     // 평탄화된 출력 인덱스
    int Ho = (H - poolH)/stride + 1;                  // 출력 높이
    int Wo = (W - poolW)/stride + 1;                  // 출력 너비
    int total = N * C * Ho * Wo;                      // 전체 연산 수

    if (idx < total) {
        int p = idx;
        int ow = p % Wo; p /= Wo;                     // 출력 x 좌표
        int oh = p % Ho; p /= Ho;                     // 출력 y 좌표
        int c  = p % C;  p /= C;                      // 채널 인덱스
        int n  = p;                                   // 배치 인덱스

        int h_start = oh*stride;                      // 입력 시작 y
        int w_start = ow*stride;                      // 입력 시작 x
        float max_val = -FLT_MAX;                     // 초기값

        for (int i = 0; i < poolH; ++i) {
            for (int j = 0; j < poolW; ++j) {
                int in_y = h_start + i;               // 입력 y
                int in_x = w_start + j;               // 입력 x
                float v = input[n*C*H*W + c*H*W + in_y*W + in_x]; // input[n,c,in_y,in_x]
                if (v > max_val) max_val = v;
            }
        }
        // output[n, c, oh, ow]
        output[idx] = max_val;
    }
}

// Fully Connected 레이어 커널
// 입력: input [N, inFeatures]
// 가중치: weight [outFeatures, inFeatures]
// 편향: bias [outFeatures]
// 출력: output [N, outFeatures]
__global__ void fc_kernel(const float* input, const float* weight, const float* bias,
                          float* output, int N, int inF, int outF) {
    int n = blockIdx.x;          // 배치 인덱스
    int o = blockIdx.y;          // 출력 특징 인덱스
    if (n < N && o < outF) {
        float sum = bias[o];     // 편향 초기화
        for (int i = 0; i < inF; ++i) {
            // input[n, i] * weight[o, i]
            sum += input[n*inF + i] * weight[o*inF + i];
        }
        // output[n, o]
        output[n*outF + o] = sum;
    }
}

// Softmax 커널
// 입력/출력: data [N, C]
__global__ void softmax_kernel(float* data, int N, int C) {
    int n = blockIdx.x; // 배치 인덱스
    if (n < N) {
        float maxv = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            maxv = fmaxf(maxv, data[n*C + c]);
        }
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            data[n*C + c] = expf(data[n*C + c] - maxv);
            sum += data[n*C + c];
        }
        for (int c = 0; c < C; ++c) {
            data[n*C + c] /= sum; // 확률값
        }
    }
}

int main() {
    // 배치 크기 N=1로 고정
    int N = 1; // [N]

    // ========== Conv1 설정 ==========
    // 입력 텐서: [1,3,227,227]
    int C1=3, H1=227, W1=227;
    // Conv1 가중치: [96,3,11,11], stride=4, pad=0 -> 출력 [1,96,55,55]
    int K1=96, Kh1=11, Kw1=11, pad1=0, stride1=4;
    int Ho1=(H1+2*pad1-Kh1)/stride1+1; // 55
    int Wo1=(W1+2*pad1-Kw1)/stride1+1; // 55

    // 메모리 크기 계산
    size_t szInput1 = N*C1*H1*W1 * sizeof(float);    // input 크기
    size_t szW1     = K1*C1*Kh1*Kw1 * sizeof(float); // weight1 크기
    size_t szB1     = K1 * sizeof(float);            // bias1 크기
    size_t szOut1   = N*K1*Ho1*Wo1 * sizeof(float);  // output1 크기

    // GPU 메모리 할당
    float *d_input1, *d_w1, *d_b1, *d_out1;
    cudaMalloc(&d_input1, szInput1);                // input1 메모리
    cudaMalloc(&d_w1,     szW1);                    // weight1 메모리
    cudaMalloc(&d_b1,     szB1);                    // bias1 메모리
    cudaMalloc(&d_out1,   szOut1);                  // output1 메모리

    // (가중치와 입력은 외부에서 로드했다고 가정)

    // Conv1 실행: grid=(N,K1,Ho1*Wo1)
    dim3 grid1(N, K1, Ho1*Wo1);
    conv2d_kernel<<<grid1,1>>>(d_input1, d_w1, d_b1, d_out1,
                               N,C1,H1,W1,K1,Kh1,Kw1,pad1,stride1);
    cudaDeviceSynchronize();

    // ReLU1: 출력 [1,96,55,55] 요소 수 = N*K1*Ho1*Wo1
    int sizeRelu1 = N*K1*Ho1*Wo1;
    relu_kernel<<<(sizeRelu1+255)/256,256>>>(d_out1, sizeRelu1);
    cudaDeviceSynchronize();

    // Pool1: 입력 [1,96,55,55], 풀링 3x3, stride=2 -> 출력 [1,96,27,27]
    int poolH1=3, poolW1=3, strideP1=2;\
