#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <float.h>

// =====================
// Tiled 2D Convolution 커널 (공유메모리 사용)
// 입력:  input [N, C, H, W]
// weight: weight [Kout, C, Kh, Kw] (constant memory 권장)
// bias:   bias [Kout]
// 출력:  output [N, Kout, Ho, Wo]
// 타일 크기: TILE_DIM x TILE_DIM
// =====================
#define TILE_DIM 16  // 출력 타일 너비

__global__ void tiled_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int Kout, int Kh, int Kw,
    int pad, int stride)
{
    // 블록이 담당할 출력 타일의 좌상단 (x0, y0) 글로벌 좌표
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int n  = blockIdx.z;  // 배치 인덱스

    // 출력 크기 계산
    int Ho = (H + 2*pad - Kh) / stride + 1;
    int Wo = (W + 2*pad - Kw) / stride + 1;

    // 공유메모리 선언: (TILE_DIM + Kh -1) x (TILE_DIM + Kw -1)
    extern __shared__ float tile[];
    int Sh = TILE_DIM + Kh - 1;
    int Sw = TILE_DIM + Kw - 1;

    // 스레드 위치
    int tx = threadIdx.x;  // [0, TILE_DIM)
    int ty = threadIdx.y;  // [0, TILE_DIM)

    // 모든 출력 채널 k에 대해 반복
    for (int k = 0; k < Kout; ++k) {
        // 각 입력 채널 c 별 협력 로드
        for (int c = 0; c < C; ++c) {
            // 입력 타일 + halo 영역을 블록 내 스레드가 협력하여 로드
            for (int i = ty; i < Sh; i += blockDim.y) {
                for (int j = tx; j < Sw; j += blockDim.x) {
                    // global 좌표 (halo 포함)
                    int in_x = bx * stride + j - pad;
                    int in_y = by * stride + i - pad;
                    float v = 0.0f;
                    // 유효 범위 내에 있을 때만 로드
                    if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                        // flatten 인덱스: 배치 n, 채널 c, 행 in_y, 열 in_x
                        v = input[n*C*H*W + c*H*W + in_y*W + in_x];
                    }
                    tile[i * Sw + j] = v; // shared memory에 저장
                }
            }
            __syncthreads(); // 모든 스레드가 tile 로드 완료 대기

            // 타일 내 합성곱 계산: 출력 픽셀 하나 per 스레드
            int out_x = bx + tx;
            int out_y = by + ty;
            if (out_x < Wo && out_y < Ho) {
                float sum = bias[k]; // 채널별 편향 초기값
                for (int i = 0; i < Kh; ++i) {
                    for (int j = 0; j < Kw; ++j) {
                        // shared memory 내에서 해당 위치 값 로드
                        float val = tile[(ty*stride + i) * Sw + (tx*stride + j)];
                        // weight[k, c, i, j]
                        float w = weight[k*C*Kh*Kw + c*Kh*Kw + i*Kw + j];
                        sum += val * w;
                    }
                }
                // flatten 인덱스: 배치 n, 채널 k, 행 out_y, 열 out_x
                output[n*Kout*Ho*Wo + k*Ho*Wo + out_y*Wo + out_x] = sum;
            }
            __syncthreads(); // 다음 채널 전환 대기
        }
    }
}
