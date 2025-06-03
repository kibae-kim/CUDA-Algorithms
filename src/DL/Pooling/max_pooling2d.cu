%%writefile max_pooling2d.cu
#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
 * 2-D Max-Pooling 커널
 *   input  : [N, C, H, W]
 *   output : [N, C, Ho, Wo]
 *   Ho = ((pad + H + pad) - Ph) / stride + 1
 *   Wo = ((pad + W + pad) - Pw) / stride + 1
 */
 
__global__ void max_pooling2d_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int  N, int C, int H, int W,
    int  Ph, int Pw,
    int  pad, int stride)
{
    /* -------- 1. 출력 feature-map 좌표 계산 -------- */
    const int Ho = ((pad + H + pad - Ph) / stride) + 1;
    const int Wo = ((pad + W + pad - Pw) / stride) + 1;

    const int n      = blockIdx.z;           // 배치
    const int c      = blockIdx.y;           // 채널
    const int out_y  = blockIdx.x * blockDim.y + threadIdx.y;
    const int out_x  = threadIdx.x;

    if (out_y >= Ho || out_x >= Wo) return;

    /* -------- 2. (out_y, out_x)의 풀링 윈도우 ↔ 입력 인덱스 -------- */
    const int in_y_origin = out_y * stride - pad;
    const int in_x_origin = out_x * stride - pad;

    float max_val = -FLT_MAX;

    for (int ky = 0; ky < Ph; ++ky) {
        int in_y = in_y_origin + ky;
        if (in_y < 0 || in_y >= H) continue;

        for (int kx = 0; kx < Pw; ++kx) {
            int in_x = in_x_origin + kx;
            if (in_x < 0 || in_x >= W) continue;

            float val = input[n*C*H*W + c*H*W + in_y*W + in_x];
            max_val   = fmaxf(max_val, val);
        }
    }

    output[n*C*Ho*Wo + c*Ho*Wo + out_y*Wo + out_x] = max_val;
}

/* ─────────────────────────────
 *  단독 테스트 / 예시 용도
 *  (conv 출력 → pool 연결 시에 shape 만
 *   맞춰서 호출만 해주면 됩니다.)
 * ───────────────────────────── */
int main() {
    /* 1. 파라미터 ----------------------------- */
    constexpr int N = 1, C = 1, H = 4, W = 4;   // 입력 4×4
    constexpr int Ph = 2, Pw = 2;               // 2×2 Max-Pool
    constexpr int pad = 0, stride = 2;          // classic 2×2/stride 2
    constexpr int Ho  = ((pad + H + pad - Ph) / stride) + 1; // 2
    constexpr int Wo  = ((pad + W + pad - Pw) / stride) + 1; // 2

    /* 2. 호스트 버퍼 --------------------------- */
    float h_in[H*W] = { 1, 2, 3, 4,
                        5, 6, 7, 8,
                        9,10,11,12,
                       13,14,15,16 };
    float h_out[Ho*Wo] = {0};

    /* 3. 디바이스 메모리 ----------------------- */
    float *d_in, *d_out;
    cudaMalloc(&d_in , sizeof(h_in ));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);

    /* 4. 커널 실행 ---------------------------- */
    dim3 block(/*x*/ Wo, /*y*/ 1);          // (out_x)
    dim3 grid (/*x*/ (Ho + block.y - 1) / block.y,
               /*y*/ C,
               /*z*/ N);

    max_pooling2d_kernel<<<grid, block>>>(d_in, d_out,
        N, C, H, W, Ph, Pw, pad, stride);
    cudaDeviceSynchronize();

    /* 5. 결과 확인 ---------------------------- */
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    printf("Max-Pool Output (%d×%d):\n", Ho, Wo);
    for (int i = 0; i < Ho; ++i) {
        for (int j = 0; j < Wo; ++j)
            printf("%6.1f ", h_out[i*Wo + j]);
        printf("\n");
    }

    /* 6. 자원 반환 ---------------------------- */
    cudaFree(d_in);  cudaFree(d_out);
    return 0;
}
