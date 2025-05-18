#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 1) 2D Convolution Kernel
__global__ void conv2d_kernel(
    const float* __restrict__ input,   // 입력 텐서: [N][C_in][H][W]
    float* __restrict__ output,        // 출력 텐서: [N][C_out][H][W]
    const float* __restrict__ weight,  // 가중치 텐서: [C_out][C_in][K][K]
    const float* __restrict__ bias,    // 바이어스 벡터: [C_out]
    int N, int C_in, int H, int W,
    int C_out, int K, int pad)          // pad = K/2 (same padding)
{
    // -------------------------------------------------------------
    // 스레드별로 처리할 출력 픽셀 좌표 계산
    int px      = blockIdx.x * blockDim.x + threadIdx.x;  // 가로 위치: 0 ≤ px < W
    int py      = blockIdx.y * blockDim.y + threadIdx.y;  // 세로 위치: 0 ≤ py < H
    int oc_flat = blockIdx.z;                             // flat index: 0 ≤ oc_flat < N * C_out
    int n       = oc_flat / C_out;                        // 배치 인덱스 n: 0 ≤ n < N
    int oc      = oc_flat % C_out;                        // 출력 채널 인덱스 oc: 0 ≤ oc < C_out
    // -------------------------------------------------------------
    if (n < N && oc < C_out && py < H && px < W) {
        // bias[oc] 불러오기 (없으면 0)
        float sum = bias ? bias[oc] : 0.0f;  // sum 초기값

        // 모든 입력 채널에 대해 커널 윈도우 순회
        for (int ic = 0; ic < C_in; ++ic) {     // 입력 채널 0 ≤ ic < C_in
            for (int ky = 0; ky < K; ++ky) {    // 커널 세로 0 ≤ ky < K
                for (int kx = 0; kx < K; ++kx) {// 커널 가로 0 ≤ kx < K
                    int iy = py + ky - pad;      // 대응하는 입력 y: (py + ky − pad)
                    int ix = px + kx - pad;      // 대응하는 입력 x: (px + kx − pad)
                    // 경계 체크 (same padding)
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        // 입력 인덱스 계산:
                        // ((n * C_in + ic) * H + iy) * W + ix
                        float in_val = input[((n * C_in + ic) * H + iy) * W + ix];
                        // 가중치 인덱스 계산:
                        // ((oc * C_in + ic) * K + ky) * K + kx
                        float w = weight[((oc * C_in + ic) * K + ky) * K + kx];
                        sum += in_val * w;          // 누적 합산
                    }
                }
            }
        }
        // 결과 저장:
        // output[((n * C_out + oc) * H + py) * W + px] = sum
        output[((n * C_out + oc) * H + py) * W + px] = sum;
    }
}

// 2) ReLU Activation Kernel
__global__ void relu_kernel(
    float* data,     // 처리할 텐서(flatten): [N*C*H*W]
    int total)       // 총 요소 수: N * C * H * W
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 플랫 인덱스
    if (idx < total) {
        float v = data[idx];                           // 원소 값 로드
        data[idx] = v > 0.0f ? v : 0.0f;               // ReLU: max(v, 0)
    }
}

// 3) Skip Connection 덧셈 Kernel
__global__ void add_kernel(
    const float* x,      // 입력 텐서 X: [N*C*H*W]
    const float* f,      // 변환된 텐서 F(X): [N*C*H*W]
    float* out,          // 결과 저장 텐서: [N*C*H*W]
    int total)           // 총 요소 수
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = x[idx] + f[idx];  // X + F(X)
    }
}

// 4) Host 함수: Residual Learning Block 실행
void residual_block(
    float* d_input,        // 디바이스 입력: [N][C][H][W]
    float* d_output,       // 디바이스 출력: [N][C][H][W]
    const float* d_w1,     // Conv1 weight: [C][C][K][K]
    const float* d_b1,     // Conv1 bias: [C]
    const float* d_w2,     // Conv2 weight: [C][C][K][K]
    const float* d_b2,     // Conv2 bias: [C]
    int N, int C, int H, int W, int K)
{
    // 중간 버퍼 할당: [N][C][H][W]
    float* d_temp;
    cudaMalloc(&d_temp, sizeof(float) * N * C * H * W);

    // 그리드·블록 설정 (Conv 전용)
    dim3 block(16, 16, 1);                                      // blockDim: 16×16
    dim3 grid((W + block.x - 1) / block.x,                      // gridDim.x = ceil(W/16)
              (H + block.y - 1) / block.y,                      // gridDim.y = ceil(H/16)
              N * C);                                           // gridDim.z = N * C (배치×채널)
    int total   = N * C * H * W;                                // 플랫 요소 수
    int threads = 256;                                          // ReLU/Add용 스레드 수
    int blocks  = (total + threads - 1) / threads;              // ReLU/Add용 블록 수

    // ---- 1) 첫 번째 Conv + bias ----
    conv2d_kernel<<<grid, block>>>(
        d_input, d_temp, d_w1, d_b1,
        N, C, H, W, C, K, K/2);

    // ---- 2) 첫 번째 ReLU ----
    relu_kernel<<<blocks, threads>>>(d_temp, total);

    // ---- 3) 두 번째 Conv + bias ----
    conv2d_kernel<<<grid, block>>>(
        d_temp, d_temp, d_w2, d_b2,
        N, C, H, W, C, K, K/2);

    // ---- 4) Skip Connection: X + F(X) ----
    add_kernel<<<blocks, threads>>>(d_input, d_temp, d_temp, total);

    // ---- 5) 두 번째 ReLU ----
    relu_kernel<<<blocks, threads>>>(d_temp, total);

    // 최종 결과 복사
    cudaMemcpy(d_output, d_temp,
               sizeof(float) * total,
               cudaMemcpyDeviceToDevice);

    cudaFree(d_temp);
}
