#include <cuda_runtime.h> // CUDA 런타임 헤더
#include <device_launch_parameters.h> // CUDA 런치 파라미터 헤더
#include <math.h> // 수학 함수 헤더

// QKV projection 커널
__global__
void qkv_projection_kernel(const float* input, const float* Wq, const float* Wk, const float* Wv,
                           float* Q, float* K, float* V,
                           int B, int T, int D) {
    int b = blockIdx.x; // b: 배치 인덱스, input/Q/K/V shape [B, T, D]
    int t = blockIdx.y; // t: 시퀀스 위치 인덱스, input/Q/K/V shape [B, T, D]
    int d = threadIdx.x; // d: 모델 차원 인덱스, input/Q/K/V shape [B, T, D]
    int idx = (b * T + t) * D + d; // idx: 1D 인덱스, maps to [b, t, d]
    float x = input[idx]; // input[b, t, d]
    float q = 0.0f; // Q[b, t, d]
    for (int k = 0; k < D; ++k) {
        q += input[(b * T + t) * D + k] * Wq[k * D + d]; // input[b, t, k] * Wq[k, d]
    }
    Q[idx] = q; // Q[b, t, d]
    float k_val = 0.0f; // K[b, t, d]
    for (int k = 0; k < D; ++k) {
        k_val += input[(b * T + t) * D + k] * Wk[k * D + d]; // input[b, t, k] * Wk[k, d]
    }
    K[idx] = k_val; // K[b, t, d]
    float v_val = 0.0f; // V[b, t, d]
    for (int k = 0; k < D; ++k) {
        v_val += input[(b * T + t) * D + k] * Wv[k * D + d]; // input[b, t, k] * Wv[k, d]
    }
    V[idx] = v_val; // V[b, t, d]
}

// Scaled Dot-Product Attention 커널
__global__
void sdpa_kernel(const float* Q, const float* K, const float* V, float* O,
                 int B, int T, int D, int H) {
    extern __shared__ float scores[]; // scores[T]
    int d_h = D / H; // 각 헤드 차원 크기
    int b = blockIdx.x; // 배치 인덱스
    int h = blockIdx.y; // 헤드 인덱스
    int t = blockIdx.z; // 시퀀스 위치 인덱스
    int dh = threadIdx.x; // 헤드 내부 차원 인덱스
    const float* Qh = Q + (b * T * D) + t * D + h * d_h; // Q[b, t, h*d_h : h*d_h+d_h]
    const float* Kh = K + (b * T * D) + 0 * D + h * d_h; // K[b, :, h*d_h : h*d_h+d_h]
    if (dh == 0) {
        for (int j = 0; j < T; ++j) {
            float dot = 0.0f; // 점곱
            for (int k = 0; k < d_h; ++k) {
                dot += Qh[k] * Kh[j * D + k]; // Q[b,t,h,k] * K[b,j,h,k]
            }
            scores[j] = dot / sqrtf((float)d_h); // 정규화
        }
        float max_score = scores[0]; // softmax 최대값
        for (int j = 1; j < T; ++j) if (scores[j] > max_score) max_score = scores[j];
        float sum_exp = 0.0f; // exp 합산
        for (int j = 0; j < T; ++j) {
            scores[j] = expf(scores[j] - max_score); // exp
            sum_exp += scores[j];
        }
        for (int j = 0; j < T; ++j) scores[j] /= sum_exp; // softmax 확률
    }
    __syncthreads();
    float out = 0.0f; // O[b, t, h, dh]
    for (int j = 0; j < T; ++j) {
        out += scores[j] * V[(b * T * D) + j * D + h * d_h + dh]; // V[b,j,h,dh]
    }
    O[(b * T * D) + t * D + h * d_h + dh] = out; // O[b,t,h,dh]
}

// Output projection 커널
__global__
void output_projection_kernel(const float* O, const float* Wo, float* output,
                              int B, int T, int D) {
    int b = blockIdx.x; // 배치 인덱스
    int t = blockIdx.y; // 시퀀스 위치 인덱스
    int d = threadIdx.x; // 모델 차원 인덱스
    int idx = (b * T + t) * D + d; // 1D 인덱스
    float sum = 0.0f; // output[b, t, d]
    for (int k = 0; k < D; ++k) {
        sum += O[b * T * D + t * D + k] * Wo[k * D + d]; // O[b,t,k] * Wo[k,d]
    }
    output[idx] = sum; // output[b, t, d]
}

// Host 함수
void multi_head_attention(const float* input, const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                          float* output, int B, int T, int D, int H, cudaStream_t stream = 0) {
    int size = B * T * D; // 전체 요소 수
    float *Q, *K, *V, *O; // 중간 버퍼
    cudaMalloc(&Q, size * sizeof(float)); // Q: [B, T, D]
    cudaMalloc(&K, size * sizeof(float)); // K: [B, T, D]
    cudaMalloc(&V, size * sizeof(float)); // V: [B, T, D]
    cudaMalloc(&O, size * sizeof(float)); // O: [B, T, D]
    dim3 grid1(B, T); // 그리드 크기 (B, T)
    dim3 block1(D);   // 블록 크기 (D)
    qkv_projection_kernel<<<grid1, block1, 0, stream>>>(input, Wq, Wk, Wv, Q, K, V, B, T, D);
    dim3 grid2(B, H, T); // 그리드 크기 (B, H, T)
    dim3 block2(D / H);  // 블록 크기 (D/H)
    int shared_mem = T * sizeof(float); // 공유 메모리 크기
    sdpa_kernel<<<grid2, block2, shared_mem, stream>>>(Q, K, V, O, B, T, D, H);
    dim3 grid3(B, T); // 그리드 크기 (B, T)
    dim3 block3(D);   // 블록 크기 (D)
    output_projection_kernel<<<grid3, block3, 0, stream>>>(O, Wo, output, B, T, D);
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); // 메모리 해제
}
