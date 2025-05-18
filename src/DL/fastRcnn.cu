// faster_rcnn_cuda.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// conv2d_kernel: 2D 컨볼루션 연산 커널
// input: [C, H, W]
// weight: [outC, C, K, K]
// bias: [outC]
// output: [outC, H_out, W_out]
__global__ void conv2d_kernel(
    const float* __restrict__ input,   // 입력 피처 맵 텐서 (C × H × W)
    float* __restrict__ output,        // 출력 피처 맵 텐서 (outC × H_out × W_out)
    const float* __restrict__ weight,  // 필터 가중치 텐서 (outC × C × K × K)
    const float* __restrict__ bias,    // 편향 텐서 (outC)
    int C, int H, int W,
    int K, int outC,
    int pad, int stride
) {
    int oc = blockIdx.z;                                   // 출력 채널 인덱스 (0 ≤ oc < outC)
    int y  = blockIdx.y * blockDim.y + threadIdx.y;       // 출력 y 좌표 (0 ≤ y < H_out)
    int x  = blockIdx.x * blockDim.x + threadIdx.x;       // 출력 x 좌표 (0 ≤ x < W_out)
    if (oc < outC && y < H && x < W) {
        float sum = bias ? bias[oc] : 0.0f;                // 초기값: 편향 (bias[oc]) 또는 0
        for (int ic = 0; ic < C; ++ic) {                   // 입력 채널 반복 (0 ≤ ic < C)
            for (int ky = 0; ky < K; ++ky) {               // 커널 높이 반복
                for (int kx = 0; kx < K; ++kx) {           // 커널 너비 반복
                    int in_y = y * stride + ky - pad;      // 입력 y 좌표 계산
                    int in_x = x * stride + kx - pad;      // 입력 x 좌표 계산
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        // 입력 텐서 접근: input[(ic*H + in_y)*W + in_x]
                        float val = input[(ic * H + in_y) * W + in_x];
                        // 가중치 텐서 접근: weight[(((oc*C)+ic)*K+ky)*K + kx]
                        float w = weight[((oc * C + ic) * K + ky) * K + kx];
                        sum += val * w;                          // 누적합
                    }
                }
            }
        }
        // 출력 텐서 저장: output[(oc*H + y)*W + x]
        output[(oc * H + y) * W + x] = sum;
    }
}

// RPN 분류 헤드 커널
// feature_map: [C, H, W]
// rpn_cls_weight: [2*A, C, 1, 1]
// rpn_cls_bias: [2*A]
// rpn_cls_out: [2*A, H, W]
__global__ void rpn_cls_kernel(
    const float* feature_map,          // RPN 입력 피처 맵 (C × H × W)
    float* rpn_cls_out,                // RPN 객체 확률 맵 (2A × H × W)
    const float* rpn_cls_weight,       // RPN 분류 가중치 (2A × C × 1 × 1)
    const float* rpn_cls_bias,         // RPN 분류 편향 (2A)
    int C, int H, int W,
    int A                              // 앵커 수
) {
    int a2 = blockIdx.z;               // 앵커*2 채널 인덱스 (0 ≤ a2 < 2A)
    int y  = blockIdx.y * blockDim.y + threadIdx.y; // y 좌표
    int x  = blockIdx.x * blockDim.x + threadIdx.x; // x 좌표
    if (a2 < 2 * A && y < H && x < W) {
        float sum = rpn_cls_bias[a2];  // 초기값: bias
        for (int c = 0; c < C; ++c) {
            // 1×1 컨볼루션: 가중치 인덱스[(a2*C + c)*1*1]
            float w = rpn_cls_weight[a2 * C + c];
            float v = feature_map[(c * H + y) * W + x];
            sum += v * w;
        }
        // 분류 점수 저장
        rpn_cls_out[(a2 * H + y) * W + x] = sum;
    }
}

// RoI Align 풀링 커널 (간단화된 버전)
// feature_map: [C, H, W]
// rois: [N, 5] (batch_index, x1, y1, x2, y2)
// pooled: [N, C, P, P]
__global__ void roi_align_kernel(
    const float* feature_map,          // 입력 피처 맵 (C × H × W)
    const float* rois,                 // RoI 좌표 (N × 5)
    float* pooled,                     // 풀링 결과 (N × C × P × P)
    int N, int C, int H, int W,
    int P,                             // 출력 크기 (P × P)
    float spatial_scale
) {
    int roi_id  = blockIdx.z;          // RoI 인덱스 (0 ≤ roi_id < N)
    int c       = blockIdx.y;          // 채널 인덱스 (0 ≤ c < C)
    int py      = threadIdx.y;         // 풀링 y 좌표 (0 ≤ py < P)
    int px      = threadIdx.x;         // 풀링 x 좌표 (0 ≤ px < P)
    if (roi_id < N && c < C && py < P && px < P) {
        const float* roi_ptr = rois + roi_id * 5;
        int batch_ind = (int)roi_ptr[0];                // 배치 인덱스
        float x1 = roi_ptr[1] * spatial_scale;          // x1 좌표 변환
        float y1 = roi_ptr[2] * spatial_scale;          // y1 좌표 변환
        float x2 = roi_ptr[3] * spatial_scale;          // x2 좌표 변환
        float y2 = roi_ptr[4] * spatial_scale;          // y2 좌표 변환
        float roi_w = max(x2 - x1, 1.0f);
        float roi_h = max(y2 - y1, 1.0f);
        float bin_size_h = roi_h / P;
        float bin_size_w = roi_w / P;
        float start_y = y1 + py * bin_size_h;
        float start_x = x1 + px * bin_size_w;
        // 단일 포인트 샘플링 단순화
        int in_y = min(max((int)(start_y), 0), H-1);
        int in_x = min(max((int)(start_x), 0), W-1);
        // 풀링 결과 저장: pooled[((roi_id*C + c)*P + py)*P + px]
        pooled[((roi_id * C + c) * P + py) * P + px] = feature_map[(c * H + in_y) * W + in_x];
    }
}

// 분류 헤드 (FC) 커널
// pooled: [N, C, P, P]
// cls_score: [N, num_classes]
__global__ void fc_cls_kernel(
    const float* pooled,               // 풀링된 피처 (N × C × P × P)
    float* cls_score,                  // 클래스 점수 (N × num_classes)
    const float* fc_weight,            // 가중치 (num_classes × (C*P*P))
    const float* fc_bias,              // 편향 (num_classes)
    int N, int C, int P,
    int num_classes
) {
    int n = blockIdx.x;                 // RoI 인덱스 (0 ≤ n < N)
    int k = threadIdx.x;                // 클래스 인덱스 (0 ≤ k < num_classes)
    if (n < N && k < num_classes) {
        float sum = fc_bias[k];         // 편향
        int feat_size = C * P * P;
        for (int i = 0; i < feat_size; ++i) {
            float v = pooled[n * feat_size + i];
            float w = fc_weight[k * feat_size + i];
            sum += v * w;                 // FC 연산
        }
        cls_score[n * num_classes + k] = sum;
    }
}

// TODO: bbox regression 헤드, NMS, 후처리 등을 추가적으로 구현 예정
// 호스트 코드에서 CUDA 메모리 할당, 커널 호출, 데이터 전송 등을 처리해야함
