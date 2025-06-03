// ────────────────────────────────────────────────────────
// tiled_conv2d.cu  (추가 -- backward kernels)
// ────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

/* -------------------------------------------------------
   Helper:  index calc to keep 코드 짧게
------------------------------------------------------- */
__device__ __forceinline__
int idx4(int n,int c,int h,int w,int C,int H,int W){return ((n*C+c)*H+h)*W+w;}

/* =======================================================
   1) ∇x  (gradient w.r.t. input feature map)
   -------------------------------------------------------
   각 thread → 하나의 (n,cin,h,w) 를 담당
   Ho,Wo 는 forward 와 같은 식으로 미리 계산
======================================================= */
__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ dOut,     // [N,Kout,Ho,Wo]
    const float* __restrict__ weight,   // [Kout,C,Kh,Kw]
    float* __restrict__ dInput,         // [N,C,H,W]  (output)
    int N,int C,int H,int W,
    int Kout,int Kh,int Kw,
    int pad,int stride)
{
    const int n   = blockIdx.z;
    const int cin = blockIdx.y*blockDim.y + threadIdx.y;
    const int h   = blockIdx.x*blockDim.x + threadIdx.x;

    if(cin>=C || h>=H) return;

    for(int w=0; w<W; ++w){
        float sum = 0.f;

        // 역전파: ∑_{kout,ki,kj} dOut * w_flip
        for(int kout=0; kout<Kout; ++kout){
            for(int ki=0; ki<Kh; ++ki){
                int ho = (h + pad - ki);
                if(ho % stride) continue;
                ho /= stride;
                if(ho<0)       continue;
                for(int kj=0; kj<Kw; ++kj){
                    int wo = (w + pad - kj);
                    if(wo % stride) continue;
                    wo /= stride;
                    if(wo<0)       continue;

                    // dOut 경계
                    // Ho Wo 는 forward 식
                    const int Ho = ((pad+H+pad)-Kh)/stride + 1;
                    const int Wo = ((pad+W+pad)-Kw)/stride + 1;
                    if(ho>=Ho || wo>=Wo) continue;

                    float grad_out = dOut[((n*Kout+kout)*Ho + ho)*Wo + wo];
                    float w_val    = weight[((kout*C+cin)*Kh + ki)*Kw + kj];
                    sum += grad_out * w_val;
                }
            }
        }
        dInput[idx4(n,cin,h,w,C,H,W)] = sum;
    }
}

/* =======================================================
   2) ∇w, ∇b  (gradient w.r.t. weight / bias)
   -------------------------------------------------------
   * 하나의 thread → 하나의 (kout,cin,ki,kj)
   * 여러 배치/공간 위치에 대한 합산은               \
         atomicAdd 로 전역 메모리에 누적.            /
   * dBias 는 (kout) 당 하나의 값이므로              \
         같은 그리드에서 병렬로 처리 (atomic)       /
======================================================= */
__global__ void conv2d_backward_weight_bias_kernel(
    const float* __restrict__ dOut,     // [N,Kout,Ho,Wo]
    const float* __restrict__ input,    // [N,C,H,W]
    float* __restrict__ dWeight,        // [Kout,C,Kh,Kw] (output)
    float* __restrict__ dBias,          // [Kout]         (output)
    int N,int C,int H,int W,
    int Kout,int Kh,int Kw,
    int pad,int stride)
{
    const int kout = blockIdx.z;              // 한 그리드 z축 → 한 출력채널
    const int cin  = blockIdx.y;              // y축 → 입력채널
    const int tid  = threadIdx.x;             // 0‥Kh*Kw-1 중 하나
    const int ki   = tid / Kw;
    const int kj   = tid % Kw;
    if(ki>=Kh) return;

    // Ho,Wo forward 와 동일
    const int Ho = ((pad+H+pad)-Kh)/stride + 1;
    const int Wo = ((pad+W+pad)-Kw)/stride + 1;

    float w_grad = 0.f;
    float b_grad = 0.f;

    for(int n=0;n<N;++n){
        for(int ho=0;ho<Ho;++ho){
            int h_in = ho*stride - pad + ki;
            if(h_in<0 || h_in>=H) continue;
            for(int wo=0;wo<Wo;++wo){
                int w_in = wo*stride - pad + kj;
                if(w_in<0 || w_in>=W) continue;

                float grad_out = dOut[((n*Kout+kout)*Ho + ho)*Wo + wo];
                float in_val   = input[idx4(n,cin,h_in,w_in,C,H,W)];

                w_grad += grad_out * in_val;

                if(cin==0 && tid==0)          // b_grad는 (kout) 당 한 번만 더함
                    b_grad += grad_out;
            }
        }
    }
    // 병렬 누적
    atomicAdd(&dWeight[(((kout*C)+cin)*Kh + ki)*Kw + kj], w_grad);
    if(cin==0 && tid==0)
        atomicAdd(&dBias[kout], b_grad);
}
