%%writefile max_pooling2d_dynamic.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

__global__ void maxpool2d_kernel(
        const float* __restrict__ input,   // [N,C,H,W]
        float*       __restrict__ output,  // [N,C,Ho,Wo]
        int N, int C, int H, int W,
        int pooling_window_Height, int pooling_window_Width,
        int pad, int stride)
{
    // 출력 피쳐맵이 그것의 행과 열 방향으로 몇개의 픽셀을 가지고 있는지 계산
    // 하나의 스레드가 하나의 윈도우에서 최댓값을 탐색해서 추출함으로써 하나의 출력픽셀을 산출
    // 출력피쳐맵의 전체 픽셀 개수가 맥스풀링을 수행하는 총 횟수와 일치하게 됨
    const int Ho = ((pad + H + pad) - pooling_window_Height) / stride + 1;
    const int Wo = ((pad + W + pad) - pooling_window_Width) / stride + 1;
    const int totalPoolingTime = Ho * Wo;
    
    // 먼저 스레드블록의 단위에서 생각해보면,
    // z방향으로 각 스레드 블록이 배치안의 한 이미지안의 특정 채널인 (n,c)조합을 몇개 다루는지 결정해야함
    // (n,c)조합의 총 개수인 N*C보다 z방향으로의 스레드블록의 개수가 큰지 작은지 따져서  
    // 큰 경우에는 하나의 (n,c)조합이 여러개의 스레드블록에 의해 샘플링(풀링)될 수 있도록 하고
    // 작은 경우에는 하나의 스레드블록이 하나의 (n,c)를 풀링한 후 다음(n,c)로 이동해서 풀링을 수행함

    // 스레드 단위에서 생각해보면, 스레드블록의 개수가 총 (n,c)조합의 개수보다 작든 크든,
    // 한 (n,c)가 처리될 때 배당된 스레드블록들 안의 스레드들의 개수가 Ho*Wo의 개수보다 
    // 부족하면 그들 중 몇몇 스레드들은 2개이상의 윈도우에서 최댓값을 샘플링 해야하는 요구사항을 만족해야하며
    // 이는 xy평면에서 일직선으로 평탄화된 스레드들이 그들의 개수만큼 오프셋으로 이동하면서 반복해서 풀링을 수행 
    // 스레드블록들과 스레드들 모두에 대해서 경우가 나뉘는 이러한 경우에 if-else문을 이용해서 로직을 구현하면
    // 워프 단위 분기 발생 가능성이 높으므로 max함수를 이용해서 임의의 경우에 한 블록이 처리할 (n,c)개수를 구하고
    // 내부적으로는 항상 같은 로직 흐름을 타도록 구현해서 워프 단위 분기 발생 가능성을 줄여야 함
  
    const int totalNC = N * C;
    const int threadsPerBlk_MovOffset = blockDim.x * blockDim.y;  
    const int planeXY_flatten_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_nc_PerBlk = max(1, (totalNC + gridDim.z - 1) / gridDim.z);

    // 스레드블록에 의해 처리되는 첫번째(n,c)조합과 마지막(n,c)조합의 위치
    const int firstNC_ByBlk = blockIdx.z * num_nc_PerBlk;
    const int lastNC_ByBlk  = min(firstNC + num_nc_PerBlk, totalNC);

    // 각 스레드블록에 대해서 처리하는 모든 (n,c)조합을 순회
    for (int nc_idx = firstNC_ByBlk; nc_idx < lastNC_ByBlk; ++nc_idx) {
        int n = nc_idx / C; // 배치안의 특정 이미지가 안에서 몇번째인지 식별
        int c = nc_idx % C; // 배치안의 한 이미지의 몇번째 채널인지를 복원

        for (int loader = planeXY_flatten_idx; loader < totalPoolingTime; loader += threadsPerBlk_MovOffset) {

            
            int ho  = loader / Wo;
            int wo  = loader % Wo;
            if (ho >= Ho || wo >= Wo) continue;

            int h0 = ho * stride - pad; 
            int w0 = wo * stride - pad;
            float maxv = -FLT_MAX;

            for (int ph = 0; ph < pooling_window_Height; ++ph) {
                int h = h0 + ph;
                if (h < 0 || h >= H) continue;
              
                for (int pw = 0; pw < pooling_window_Width; ++pw) {
                    int w = w0 + pw;
                    if (w < 0 || w >= W) continue;
                  
                    float v = input[((n * C + c) * H + h) * W + w];
                    maxv = v > maxv ? v : maxv;
                }
            }

            output[((n * C + c) * Ho + ho) * Wo + wo] = maxv;
        }
    }
}
