#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <float.h>

// =====================
// Tiled 2D Convolution 커널 (공유메모리)
// 입력:  input [N, C, H, W] 배치안입력피쳐맵수,입력피쳐맵채널수,입력피쳐맵높이,입력피쳐맵폭
// weight: weight [Kout, C, Kh, Kw] 출력피쳐맵채널수,입력피쳐맵채널수,마스크높이,마스크폭
// bias:   bias [Kout] 편향벡터는출력피쳐맵의채널수만큼의크기를가짐
// 출력:  output [N, Kout, Ho, Wo] 입력과동일한배치크기,출력피쳐맵채널수,컨볼루션결과높이,컨볼루션결과폭
// 타일 크기: TILE_DIM x TILE_DIM
// =====================
#define TILE_DIM 16  // 출력 타일 폭

__global__ void tiled_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight, // 커널은임의의스레드마다컨볼루션계산시매번사용해야함
    const float* __restrict__ bias,   // 각각의스레드가매번커널을전역메모리에서불러오면레이턴시증가
    float* __restrict__ output,       // 적은저장리소스를가진공유메모리에계속저장되어있으면리소스낭비
    int N, int C, int H, int W,       // 따라서커널(마스크)의원소들은상수메모리에저장하는것으로함
    int Kout, int Kh, int Kw,
    int pad, int stride)
{
    // 아웃풋피쳐맵이그것의행과열방향으로몇개의픽셀을가지고있는지계산
    // 이시점에서결국하나의스레드가아웃풋피쳐맵에속한하나의픽셀을담당하게됨을염두해둠
    // 패딩은상하좌우시작끝에동일한크기만큼적용한다고가정
    int Ho = ((pad + H + pad) - Kh) / stride + 1; // 행, 높이, y축, 상하동일량패딩
    int Wo = ((pad + W + pad) - Kw) / stride + 1; // 열, 너비, x축, 좌우동일량패딩
    
    // 타일드컨볼루션을구현하려면전역메모리(DRAM/HBM)에저장되어있는입력피쳐맵을
    // 공유메모리(SRAMonGPUchip)으로가져와야하고이때사용해야할공유메모리의크기를지정해야함
    // 출력타일의한픽셀을계산하는각각의스레드는커널만큼의크기를가진입력피쳐맵의일부분을가져와야함
    // 이때중요한건한스레드와그인접스레드가가져와야할픽셀들은스트라이드가커널의폭에비해서작은이상
    // 겹쳐질수밖에없고,겹쳐지는구체적인크기가바로커널의폭에서스트라이드의크기를뺀것임
    // 그렇다면,겹쳐지지않는크기는스트라이드의크기그자체이고
    // 첫번째스레드가컨볼루션해야할픽셀들을커널의폭만큼공유메모리로가져온후
    // 추가적으로스트라이드만큼픽셀들을공유메모리로가져오는작업을(블록안에속한스레드의개수-1)만큼반복해야
    // 나머지스레드들이컨볼루션해야할픽셀들을모두가져올수가있게된다.
    extern __shared__ float tile[];
    int Sh = Kh + stride * (blockDim.y - 1); // 행, 높이, y축
    int Sw = Kw + stride * (blockDim.x - 1); // 열, 너비, x축
 
    // 위와같이공유메모리의영역을할당한이후에는
    // 스레드블록들과그것들각각에속한스레드들의전역좌표를알아내야지
    // 실질적으로전역메모리인(DRAM/HBM)에저장되어있는인풋피쳐맵의픽셀들을공유메모리로불러올수있음
    // 쿠다커널의각스레드들은서로서로단하나의베이스포인터float*input을동일하게사용함에도불구하고
    // 자신들의블록인덱스와스레드인덱스로계산된서로다른오프셋을더함으로써
    // 스레드및그인접스레드들이읽고쓸전역메모리(DRAM/HBM)셀들을정확하게지정함
    // 스레드블록의좌표에각블록이가진스레드의개수의개수를곱하면각블록이가진첫번째스레드의글로벌좌표임
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;  // [0, blockDim.x)
    int ty = threadIdx.y;  // [0, blockDim.y)
    int n  = blockIdx.z; //배치인덱스

    // 구체적으로컨볼루션의절차가어떻게이루어지는지생각해보면
    // 4차원텐서인배치에서3차원텐서인입력이미지를꺼내면,입력이미지당
    // 서로가진성분은다르지만동일한폭과높이를가진마스크들이입력채널의개수만큼존재하고
    // 그마스크들이상응하는각각의입력피쳐맵의채널에컨볼루션을출력채널의갯수만큼반복하고
    // 각출력채널의산출은각반복횟수당모든입력채널들에대한컨볼루션결과를합치는것임
    // 따라서공유메모리로로딩할때입력채널수에대해서뿐만아니라출력채널수에대해서도순회해야함
    // 신호처리에서,LSI필터의개념을생각해봤을때,특정한시점에서의출력탭y[n]을얻기위해서일지라도
    // 입력탭들이x[]여러개,정확하게는커널의폭만큼의시점들에서x[]들이필요함
    // 컨볼루션(사실은크로스코얼레이션)이적용될입력신호의인덱스는일반적으로형태가x[n+m]
    // 스트라이드stride와패딩pad를도입하면,x[n*stride+m-pad]로주어짐,여기서n은시작점
    // 병렬디지털신호처리에서시작점n은스레드블록들의첫번째스레드의전역인덱스이고
    // 마스크순회인덱스m은스레드들의자신들이속한블록안에서의인덱스임
    //
    for (int k = 0; k < Kout; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = ty; i < Sh; i += blockDim.y) {
                for (int j = tx; j < Sw; j += blockDim.x) {
                    int in_x = bx * stride + j - pad;
                    int in_y = by * stride + i - pad;
                    float v = 0.0f;
                    if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
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
