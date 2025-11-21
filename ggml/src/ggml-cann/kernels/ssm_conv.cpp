
#include "kernel_operator.h"

typedef float Float;
#define BUFFER_NUM 2

class SSMConv {
  public:


  private:
    AscendC::GlobalTensor<Float>                            global_y, global_x, global_w;
    AscendC::TPipe                                          pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>  x_to_compute;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>  w_to_compute;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> y_from_compute;
};

extern "C" __global__ __aicore__ void ggml_cann_ssm_conv_run_kernel_impl(GM_ADDR y,
                                                                         GM_ADDR x,
                                                                         GM_ADDR w,
                                                                         int64_t d_conv,
                                                                         int64_t d_inner,
                                                                         int64_t nt,
                                                                         int64_t ns) {
    AscendC::printf("Running with d_conv=%d, d_inner=%d, nt=%d, ns=%d\n", d_conv, d_inner, nt, ns);
}

void ggml_cann_ssm_conv_run_kernel(uint8_t * y,
                                   uint8_t * x,
                                   uint8_t * w,
                                   int64_t   block_cnt,
                                   int64_t   d_conv,
                                   int64_t   d_inner,
                                   int64_t   nt,
                                   int64_t   ns,
                                   void *    stream) {
    ggml_cann_ssm_conv_run_kernel_impl<<<block_cnt, nullptr, stream>>>(y, x, w, d_conv, d_inner, nt, ns);
}
