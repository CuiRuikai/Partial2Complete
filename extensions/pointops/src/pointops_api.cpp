#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knnquery/knnquery_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knnquery_cuda", &knnquery_cuda, "knnquery_cuda");
    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");
}
