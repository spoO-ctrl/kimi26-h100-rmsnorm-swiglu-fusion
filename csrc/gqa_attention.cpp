#include <torch/extension.h>

// V2: Per-query-head decode attention (baseline)
void gqa_decode_v2_cuda(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor output,
    float scale
);

// V3: Per-KV-head decode attention (shared K,V)
void gqa_decode_v3_cuda(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor output,
    int group_size,
    float scale
);

// Python wrappers
void gqa_decode_v2(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor output,
    float scale
) {
    gqa_decode_v2_cuda(Q, K_cache, V_cache, output, scale);
}

void gqa_decode_v3(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor output,
    int group_size,
    float scale
) {
    gqa_decode_v3_cuda(Q, K_cache, V_cache, output, group_size, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_decode_v2", &gqa_decode_v2,
          "GQA decode attention V2 (per query head)",
          py::arg("Q"),
          py::arg("K_cache"),
          py::arg("V_cache"),
          py::arg("output"),
          py::arg("scale"));

    m.def("gqa_decode_v3", &gqa_decode_v3,
          "GQA decode attention V3 (per KV head, shared K,V)",
          py::arg("Q"),
          py::arg("K_cache"),
          py::arg("V_cache"),
          py::arg("output"),
          py::arg("group_size"),
          py::arg("scale"));
}
