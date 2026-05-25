/*
 * GQA-optimized decode attention kernels.
 *
 * V2 (baseline): Grid = (num_q_heads, batch_size)
 *   Each block handles one query head. Loads K,V for the corresponding
 *   KV head from global memory — redundant loads when group_size > 1.
 *
 * V3 (optimized): Grid = (num_kv_heads, batch_size)
 *   Each block handles one KV head. Loads K,V into shared memory ONCE,
 *   then computes attention for ALL group_size query heads that share
 *   this KV head. Eliminates redundant global memory reads.
 *
 * Both kernels use online softmax for numerical stability.
 *
 * Input layout:
 *   Q:       [batch, num_q_heads, head_dim]          (single decode token)
 *   K_cache: [batch, context_len, num_kv_heads, head_dim]
 *   V_cache: [batch, context_len, num_kv_heads, head_dim]
 *   Output:  [batch, num_q_heads, head_dim]
 *
 * Template parameter HEAD_DIM = 64 (TinyLlama) or 128 (Llama-3, GPT-OSS).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

// Number of warps = HEAD_DIM / 32
constexpr int NUM_WARPS_64  = 2;
constexpr int NUM_WARPS_128 = 4;

// ─────────────────────────────────────────────────────────────────────
// Dot-product reduction: warp shuffle + cross-warp shared memory
// ─────────────────────────────────────────────────────────────────────

// Reduce a partial product across all HEAD_DIM threads to a single scalar.
// Returns the reduced value (valid in all threads via shared memory broadcast).
template <int HEAD_DIM>
__device__ __forceinline__ float reduce_dot(float partial, int tid, float* smem_reduce) {
    // Step 1: In-warp reduction (shuffle only works within a 32-thread warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }

    // Step 2: Cross-warp reduction via shared memory
    constexpr int NUM_WARPS = HEAD_DIM / 32;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        smem_reduce[warp_id] = partial;
    }
    __syncthreads();

    // Thread 0 sums across warps
    if (tid == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            sum += smem_reduce[w];
        }
        smem_reduce[0] = sum;
    }
    __syncthreads();

    return smem_reduce[0];
}


// ─────────────────────────────────────────────────────────────────────
// V2: Per-query-head decode attention (baseline)
// ─────────────────────────────────────────────────────────────────────

template <int HEAD_DIM>
__global__ void gqa_decode_v2_fp32_kernel(
    const float* __restrict__ Q,         // [batch, num_q_heads, HEAD_DIM]
    const float* __restrict__ K_cache,   // [batch, ctx_len, num_kv_heads, HEAD_DIM]
    const float* __restrict__ V_cache,   // [batch, ctx_len, num_kv_heads, HEAD_DIM]
    float* __restrict__ output,          // [batch, num_q_heads, HEAD_DIM]
    int num_q_heads,
    int num_kv_heads,
    int ctx_len,
    float scale
) {
    const int q_head = blockIdx.x;
    const int batch  = blockIdx.y;
    const int tid    = threadIdx.x;

    if (tid >= HEAD_DIM) return;

    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    float q_val = Q[batch * num_q_heads * HEAD_DIM + q_head * HEAD_DIM + tid];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc = 0.0f;

    __shared__ float smem_reduce[HEAD_DIM / 32];

    const float* k_base = K_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                          + kv_head * HEAD_DIM;
    const float* v_base = V_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                          + kv_head * HEAD_DIM;

    for (int t = 0; t < ctx_len; t++) {
        float k_val = k_base[t * num_kv_heads * HEAD_DIM + tid];
        float qk_partial = q_val * k_val;

        float qk = reduce_dot<HEAD_DIM>(qk_partial, tid, smem_reduce) * scale;

        // Online softmax update
        float new_max = fmaxf(running_max, qk);
        float correction = expf(running_max - new_max);
        float p = expf(qk - new_max);
        running_sum = running_sum * correction + p;
        acc = acc * correction + p * v_base[t * num_kv_heads * HEAD_DIM + tid];
        running_max = new_max;
    }

    if (running_sum > 0.0f) {
        acc /= running_sum;
    }

    output[batch * num_q_heads * HEAD_DIM + q_head * HEAD_DIM + tid] = acc;
}


template <int HEAD_DIM>
__global__ void gqa_decode_v2_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int ctx_len,
    float scale
) {
    const int q_head = blockIdx.x;
    const int batch  = blockIdx.y;
    const int tid    = threadIdx.x;

    if (tid >= HEAD_DIM) return;

    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    float q_val = __bfloat162float(Q[batch * num_q_heads * HEAD_DIM + q_head * HEAD_DIM + tid]);

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc = 0.0f;

    __shared__ float smem_reduce[HEAD_DIM / 32];

    const __nv_bfloat16* k_base = K_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                                  + kv_head * HEAD_DIM;
    const __nv_bfloat16* v_base = V_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                                  + kv_head * HEAD_DIM;

    for (int t = 0; t < ctx_len; t++) {
        float k_val = __bfloat162float(k_base[t * num_kv_heads * HEAD_DIM + tid]);
        float qk_partial = q_val * k_val;

        float qk = reduce_dot<HEAD_DIM>(qk_partial, tid, smem_reduce) * scale;

        float new_max = fmaxf(running_max, qk);
        float correction = expf(running_max - new_max);
        float p = expf(qk - new_max);
        running_sum = running_sum * correction + p;
        acc = acc * correction + p * __bfloat162float(v_base[t * num_kv_heads * HEAD_DIM + tid]);
        running_max = new_max;
    }

    if (running_sum > 0.0f) {
        acc /= running_sum;
    }

    output[batch * num_q_heads * HEAD_DIM + q_head * HEAD_DIM + tid] = __float2bfloat16(acc);
}


// ─────────────────────────────────────────────────────────────────────
// V3: Per-KV-head decode attention (shared K,V via shared memory)
// ─────────────────────────────────────────────────────────────────────

#define TOKENS_PER_ITER 32

template <int HEAD_DIM, int MAX_GROUP_SIZE>
__global__ void gqa_decode_v3_fp32_kernel(
    const float* __restrict__ Q,         // [batch, num_q_heads, HEAD_DIM]
    const float* __restrict__ K_cache,   // [batch, ctx_len, num_kv_heads, HEAD_DIM]
    const float* __restrict__ V_cache,   // [batch, ctx_len, num_kv_heads, HEAD_DIM]
    float* __restrict__ output,          // [batch, num_q_heads, HEAD_DIM]
    int num_q_heads,
    int num_kv_heads,
    int ctx_len,
    int group_size,
    float scale
) {
    const int kv_head = blockIdx.x;
    const int batch   = blockIdx.y;
    const int tid     = threadIdx.x;

    if (tid >= HEAD_DIM) return;

    const int first_q_head = kv_head * group_size;

    // Load Q for all query heads in this group into registers
    float q_reg[MAX_GROUP_SIZE];
    for (int g = 0; g < group_size; g++) {
        q_reg[g] = Q[batch * num_q_heads * HEAD_DIM + (first_q_head + g) * HEAD_DIM + tid];
    }

    // Per-query-head accumulators
    float running_max[MAX_GROUP_SIZE];
    float running_sum[MAX_GROUP_SIZE];
    float acc[MAX_GROUP_SIZE];
    for (int g = 0; g < group_size; g++) {
        running_max[g] = -FLT_MAX;
        running_sum[g] = 0.0f;
        acc[g] = 0.0f;
    }

    // Shared memory for K,V tiles and dot product reduction
    __shared__ float k_tile[TOKENS_PER_ITER][HEAD_DIM];
    __shared__ float v_tile[TOKENS_PER_ITER][HEAD_DIM];
    __shared__ float smem_reduce[HEAD_DIM / 32];

    const float* k_base = K_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                          + kv_head * HEAD_DIM;
    const float* v_base = V_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                          + kv_head * HEAD_DIM;

    for (int t_start = 0; t_start < ctx_len; t_start += TOKENS_PER_ITER) {
        int t_end = min(t_start + TOKENS_PER_ITER, ctx_len);
        int tile_len = t_end - t_start;

        // Cooperatively load K,V tiles into shared memory
        for (int t_off = 0; t_off < tile_len; t_off++) {
            k_tile[t_off][tid] = k_base[(t_start + t_off) * num_kv_heads * HEAD_DIM + tid];
            v_tile[t_off][tid] = v_base[(t_start + t_off) * num_kv_heads * HEAD_DIM + tid];
        }
        __syncthreads();

        // Process each token in the tile
        for (int t_off = 0; t_off < tile_len; t_off++) {
            float v_val = v_tile[t_off][tid];
            // For each query head in the group
            for (int g = 0; g < group_size; g++) {
                float qk_partial = q_reg[g] * k_tile[t_off][tid];
                float qk = reduce_dot<HEAD_DIM>(qk_partial, tid, smem_reduce) * scale;

                // Online softmax update
                float new_max = fmaxf(running_max[g], qk);
                float correction = expf(running_max[g] - new_max);
                float p = expf(qk - new_max);
                running_sum[g] = running_sum[g] * correction + p;
                acc[g] = acc[g] * correction + p * v_val;
                running_max[g] = new_max;
            }
        }
        __syncthreads();
    }

    // Write output for all query heads
    for (int g = 0; g < group_size; g++) {
        float final_val = (running_sum[g] > 0.0f) ? (acc[g] / running_sum[g]) : 0.0f;
        output[batch * num_q_heads * HEAD_DIM + (first_q_head + g) * HEAD_DIM + tid] = final_val;
    }
}


template <int HEAD_DIM, int MAX_GROUP_SIZE>
__global__ void gqa_decode_v3_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int ctx_len,
    int group_size,
    float scale
) {
    const int kv_head = blockIdx.x;
    const int batch   = blockIdx.y;
    const int tid     = threadIdx.x;

    if (tid >= HEAD_DIM) return;

    const int first_q_head = kv_head * group_size;

    float q_reg[MAX_GROUP_SIZE];
    for (int g = 0; g < group_size; g++) {
        q_reg[g] = __bfloat162float(Q[batch * num_q_heads * HEAD_DIM + (first_q_head + g) * HEAD_DIM + tid]);
    }

    float running_max[MAX_GROUP_SIZE];
    float running_sum[MAX_GROUP_SIZE];
    float acc[MAX_GROUP_SIZE];
    for (int g = 0; g < group_size; g++) {
        running_max[g] = -FLT_MAX;
        running_sum[g] = 0.0f;
        acc[g] = 0.0f;
    }

    __shared__ float k_tile[TOKENS_PER_ITER][HEAD_DIM];
    __shared__ float v_tile[TOKENS_PER_ITER][HEAD_DIM];
    __shared__ float smem_reduce[HEAD_DIM / 32];

    const __nv_bfloat16* k_base = K_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                                  + kv_head * HEAD_DIM;
    const __nv_bfloat16* v_base = V_cache + batch * ctx_len * num_kv_heads * HEAD_DIM
                                  + kv_head * HEAD_DIM;

    for (int t_start = 0; t_start < ctx_len; t_start += TOKENS_PER_ITER) {
        int t_end = min(t_start + TOKENS_PER_ITER, ctx_len);
        int tile_len = t_end - t_start;

        for (int t_off = 0; t_off < tile_len; t_off++) {
            k_tile[t_off][tid] = __bfloat162float(k_base[(t_start + t_off) * num_kv_heads * HEAD_DIM + tid]);
            v_tile[t_off][tid] = __bfloat162float(v_base[(t_start + t_off) * num_kv_heads * HEAD_DIM + tid]);
        }
        __syncthreads();

        for (int t_off = 0; t_off < tile_len; t_off++) {
            float v_val = v_tile[t_off][tid];
            for (int g = 0; g < group_size; g++) {
                float qk_partial = q_reg[g] * k_tile[t_off][tid];
                float qk = reduce_dot<HEAD_DIM>(qk_partial, tid, smem_reduce) * scale;

                float new_max = fmaxf(running_max[g], qk);
                float correction = expf(running_max[g] - new_max);
                float p = expf(qk - new_max);
                running_sum[g] = running_sum[g] * correction + p;
                acc[g] = acc[g] * correction + p * v_val;
                running_max[g] = new_max;
            }
        }
        __syncthreads();
    }

    for (int g = 0; g < group_size; g++) {
        float final_val = (running_sum[g] > 0.0f) ? (acc[g] / running_sum[g]) : 0.0f;
        output[batch * num_q_heads * HEAD_DIM + (first_q_head + g) * HEAD_DIM + tid] = __float2bfloat16(final_val);
    }
}


// ─────────────────────────────────────────────────────────────────────
// Host dispatch functions
// ─────────────────────────────────────────────────────────────────────

void gqa_decode_v2_cuda(
    torch::Tensor Q,        // [batch, num_q_heads, head_dim]
    torch::Tensor K_cache,  // [batch, ctx_len, num_kv_heads, head_dim]
    torch::Tensor V_cache,  // [batch, ctx_len, num_kv_heads, head_dim]
    torch::Tensor output,   // [batch, num_q_heads, head_dim]
    float scale
) {
    const int batch_size = Q.size(0);
    const int num_q_heads = Q.size(1);
    const int head_dim = Q.size(2);
    const int ctx_len = K_cache.size(1);
    const int num_kv_heads = K_cache.size(2);

    dim3 grid(num_q_heads, batch_size);
    dim3 block(head_dim);

    if (Q.scalar_type() == torch::kFloat32) {
        if (head_dim == 64) {
            gqa_decode_v2_fp32_kernel<64><<<grid, block>>>(
                Q.data_ptr<float>(), K_cache.data_ptr<float>(),
                V_cache.data_ptr<float>(), output.data_ptr<float>(),
                num_q_heads, num_kv_heads, ctx_len, scale);
        } else {
            gqa_decode_v2_fp32_kernel<128><<<grid, block>>>(
                Q.data_ptr<float>(), K_cache.data_ptr<float>(),
                V_cache.data_ptr<float>(), output.data_ptr<float>(),
                num_q_heads, num_kv_heads, ctx_len, scale);
        }
    } else {  // BF16
        if (head_dim == 64) {
            gqa_decode_v2_bf16_kernel<64><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                num_q_heads, num_kv_heads, ctx_len, scale);
        } else {
            gqa_decode_v2_bf16_kernel<128><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                num_q_heads, num_kv_heads, ctx_len, scale);
        }
    }
}


void gqa_decode_v3_cuda(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor output,
    int group_size,
    float scale
) {
    const int batch_size = Q.size(0);
    const int num_q_heads = Q.size(1);
    const int head_dim = Q.size(2);
    const int ctx_len = K_cache.size(1);
    const int num_kv_heads = K_cache.size(2);

    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);

    if (Q.scalar_type() == torch::kFloat32) {
        if (head_dim == 64) {
            if (group_size <= 4) {
                gqa_decode_v3_fp32_kernel<64, 4><<<grid, block>>>(
                    Q.data_ptr<float>(), K_cache.data_ptr<float>(),
                    V_cache.data_ptr<float>(), output.data_ptr<float>(),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            } else {
                gqa_decode_v3_fp32_kernel<64, 8><<<grid, block>>>(
                    Q.data_ptr<float>(), K_cache.data_ptr<float>(),
                    V_cache.data_ptr<float>(), output.data_ptr<float>(),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            }
        } else {
            if (group_size <= 4) {
                gqa_decode_v3_fp32_kernel<128, 4><<<grid, block>>>(
                    Q.data_ptr<float>(), K_cache.data_ptr<float>(),
                    V_cache.data_ptr<float>(), output.data_ptr<float>(),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            } else {
                gqa_decode_v3_fp32_kernel<128, 8><<<grid, block>>>(
                    Q.data_ptr<float>(), K_cache.data_ptr<float>(),
                    V_cache.data_ptr<float>(), output.data_ptr<float>(),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            }
        }
    } else {  // BF16
        if (head_dim == 64) {
            if (group_size <= 4) {
                gqa_decode_v3_bf16_kernel<64, 4><<<grid, block>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            } else {
                gqa_decode_v3_bf16_kernel<64, 8><<<grid, block>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            }
        } else {
            if (group_size <= 4) {
                gqa_decode_v3_bf16_kernel<128, 4><<<grid, block>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            } else {
                gqa_decode_v3_bf16_kernel<128, 8><<<grid, block>>>(
                    reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(K_cache.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(V_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_q_heads, num_kv_heads, ctx_len, group_size, scale);
            }
        }
    }
}
