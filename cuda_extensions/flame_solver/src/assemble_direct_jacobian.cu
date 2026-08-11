#include "utils.h"
#include "flame_tracking.h"


namespace AssembleDirectJacobian {

template<int N_VERTS>
__global__ void AssembleDirectVertexXJacobianCuda(
    const bool calc_jacobian,
    const int batch_size,
    const int num_unknows,
    const int num_items,
    const int num_residuals,
    const int64_t* __restrict__ vertex_indices,
    const float vertex_weight,
    const glm::vec3* __restrict__ batched_vertices,
    const glm::vec3* __restrict__ gt_vertices,
    const glm::vec3* __restrict__ batched_v_jacobian,
    float* __restrict__ residual,
    float* __restrict__ jacobian
) {
    const int batch_idx = blockIdx.x;
    const int item_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || item_idx >= num_items) return;

    int v_idx = item_idx;
    if (vertex_indices != nullptr) { v_idx = vertex_indices[item_idx]; }

    // size_t offsets so batch_idx * num_residuals * num_unknows can scale past 2^31.
    glm::vec3 vertex = batched_vertices[(size_t)batch_idx * N_VERTS + v_idx];
    glm::vec3 gt_vertex = gt_vertices[(size_t)batch_idx * N_VERTS + v_idx];
    glm::vec3 delta_vertex = vertex - gt_vertex;

    const int data_row = item_idx * 3;
    const size_t out_offset = (size_t)batch_idx * num_residuals + data_row;
    residual[out_offset + 0] = delta_vertex.x * vertex_weight;
    residual[out_offset + 1] = delta_vertex.y * vertex_weight;
    residual[out_offset + 2] = delta_vertex.z * vertex_weight;

    if (calc_jacobian) {
        const size_t batch_v_jac_base = (size_t)batch_idx * N_VERTS * num_unknows;
        const size_t batch_jac_base = (size_t)batch_idx * num_unknows * num_residuals;
        for (int i = 0; i < num_unknows; i++) {
            glm::vec3 v_jacobian = batched_v_jacobian[batch_v_jac_base + (size_t)v_idx * num_unknows + i];
            jacobian[batch_jac_base + (size_t)(data_row + 0) * num_unknows + i] = vertex_weight * v_jacobian.x;
            jacobian[batch_jac_base + (size_t)(data_row + 1) * num_unknows + i] = vertex_weight * v_jacobian.y;
            jacobian[batch_jac_base + (size_t)(data_row + 2) * num_unknows + i] = vertex_weight * v_jacobian.z;
        }
    }
}


template<int N_VERTS>
__global__ void AssembleDirectVertexIdentityJacobianCuda(
    const bool calc_jacobian,
    const int batch_size,
    const int num_unknows,
    const int num_items,
    const int num_residuals_per_frame,
    const int64_t* __restrict__ vertex_indices,
    const float vertex_weight,
    const glm::vec3* __restrict__ batched_vertices,
    const glm::vec3* __restrict__ gt_vertices,
    const glm::vec3* __restrict__ batched_v_jacobian,
    float* __restrict__ residual,
    float* __restrict__ jacobian
) {
    const int batch_idx = blockIdx.x;
    const int item_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || item_idx >= num_items) return;

    int v_idx = item_idx;
    if (vertex_indices != nullptr) { v_idx = vertex_indices[item_idx]; }

    glm::vec3 vertex = batched_vertices[(size_t)batch_idx * N_VERTS + v_idx];
    glm::vec3 gt_vertex = gt_vertices[(size_t)batch_idx * N_VERTS + v_idx];
    glm::vec3 delta_vertex = vertex - gt_vertex;

    // data_row spans the full flat residual block (batch×R_per_frame), so this
    // can grow into the millions — keep it size_t and any downstream
    // multiplication by num_unknows stays in 64-bit space.
    const size_t data_row = (size_t)batch_idx * num_residuals_per_frame + item_idx * 3;
    residual[data_row + 0] = delta_vertex.x * vertex_weight;
    residual[data_row + 1] = delta_vertex.y * vertex_weight;
    residual[data_row + 2] = delta_vertex.z * vertex_weight;

    if (calc_jacobian) {
        const size_t batch_v_jac_base = (size_t)batch_idx * N_VERTS * num_unknows;
        for (int i = 0; i < num_unknows; i++) {
            glm::vec3 v_jacobian = batched_v_jacobian[batch_v_jac_base + (size_t)v_idx * num_unknows + i];
            jacobian[(data_row + 0) * num_unknows + i] = vertex_weight * v_jacobian.x;
            jacobian[(data_row + 1) * num_unknows + i] = vertex_weight * v_jacobian.y;
            jacobian[(data_row + 2) * num_unknows + i] = vertex_weight * v_jacobian.z;
        }
    }
}

}


template<int N_VERTS, int N_EXPR>
static void launch_assemble_x_direct_jacobian_impl(
    const int batch_size,
    const int num_items,
    const int64_t* vertex_indices,
    const float vertex_weight,
    const float* batched_x,
    const float exp_reg_weight,
    const float pose_reg_weight,
    const glm::vec3* batched_vertices,
    const glm::vec3* gt_vertices,
    const glm::vec3* batched_v_jacobian,
    float* residual,
    float* jacobian
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);
    const int reg_residuals = N_EXPR + (NUM_FLAME_JOINTS - 1) * 3;
    const int num_residuals = num_items * 3 + reg_residuals;

    dim3 block_a(256, 1, 1);
    dim3 grid_a(batch_size, 1, (num_items + block_a.x - 1) / block_a.x);
    AssembleDirectJacobian::AssembleDirectVertexXJacobianCuda<N_VERTS><<<grid_a, block_a>>>(
        true,
        batch_size,
        DIM_X,
        num_items,
        num_residuals,
        vertex_indices,
        vertex_weight,
        batched_vertices,
        gt_vertices,
        batched_v_jacobian,
        residual,
        jacobian
    );

    dim3 block_b(256, 1, 1);
    dim3 grid_b(batch_size, 1, (reg_residuals + block_b.x - 1) / block_b.x);
    AssembleJacobian::AssembleExpPoseRegJacobianCuda<N_EXPR><<<grid_b, block_b>>>(
        true,
        batch_size,
        num_items * 3,
        num_residuals,
        batched_x,
        exp_reg_weight,
        pose_reg_weight,
        residual,
        jacobian
    );
}

template<int N_VERTS>
static void dispatch_assemble_x_direct_jacobian_by_expr(
    const int num_expressions,
    const int batch_size,
    const int num_items,
    const int64_t* vertex_indices,
    const float vertex_weight,
    const float* batched_x,
    const float exp_reg_weight,
    const float pose_reg_weight,
    const glm::vec3* batched_vertices,
    const glm::vec3* gt_vertices,
    const glm::vec3* batched_v_jacobian,
    float* residual,
    float* jacobian
) {
    switch (num_expressions) {
#define DISPATCH_CASE(N) \
        case N: \
            launch_assemble_x_direct_jacobian_impl<N_VERTS, N>( \
                batch_size, num_items, vertex_indices, vertex_weight, \
                batched_x, exp_reg_weight, pose_reg_weight, \
                batched_vertices, gt_vertices, batched_v_jacobian, \
                residual, jacobian \
            ); \
            break;
        FLAME_FOREACH_NEXPR(DISPATCH_CASE)
#undef DISPATCH_CASE
        default:
            throw std::runtime_error("assemble_x_direct_jacobian: unsupported num_expressions");
    }
}


void FlameTracking::assemble_x_direct_jacobian(
    const int num_vertices,
    const int num_expressions,
    const int batch_size,
    const int num_items,
    const int64_t* vertex_indices,
    const float vertex_weight,
    const float* batched_x,
    const float exp_reg_weight,
    const float pose_reg_weight,
    const glm::vec3* batched_vertices,
    const glm::vec3* gt_vertices,
    const glm::vec3* batched_v_jacobian,
    float* residual,
    float* jacobian
) {
#define DISPATCH_VERTS_CASE(N_VERTS) \
        case N_VERTS: \
            dispatch_assemble_x_direct_jacobian_by_expr<N_VERTS>( \
                num_expressions, batch_size, num_items, vertex_indices, vertex_weight, \
                batched_x, exp_reg_weight, pose_reg_weight, \
                batched_vertices, gt_vertices, batched_v_jacobian, \
                residual, jacobian \
            ); \
            break;
    switch (num_vertices) {
        FLAME_FOREACH_NVERTS(DISPATCH_VERTS_CASE)
        default:
            throw std::runtime_error("assemble_x_direct_jacobian: unsupported num_vertices");
    }
#undef DISPATCH_VERTS_CASE
}


void FlameTracking::assemble_identity_direct_jacobian(
    const int num_vertices,
    const int batch_size,
    const int num_items,
    const int64_t* vertex_indices,
    const float vertex_weight,
    const float* identity,
    const float identity_reg_weight,
    const glm::vec3* batched_vertices,
    const glm::vec3* gt_vertices,
    const glm::vec3* batched_v_jacobian,
    float* residual,
    float* jacobian
) {
    const int num_residuals_per_frame = num_items * 3;

    dim3 block(256, 1, 1);
    dim3 grid(batch_size, 1, (num_items + block.x - 1) / block.x);
    switch (num_vertices) {
#define DISPATCH_VERTS_CASE(N_VERTS) \
        case N_VERTS: \
            AssembleDirectJacobian::AssembleDirectVertexIdentityJacobianCuda<N_VERTS><<<grid, block>>>( \
                true, batch_size, NUM_FLAME_IDENTITY_BASIS, num_items, num_residuals_per_frame, \
                vertex_indices, vertex_weight, batched_vertices, gt_vertices, batched_v_jacobian, residual, jacobian \
            ); \
            break;
        FLAME_FOREACH_NVERTS(DISPATCH_VERTS_CASE)
#undef DISPATCH_VERTS_CASE
        default:
            throw std::runtime_error("assemble_identity_direct_jacobian: unsupported num_vertices");
    }

    dim3 reg_block(256, 1, 1);
    dim3 reg_grid(1, 1, (NUM_FLAME_IDENTITY_BASIS + reg_block.x - 1) / reg_block.x);
    AssembleJacobian::AssembleIdenRegJacobianCuda<<<reg_grid, reg_block>>>(
        true,
        num_residuals_per_frame * batch_size,
        identity,
        identity_reg_weight,
        residual,
        jacobian
    );
}
