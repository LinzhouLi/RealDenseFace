#include "utils.h"
#include "flame_tracking.h"


namespace AssembleJacobian {

template<int N_VERTS, bool UseUncertainty, bool UseDepth>
__global__ void AssembleAlignmentDepthJacobianCuda(
    const bool calc_jacobian,
    const int batch_size,
    const int num_views,
    const int num_unknows,
    const int num_items,
    const int num_residuals,
    const int camera_view_stride,
    const int64_t* __restrict__ vertex_indices,
    const float alignment_weight,
    const float rel_depth_weight,

    const glm::mat4* __restrict__ view_mat,
    const glm::mat4* __restrict__ viewproj_mat,
    const glm::vec3* __restrict__ batched_vertices,
    const glm::vec3* __restrict__ batched_joints,
    const glm::vec2* __restrict__ gt_projected_vertices,
    const float* __restrict__ gt_vertex_rel_depths,
    const float* __restrict__ gt_projected_log_vars,
    const float* __restrict__ gt_vertex_rel_depth_log_vars,

    const glm::vec3* __restrict__ batched_v_jacobian,
    const glm::vec3* __restrict__ batched_j_jacobian,

    float* __restrict__ residual,
    float* __restrict__ jacobian
) {
    const int batch_idx = blockIdx.x;
    const int view_idx = blockIdx.y;
    const int item_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || view_idx >= num_views || item_idx >= num_items) return;

    constexpr int obs_dim = UseDepth ? 3 : 2;

    int v_idx = item_idx;
    if (vertex_indices != nullptr) { v_idx = vertex_indices[item_idx]; }

    const int cam_offset = batch_idx * camera_view_stride + view_idx;
    glm::mat4 camera_view_mat = glm::transpose(view_mat[cam_offset]);
    glm::mat4 camera_viewproj_mat = glm::transpose(viewproj_mat[cam_offset]);

    // All offsets below are computed as size_t to avoid int32 overflow when
    // batch_size * num_views * num_items * num_unknows can exceed 2^31.
    glm::vec3 vertex = batched_vertices[(size_t)batch_idx * N_VERTS + v_idx];
    const size_t obs_offset = ((size_t)batch_idx * num_views + view_idx) * N_VERTS + v_idx;
    glm::vec2 gt_projected_vertex = gt_projected_vertices[obs_offset];

    float eff_alignment_weight = alignment_weight;
    float eff_rel_depth_weight = rel_depth_weight;
    if (UseUncertainty) {
        const float projected_log_var = gt_projected_log_vars[obs_offset];
        eff_alignment_weight = alignment_weight * 0.018f * __expf(-0.5f * projected_log_var);
        if constexpr (UseDepth) {
            const float rel_depth_log_var = gt_vertex_rel_depth_log_vars[obs_offset];
            eff_rel_depth_weight = rel_depth_weight * 0.018f * __expf(-0.5f * rel_depth_log_var);
        }
    }

    glm::vec4 vertex_homo = glm::vec4(vertex.x, vertex.y, vertex.z, 1.0);
    glm::vec4 projected_vertex_homo = camera_viewproj_mat * vertex_homo;
    glm::vec2 projected_vertex = glm::vec2(projected_vertex_homo.x, projected_vertex_homo.y) / std::max(projected_vertex_homo.w, 1e-8f);
    projected_vertex = projected_vertex * 0.5f + 0.5f;
    glm::vec2 delta_projected_vertex = projected_vertex - gt_projected_vertex;

    const int data_row = view_idx * num_items * obs_dim + item_idx * obs_dim;
    const size_t out_offset = (size_t)batch_idx * num_residuals + data_row;
    residual[out_offset + 0] = delta_projected_vertex.x * eff_alignment_weight;
    residual[out_offset + 1] = delta_projected_vertex.y * eff_alignment_weight;

    glm::vec3 J_u;
    glm::vec3 J_v;
    if (calc_jacobian) {
        float v0 = projected_vertex_homo.x;
        float v1 = projected_vertex_homo.y;
        float v3 = projected_vertex_homo.w;

        glm::vec3 P0 = glm::vec3(camera_viewproj_mat[0][0], camera_viewproj_mat[1][0], camera_viewproj_mat[2][0]);
        glm::vec3 P1 = glm::vec3(camera_viewproj_mat[0][1], camera_viewproj_mat[1][1], camera_viewproj_mat[2][1]);
        glm::vec3 P3 = glm::vec3(camera_viewproj_mat[0][3], camera_viewproj_mat[1][3], camera_viewproj_mat[2][3]);

        float v3_2 = std::max(v3 * v3, 1e-8f);
        J_u = (v3 * P0 - v0 * P3) / v3_2;
        J_v = (v3 * P1 - v1 * P3) / v3_2;
    }

    // Per-batch base offsets in size_t — these dominate the int32 overflow risk
    // when batch_size * num_residuals * num_unknows can exceed 2^31. Hoisting
    // the bases out also lets the compiler skip recomputing them in the loop.
    const size_t batch_v_jac_base = (size_t)batch_idx * N_VERTS * num_unknows;
    const size_t batch_j_jac_base = (size_t)batch_idx * NUM_FLAME_JOINTS * num_unknows;
    const size_t batch_jac_base = (size_t)batch_idx * num_unknows * num_residuals;

    if constexpr (UseDepth) {
        glm::vec3 neck_joint = batched_joints[(size_t)batch_idx * NUM_FLAME_JOINTS + 1];
        float gt_rel_depth = gt_vertex_rel_depths[obs_offset];
        glm::vec3 rel_vector = glm::mat3(camera_view_mat) * (vertex - neck_joint);
        float rel_depth = rel_vector.z;
        float delta_rel_depth = rel_depth - gt_rel_depth;
        residual[out_offset + 2] = delta_rel_depth * eff_rel_depth_weight;

        if (calc_jacobian) {
            glm::vec3 row_view_z = glm::vec3(camera_view_mat[0][2], camera_view_mat[1][2], camera_view_mat[2][2]);
            for (int i = 0; i < num_unknows; i++) {
                glm::vec3 v_jacobian = batched_v_jacobian[batch_v_jac_base + (size_t)v_idx * num_unknows + i];
                glm::vec3 j_jacobian = batched_j_jacobian[batch_j_jac_base + (size_t)1 * num_unknows + i];

                float ju = 0.5f * eff_alignment_weight * glm::dot(v_jacobian, J_u);
                float jv = 0.5f * eff_alignment_weight * glm::dot(v_jacobian, J_v);
                float jd = eff_rel_depth_weight * glm::dot(row_view_z, (v_jacobian - j_jacobian));

                jacobian[batch_jac_base + (size_t)(data_row + 0) * num_unknows + i] = ju;
                jacobian[batch_jac_base + (size_t)(data_row + 1) * num_unknows + i] = jv;
                jacobian[batch_jac_base + (size_t)(data_row + 2) * num_unknows + i] = jd;
            }
        }
    } else if (calc_jacobian) {
        for (int i = 0; i < num_unknows; i++) {
            glm::vec3 v_jacobian = batched_v_jacobian[batch_v_jac_base + (size_t)v_idx * num_unknows + i];
            float ju = 0.5f * eff_alignment_weight * glm::dot(v_jacobian, J_u);
            float jv = 0.5f * eff_alignment_weight * glm::dot(v_jacobian, J_v);
            jacobian[batch_jac_base + (size_t)(data_row + 0) * num_unknows + i] = ju;
            jacobian[batch_jac_base + (size_t)(data_row + 1) * num_unknows + i] = jv;
        }
    }
}


template<int N_EXPR>
__global__ void AssembleExpPoseRegJacobianCuda(
    const bool calc_jacobian,
    const int batch_size,
    const int num_offsets,
    const int num_residuals,
    const float* __restrict__ unknows,
    const float exp_reg_weight,
    const float pose_reg_weight,

    float* __restrict__ residual,
    float* __restrict__ jacobian
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);

    const int batch_idx = blockIdx.x;
    const int r_idx = blockIdx.z * blockDim.x + threadIdx.x;
    const int num_reg_items = N_EXPR + (NUM_FLAME_JOINTS - 1) * 3;
    if (batch_idx >= batch_size || r_idx >= num_reg_items) return;

    int x_idx = r_idx;
    float reg_weight = exp_reg_weight;
    if (r_idx >= N_EXPR) {
        x_idx += 3;
        reg_weight = pose_reg_weight;
    }

    // size_t casts to keep batch_idx * num_residuals * DIM_X products from
    // overflowing int32 on multi-frame multi-view problems.
    residual[(size_t)batch_idx * num_residuals + num_offsets + r_idx] =
        unknows[(size_t)batch_idx * DIM_X + x_idx] * reg_weight;
    if (calc_jacobian) {
        jacobian[(size_t)batch_idx * DIM_X * num_residuals + (size_t)(num_offsets + r_idx) * DIM_X + x_idx] = reg_weight;
    }
}


__global__ void AssembleIdenRegJacobianCuda(
    const bool calc_jacobian,
    const int num_offsets,
    const float* __restrict__ unknows,
    const float iden_reg_weight,

    float* __restrict__ residual,
    float* __restrict__ jacobian
) {
    const int x_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (x_idx >= NUM_FLAME_IDENTITY_BASIS) return;

    // num_offsets here is K * num_views * num_items * obs_dim (millions for
    // multi-frame multi-view), and the jacobian write multiplies by 300, so
    // the offset can easily exceed 2^31 — cast to size_t.
    residual[(size_t)num_offsets + x_idx] = unknows[x_idx] * iden_reg_weight;
    if (calc_jacobian) {
        jacobian[((size_t)num_offsets + x_idx) * NUM_FLAME_IDENTITY_BASIS + x_idx] = iden_reg_weight;
    }
}

}  // namespace AssembleJacobian


// ---------- host-side templated launcher + dispatch ----------

template<int N_VERTS, int N_EXPR>
static void launch_assemble_x_jacobian_impl(
    const int batch_size,
    const int num_views,
    const int num_items,
    const int camera_view_stride,
    const int64_t* vertex_indices,
    const float alignment_weight,
    const float rel_depth_weight,

    const float* batched_x,
    const float exp_reg_weight,
    const float pose_reg_weight,

    const glm::mat4* view_mat,
    const glm::mat4* viewproj_mat,
    const glm::vec3* batched_vertices,
    const glm::vec3* batched_joints,
    const glm::vec2* gt_projected_vertices,
    const float* gt_vertex_rel_depths,
    const float* gt_projected_log_vars,
    const float* gt_vertex_rel_depth_log_vars,

    const glm::vec3* batched_v_jacobian,
    const glm::vec3* batched_j_jacobian,

    float* residual,
    float* jacobian
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);
    const int reg_residuals = N_EXPR + (NUM_FLAME_JOINTS - 1) * 3;
    const bool use_rel_depth = rel_depth_weight > 0.0f;
    const int obs_dim = use_rel_depth ? 3 : 2;
    const int num_residuals = num_views * num_items * obs_dim + reg_residuals;
    const bool use_uncertainty = (gt_projected_log_vars != nullptr && gt_vertex_rel_depth_log_vars != nullptr);

    dim3 block_a(256, 1, 1);
    dim3 grid_a(batch_size, num_views, (num_items + block_a.x - 1) / block_a.x);
    if (use_uncertainty) {
        if (use_rel_depth) {
            AssembleJacobian::AssembleAlignmentDepthJacobianCuda<N_VERTS, true, true><<<grid_a, block_a>>>(
                true,
                batch_size,
                num_views,
                DIM_X,
                num_items,
                num_residuals,
                camera_view_stride,
                vertex_indices,
                alignment_weight,
                rel_depth_weight,
                view_mat,
                viewproj_mat,
                batched_vertices,
                batched_joints,
                gt_projected_vertices,
                gt_vertex_rel_depths,
                gt_projected_log_vars,
                gt_vertex_rel_depth_log_vars,
                batched_v_jacobian,
                batched_j_jacobian,
                residual,
                jacobian
            );
        } else {
            AssembleJacobian::AssembleAlignmentDepthJacobianCuda<N_VERTS, true, false><<<grid_a, block_a>>>(
                true,
                batch_size,
                num_views,
                DIM_X,
                num_items,
                num_residuals,
                camera_view_stride,
                vertex_indices,
                alignment_weight,
                rel_depth_weight,
                view_mat,
                viewproj_mat,
                batched_vertices,
                batched_joints,
                gt_projected_vertices,
                gt_vertex_rel_depths,
                gt_projected_log_vars,
                gt_vertex_rel_depth_log_vars,
                batched_v_jacobian,
                batched_j_jacobian,
                residual,
                jacobian
            );
        }
    } else {
        if (use_rel_depth) {
            AssembleJacobian::AssembleAlignmentDepthJacobianCuda<N_VERTS, false, true><<<grid_a, block_a>>>(
                true,
                batch_size,
                num_views,
                DIM_X,
                num_items,
                num_residuals,
                camera_view_stride,
                vertex_indices,
                alignment_weight,
                rel_depth_weight,
                view_mat,
                viewproj_mat,
                batched_vertices,
                batched_joints,
                gt_projected_vertices,
                gt_vertex_rel_depths,
                nullptr,
                nullptr,
                batched_v_jacobian,
                batched_j_jacobian,
                residual,
                jacobian
            );
        } else {
            AssembleJacobian::AssembleAlignmentDepthJacobianCuda<N_VERTS, false, false><<<grid_a, block_a>>>(
                true,
                batch_size,
                num_views,
                DIM_X,
                num_items,
                num_residuals,
                camera_view_stride,
                vertex_indices,
                alignment_weight,
                rel_depth_weight,
                view_mat,
                viewproj_mat,
                batched_vertices,
                batched_joints,
                gt_projected_vertices,
                gt_vertex_rel_depths,
                nullptr,
                nullptr,
                batched_v_jacobian,
                batched_j_jacobian,
                residual,
                jacobian
            );
        }
    }

    dim3 block_b(256, 1, 1);
    dim3 grid_b(batch_size, 1, (reg_residuals + block_b.x - 1) / block_b.x);
    AssembleJacobian::AssembleExpPoseRegJacobianCuda<N_EXPR><<<grid_b, block_b>>>(
        true,
        batch_size,
        num_views * num_items * obs_dim,
        num_residuals,
        batched_x,
        exp_reg_weight,
        pose_reg_weight,
        residual,
        jacobian
    );
}

template<int N_VERTS>
static void dispatch_assemble_x_jacobian_by_expr(
    const int num_expressions,
    const int batch_size,
    const int num_views,
    const int num_items,
    const int camera_view_stride,
    const int64_t* vertex_indices,
    const float alignment_weight,
    const float rel_depth_weight,
    const float* batched_x,
    const float exp_reg_weight,
    const float pose_reg_weight,
    const glm::mat4* view_mat,
    const glm::mat4* viewproj_mat,
    const glm::vec3* batched_vertices,
    const glm::vec3* batched_joints,
    const glm::vec2* gt_projected_vertices,
    const float* gt_vertex_rel_depths,
    const float* gt_projected_log_vars,
    const float* gt_vertex_rel_depth_log_vars,
    const glm::vec3* batched_v_jacobian,
    const glm::vec3* batched_j_jacobian,
    float* residual,
    float* jacobian
) {
    switch (num_expressions) {
#define DISPATCH_CASE(N) \
        case N: \
            launch_assemble_x_jacobian_impl<N_VERTS, N>( \
                batch_size, num_views, num_items, camera_view_stride, vertex_indices, \
                alignment_weight, rel_depth_weight, \
                batched_x, exp_reg_weight, pose_reg_weight, \
                view_mat, viewproj_mat, batched_vertices, batched_joints, \
                gt_projected_vertices, gt_vertex_rel_depths, \
                gt_projected_log_vars, gt_vertex_rel_depth_log_vars, \
                batched_v_jacobian, batched_j_jacobian, \
                residual, jacobian \
            ); \
            break;
        FLAME_FOREACH_NEXPR(DISPATCH_CASE)
#undef DISPATCH_CASE
        default:
            throw std::runtime_error("assemble_x_jacobian: unsupported num_expressions");
    }
}


void FlameTracking::assemble_x_jacobian(
    const int num_vertices,
    const int num_expressions,
    const int batch_size,
    const int num_views,
    const int num_items,
    const int camera_view_stride,
    const int64_t* vertex_indices,
    const float alignment_weight,
    const float rel_depth_weight,

    const float* batched_x,
    const float exp_reg_weight,
    const float pose_reg_weight,

    const glm::mat4* view_mat,
    const glm::mat4* viewproj_mat,
    const glm::vec3* batched_vertices,
    const glm::vec3* batched_joints,
    const glm::vec2* gt_projected_vertices,
    const float* gt_vertex_rel_depths,
    const float* gt_projected_log_vars,
    const float* gt_vertex_rel_depth_log_vars,

    const glm::vec3* batched_v_jacobian,
    const glm::vec3* batched_j_jacobian,

    float* residual,
    float* jacobian
) {
#define DISPATCH_VERTS_CASE(N_VERTS) \
        case N_VERTS: \
            dispatch_assemble_x_jacobian_by_expr<N_VERTS>( \
                num_expressions, batch_size, num_views, num_items, camera_view_stride, vertex_indices, \
                alignment_weight, rel_depth_weight, batched_x, exp_reg_weight, pose_reg_weight, \
                view_mat, viewproj_mat, batched_vertices, batched_joints, \
                gt_projected_vertices, gt_vertex_rel_depths, gt_projected_log_vars, gt_vertex_rel_depth_log_vars, \
                batched_v_jacobian, batched_j_jacobian, residual, jacobian \
            ); \
            break;
    switch (num_vertices) {
        FLAME_FOREACH_NVERTS(DISPATCH_VERTS_CASE)
        default:
            throw std::runtime_error("assemble_x_jacobian: unsupported num_vertices");
    }
#undef DISPATCH_VERTS_CASE
}

template<int N_VERTS, bool UseUncertainty, bool UseDepth>
static void launch_assemble_identity_alignment_impl(
    const int batch_size,
    const int num_views,
    const int num_items,
    const int num_residuals_per_frame,
    const int camera_view_stride,
    const int64_t* vertex_indices,
    const float alignment_weight,
    const float rel_depth_weight,
    const glm::mat4* view_mat,
    const glm::mat4* viewproj_mat,
    const glm::vec3* batched_vertices,
    const glm::vec3* batched_joints,
    const glm::vec2* gt_projected_vertices,
    const float* gt_vertex_rel_depths,
    const float* gt_projected_log_vars,
    const float* gt_vertex_rel_depth_log_vars,
    const glm::vec3* batched_v_jacobian,
    const glm::vec3* batched_j_jacobian,
    float* residual,
    float* jacobian
) {
    dim3 block(256, 1, 1);
    dim3 grid(batch_size, num_views, (num_items + block.x - 1) / block.x);
    AssembleJacobian::AssembleAlignmentDepthJacobianCuda<N_VERTS, UseUncertainty, UseDepth><<<grid, block>>>(
        true,
        batch_size,
        num_views,
        NUM_FLAME_IDENTITY_BASIS,
        num_items,
        num_residuals_per_frame,
        camera_view_stride,
        vertex_indices,
        alignment_weight,
        rel_depth_weight,
        view_mat,
        viewproj_mat,
        batched_vertices,
        batched_joints,
        gt_projected_vertices,
        gt_vertex_rel_depths,
        gt_projected_log_vars,
        gt_vertex_rel_depth_log_vars,
        batched_v_jacobian,
        batched_j_jacobian,
        residual,
        jacobian
    );
}


void FlameTracking::assemble_identity_jacobian(
    const int num_vertices,
    const int batch_size,
    const int num_views,
    const int num_items,
    const int camera_view_stride,
    const int64_t* vertex_indices,
    const float alignment_weight,
    const float rel_depth_weight,

    const float* identity,
    const float identity_reg_weight,

    const glm::mat4* view_mat,
    const glm::mat4* viewproj_mat,
    const glm::vec3* batched_vertices,
    const glm::vec3* batched_joints,
    const glm::vec2* gt_projected_vertices,
    const float* gt_vertex_rel_depths,
    const float* gt_projected_log_vars,
    const float* gt_vertex_rel_depth_log_vars,

    const glm::vec3* batched_v_jacobian,
    const glm::vec3* batched_j_jacobian,

    float* residual,
    float* jacobian
) {
    const bool use_rel_depth = rel_depth_weight > 0.0f;
    const int obs_dim = use_rel_depth ? 3 : 2;
    const int num_residuals_per_frame = num_views * num_items * obs_dim;
    const bool use_uncertainty = (gt_projected_log_vars != nullptr && gt_vertex_rel_depth_log_vars != nullptr);

#define DISPATCH_IDENTITY_ALIGN(N_VERTS, USE_UNCERTAINTY, USE_DEPTH) \
    launch_assemble_identity_alignment_impl<N_VERTS, USE_UNCERTAINTY, USE_DEPTH>( \
        batch_size, num_views, num_items, num_residuals_per_frame, camera_view_stride, vertex_indices, \
        alignment_weight, rel_depth_weight, view_mat, viewproj_mat, batched_vertices, batched_joints, \
        gt_projected_vertices, gt_vertex_rel_depths, gt_projected_log_vars, gt_vertex_rel_depth_log_vars, \
        batched_v_jacobian, batched_j_jacobian, residual, jacobian \
    )
#define DISPATCH_VERTS_CASE(USE_UNCERTAINTY, USE_DEPTH) \
    switch (num_vertices) { \
        case NUM_FLAME_VERTICES_BASE: \
            DISPATCH_IDENTITY_ALIGN(NUM_FLAME_VERTICES_BASE, USE_UNCERTAINTY, USE_DEPTH); \
            break; \
        case NUM_FLAME_VERTICES_ORAL: \
            DISPATCH_IDENTITY_ALIGN(NUM_FLAME_VERTICES_ORAL, USE_UNCERTAINTY, USE_DEPTH); \
            break; \
        default: \
            throw std::runtime_error("assemble_identity_jacobian: unsupported num_vertices"); \
    }
    if (use_uncertainty) {
        if (use_rel_depth) {
            DISPATCH_VERTS_CASE(true, true);
        } else {
            DISPATCH_VERTS_CASE(true, false);
        }
    } else {
        if (use_rel_depth) {
            DISPATCH_VERTS_CASE(false, true);
        } else {
            DISPATCH_VERTS_CASE(false, false);
        }
    }
#undef DISPATCH_VERTS_CASE
#undef DISPATCH_IDENTITY_ALIGN

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
