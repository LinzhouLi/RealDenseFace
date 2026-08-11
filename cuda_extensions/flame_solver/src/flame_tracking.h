#pragma once
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <glm/glm.hpp>

namespace AssembleJacobian {
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
    );

    __global__ void AssembleIdenRegJacobianCuda(
        const bool calc_jacobian,
        const int num_offsets,
        const float* __restrict__ unknows,
        const float iden_reg_weight,
        float* __restrict__ residual,
        float* __restrict__ jacobian
    );
}


namespace FlameTracking {
    void flame_calc_canonical(
        const int num_vertices,
        const float* identity,
        const glm::vec3* v_template,
        const glm::vec3* shape_dirs,

        const int num_J_regressor_nonzero,
        const int* J_regressor_row,
        const int* J_regressor_col,
        const float* J_regressor_values,

        glm::vec3* v_canonical,
        glm::vec3* j_canonical
    );

    void flame_expression_jacobian(
        const int num_vertices,
        const int num_expressions,
        const int batch_size,
        const float* batched_x,
        const glm::vec3* v_canonical,
        const glm::vec3* j_canonical,

        const glm::vec3* expr_dirs,         // [B, V, 3]
        const glm::vec3* pose_dirs,         // [36, V, 3]
        const float* lbs_weights,           // [V, J]

        float* batched_pose_feat,
        glm::mat4* batched_affine_mats,
        glm::mat3* batched_right_jacobian,
        glm::mat4* batched_B_masked,
        glm::mat3* batched_Ar,

        glm::vec3* batched_v_output,
        glm::vec3* batched_j_output,
        float* batched_v_jacobian,
        float* batched_j_jacobian
    );

    void flame_expression_forward(
        const int num_vertices,
        const int num_expressions,
        const int batch_size,
        const float* batched_x,
        const glm::vec3* v_canonical,
        const glm::vec3* j_canonical,

        const glm::vec3* expr_dirs,         // [B, V, 3]
        const glm::vec3* pose_dirs,         // [36, V, 3]
        const float* lbs_weights,           // [V, J]

        float* batched_pose_feat,
        glm::mat4* batched_affine_mats,

        glm::vec3* batched_v_output,
        glm::vec3* batched_j_output
    );

    void flame_identity_jacobian(
        const int num_vertices,
        const int num_expressions,
        const int batch_size,
        const float* batched_x,
        const glm::vec3* v_canonical,
        const glm::vec3* joints,

        const glm::vec3* ref_joints_dirs,
        const glm::vec3* shape_dirs,
        const glm::vec3* expr_dirs,
        const glm::vec3* pose_dirs,
        const float* lbs_weights,

        float* batched_pose_feat,
        glm::mat4* batched_rel_affine_mats,
        glm::mat4* batched_affine_mats,

        glm::vec3* batched_v_output,
        glm::vec3* batched_j_output,
        float* batched_v_jacobian,
        float* batched_j_jacobian
    );

    void assemble_x_jacobian(
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
        const glm::vec3* batched_vertices,      // [T, V, 3]
        const glm::vec3* batched_joints,        // [T, J, 3]
        const glm::vec2* gt_projected_vertices, // [T, C, V, 2]
        const float* gt_vertex_rel_depths,      // [T, C, V, 1]
        const float* gt_projected_log_vars,     // [T, C, V, 1], nullptr means disabled
        const float* gt_vertex_rel_depth_log_vars, // [T, C, V, 1], nullptr means disabled

        const glm::vec3* batched_v_jacobian,    // [T, V, X, 3]
        const glm::vec3* batched_j_jacobian,    // [T, J, X, 3]

        float* residual,                    // [T, R], per-item obs dim is 3 if rel_depth_weight > 0, else 2
        float* jacobian                     // [T, R, X], per-item obs dim is 3 if rel_depth_weight > 0, else 2
    );

    void assemble_identity_jacobian(
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
        const glm::vec3* batched_vertices,      // [T, V, 3]
        const glm::vec3* batched_joints,        // [T, J, 3]
        const glm::vec2* gt_projected_vertices, // [T, C, V, 2]
        const float* gt_vertex_rel_depths,      // [T, C, V, 1]
        const float* gt_projected_log_vars,     // [T, C, V, 1], nullptr means disabled
        const float* gt_vertex_rel_depth_log_vars, // [T, C, V, 1], nullptr means disabled

        const glm::vec3* batched_v_jacobian,    // [T, V, X, 3]
        const glm::vec3* batched_j_jacobian,    // [T, J, X, 3]

        float* residual,                    // [R], per-item obs dim is 3 if rel_depth_weight > 0, else 2
        float* jacobian                     // [R, X], per-item obs dim is 3 if rel_depth_weight > 0, else 2
    );

    void assemble_x_direct_jacobian(
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
    );

    void assemble_identity_direct_jacobian(
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
    );

    void cholesky_solve(
        cublasHandle_t cublas_handle,
        cusolverDnHandle_t cusolver_handle,
        std::function<void*(size_t N)> mem_func,
        const float* d_J,   // [R, X]
        const float* d_r,   // [R]
        float* d_x,         // [X]  
        const int m,        // num_residuals
        const int n,        // num_unknowns
        const float lambda
    );

    void cholesky_solve_batched(
        cublasHandle_t cublas_handle,
        cusolverDnHandle_t cusolver_handle,
        std::function<void*(size_t N)> mem_func,
        const float* d_J,   // [B, R, X]
        const float* d_r,   // [B, R]
        float* d_x,         // [B, X]  
        const int m,        // num_residuals
        const int n,        // num_unknowns
        const int batch_size,
        const float lambda
    );
    
}
