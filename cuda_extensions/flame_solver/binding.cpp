#include <functional>
#include <string>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <glm/glm.hpp>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "src/constants.h"
#include "src/flame_tracking.h"


std::function<void*(size_t)> create_byte_allocator(torch::Tensor& buffer) {
    return [&buffer](size_t num_bytes) -> void* {
        buffer.resize_({static_cast<int64_t>(num_bytes)});
        return buffer.contiguous().data_ptr();
    };
}


int infer_supported_num_vertices(const int64_t num_vertices, const char* caller) {
    if (num_vertices == NUM_FLAME_VERTICES_BASE || num_vertices == NUM_FLAME_VERTICES_ORAL) {
        return static_cast<int>(num_vertices);
    }
    throw std::runtime_error(std::string(caller) + ": unsupported num_vertices.");
}


std::tuple<torch::Tensor, torch::Tensor>
flame_calc_canonical(
    const torch::Tensor& identity,
    const torch::Tensor& v_template,
    const torch::Tensor& shape_dirs,
    const torch::Tensor& J_regressor_row,
    const torch::Tensor& J_regressor_col,
    const torch::Tensor& J_regressor_values
) {
    if (v_template.dim() != 2 || v_template.size(1) != 3) {
        throw std::runtime_error("flame_calc_canonical: v_template must be [V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(v_template.size(0), "flame_calc_canonical");
    if (shape_dirs.dim() != 2 || shape_dirs.size(0) != NUM_FLAME_IDENTITY_BASIS || shape_dirs.size(1) != num_vertices * 3) {
        throw std::runtime_error("flame_calc_canonical: shape_dirs must be [300,V*3].");
    }
    torch::Tensor v_canonical = torch::zeros_like(v_template);
    torch::Tensor j_canonical = torch::zeros({NUM_FLAME_JOINTS, 3}, v_template.options());
    const int num_J_regressor_nonzero = J_regressor_values.size(0);
    FlameTracking::flame_calc_canonical(
        num_vertices,
        identity.contiguous().data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(v_template.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(shape_dirs.contiguous().data_ptr<float>()),
        num_J_regressor_nonzero,
        J_regressor_row.contiguous().data_ptr<int>(),
        J_regressor_col.contiguous().data_ptr<int>(),
        J_regressor_values.contiguous().data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(v_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(j_canonical.contiguous().data_ptr<float>())
    );
    return std::make_tuple(v_canonical, j_canonical);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
flame_expression_jacobian(
    const torch::Tensor& batched_x,     // [B, dim_x] cuda
    const torch::Tensor& v_canonical,   // [V, 3] cuda
    const torch::Tensor& j_canonical,   // [5, 3] cuda

    const torch::Tensor& expr_dirs,     // [num_expressions, 5023 * 3] cuda
    const torch::Tensor& pose_dirs,     // [ 36, 5023 * 3] cuda
    const torch::Tensor& lbs_weights,   // [5023, 5] cuda
    const int num_expressions
) {
    const int dim_x = dim_x_for(num_expressions);
    if (v_canonical.dim() != 2 || v_canonical.size(1) != 3) {
        throw std::runtime_error("flame_expression_jacobian: v_canonical must be [V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(v_canonical.size(0), "flame_expression_jacobian");
    if (
        expr_dirs.dim() != 2 || expr_dirs.size(0) != num_expressions || expr_dirs.size(1) != num_vertices * 3 ||
        pose_dirs.dim() != 2 || pose_dirs.size(1) != num_vertices * 3 ||
        lbs_weights.dim() != 2 || lbs_weights.size(0) != num_vertices || lbs_weights.size(1) != NUM_FLAME_JOINTS
    ) {
        throw std::runtime_error("flame_expression_jacobian: invalid FLAME asset tensor shape.");
    }
    int batch_size = 1.0;
    torch::Tensor batched_v_output, batched_j_output, batched_v_jacobian, batched_j_jacobian;
    if (batched_x.dim() == 1) {
        if (batched_x.size(0) != dim_x) {
            throw std::runtime_error("flame_expression_jacobian: batched_x size mismatch.");
        }
        batched_v_output = torch::zeros({num_vertices, 3}, v_canonical.options()); // [V, 3]
        batched_j_output = torch::zeros({NUM_FLAME_JOINTS, 3}, v_canonical.options()); // [J, 3]
        batched_v_jacobian = torch::zeros({num_vertices, dim_x, 3}, v_canonical.options()); // [V, dim_x, 3]
        batched_j_jacobian = torch::zeros({NUM_FLAME_JOINTS, dim_x, 3}, v_canonical.options()); // [J, dim_x, 3]
    } else if (batched_x.dim() == 2) {
        if (batched_x.size(1) != dim_x) {
            throw std::runtime_error("flame_expression_jacobian: batched_x size mismatch.");
        }
        batch_size = batched_x.size(0);
        batched_v_output = torch::zeros({batch_size, num_vertices, 3}, v_canonical.options()); // [B, V, 3]
        batched_j_output = torch::zeros({batch_size, NUM_FLAME_JOINTS, 3}, v_canonical.options()); // [B, J, 3]
        batched_v_jacobian = torch::zeros({batch_size, num_vertices, dim_x, 3}, v_canonical.options()); // [B, V, dim_x, 3]
        batched_j_jacobian = torch::zeros({batch_size, NUM_FLAME_JOINTS, dim_x, 3}, v_canonical.options()); // [B, J, dim_x, 3]
    } else {
        throw std::runtime_error("flame_expression_jacobian: invalid batched_x shape.");
    }

    const int work_size = batch_size * (
        NUM_FLAME_POSEFEAT_BASIS +
        NUM_FLAME_JOINTS * 4 * 4 +
        NUM_FLAME_JOINTS * 3 * 3 +
        NUM_FLAME_JOINTS * NUM_FLAME_JOINTS * 4 * 4 +
        NUM_FLAME_JOINTS * 3 * 3
    );
    torch::Tensor work_tensor = torch::zeros({work_size}, v_canonical.options());
    float* work_ptr = work_tensor.data_ptr<float>();
    int offset = 0;
    float* batched_pose_feat_ptr = work_ptr + offset;
    offset += batch_size * NUM_FLAME_POSEFEAT_BASIS;
    glm::mat4* batched_affine_mats_ptr = reinterpret_cast<glm::mat4*>(work_ptr + offset);
    offset += batch_size * NUM_FLAME_JOINTS * 4 * 4;
    glm::mat3* batched_right_jacobian_ptr = reinterpret_cast<glm::mat3*>(work_ptr + offset);
    offset += batch_size * NUM_FLAME_JOINTS * 3 * 3;
    glm::mat4* batched_B_masked_ptr = reinterpret_cast<glm::mat4*>(work_ptr + offset);
    offset += batch_size * NUM_FLAME_JOINTS * NUM_FLAME_JOINTS * 4 * 4;
    glm::mat3* batched_Ar_ptr = reinterpret_cast<glm::mat3*>(work_ptr + offset);

    FlameTracking::flame_expression_jacobian(
        num_vertices,
        num_expressions,
        batch_size,
        batched_x.contiguous().data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(v_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(j_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(expr_dirs.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(pose_dirs.contiguous().data_ptr<float>()),
        lbs_weights.contiguous().data_ptr<float>(),
        batched_pose_feat_ptr,
        batched_affine_mats_ptr,
        batched_right_jacobian_ptr,
        batched_B_masked_ptr,
        batched_Ar_ptr,
        reinterpret_cast<glm::vec3*>(batched_v_output.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_j_output.contiguous().data_ptr<float>()),
        batched_v_jacobian.contiguous().data_ptr<float>(),
        batched_j_jacobian.contiguous().data_ptr<float>()
    );
    return std::make_tuple(batched_v_output, batched_j_output, batched_v_jacobian, batched_j_jacobian);
}


std::tuple<torch::Tensor, torch::Tensor>
flame_expression_forward(
    const torch::Tensor& batched_x,     // [B, dim_x] cuda
    const torch::Tensor& v_canonical,   // [V, 3] cuda
    const torch::Tensor& j_canonical,   // [5, 3] cuda

    const torch::Tensor& expr_dirs,     // [num_expressions, 5023 * 3] cuda
    const torch::Tensor& pose_dirs,     // [ 36, 5023 * 3] cuda
    const torch::Tensor& lbs_weights,   // [5023, 5] cuda
    const int num_expressions
) {
    const int dim_x = dim_x_for(num_expressions);
    if (v_canonical.dim() != 2 || v_canonical.size(1) != 3) {
        throw std::runtime_error("flame_expression_forward: v_canonical must be [V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(v_canonical.size(0), "flame_expression_forward");
    if (
        expr_dirs.dim() != 2 || expr_dirs.size(0) != num_expressions || expr_dirs.size(1) != num_vertices * 3 ||
        pose_dirs.dim() != 2 || pose_dirs.size(1) != num_vertices * 3 ||
        lbs_weights.dim() != 2 || lbs_weights.size(0) != num_vertices || lbs_weights.size(1) != NUM_FLAME_JOINTS
    ) {
        throw std::runtime_error("flame_expression_forward: invalid FLAME asset tensor shape.");
    }
    int batch_size = 1.0;
    torch::Tensor batched_v_output, batched_j_output;
    if (batched_x.dim() == 1) {
        if (batched_x.size(0) != dim_x) {
            throw std::runtime_error("flame_expression_forward: batched_x size mismatch.");
        }
        batched_v_output = torch::zeros({num_vertices, 3}, v_canonical.options());
        batched_j_output = torch::zeros({NUM_FLAME_JOINTS, 3}, v_canonical.options());
    } else if (batched_x.dim() == 2) {
        if (batched_x.size(1) != dim_x) {
            throw std::runtime_error("flame_expression_forward: batched_x size mismatch.");
        }
        batch_size = batched_x.size(0);
        batched_v_output = torch::zeros({batch_size, num_vertices, 3}, v_canonical.options());
        batched_j_output = torch::zeros({batch_size, NUM_FLAME_JOINTS, 3}, v_canonical.options());
    } else {
        throw std::runtime_error("flame_expression_forward: invalid batched_x shape.");
    }

    const int work_size = batch_size * (
        NUM_FLAME_POSEFEAT_BASIS +
        NUM_FLAME_JOINTS * 4 * 4
    );
    torch::Tensor work_tensor = torch::zeros({work_size}, v_canonical.options());
    float* work_ptr = work_tensor.data_ptr<float>();
    int offset = 0;
    float* batched_pose_feat_ptr = work_ptr + offset;
    offset += batch_size * NUM_FLAME_POSEFEAT_BASIS;
    glm::mat4* batched_affine_mats_ptr = reinterpret_cast<glm::mat4*>(work_ptr + offset);

    FlameTracking::flame_expression_forward(
        num_vertices,
        num_expressions,
        batch_size,
        batched_x.contiguous().data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(v_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(j_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(expr_dirs.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(pose_dirs.contiguous().data_ptr<float>()),
        lbs_weights.contiguous().data_ptr<float>(),
        batched_pose_feat_ptr,
        batched_affine_mats_ptr,
        reinterpret_cast<glm::vec3*>(batched_v_output.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_j_output.contiguous().data_ptr<float>())
    );
    return std::make_tuple(batched_v_output, batched_j_output);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
flame_identity_jacobian(
    const torch::Tensor& batched_x,     // [B, dim_x] or [dim_x]
    const torch::Tensor& v_canonical,   // [V, 3]
    const torch::Tensor& j_canonical,   // [5, 3]

    const torch::Tensor& ref_joints_dirs, // [300, 5, 3]
    const torch::Tensor& shape_dirs,    // [300, 5023 * 3]
    const torch::Tensor& expr_dirs,     // [num_expressions, 5023 * 3]
    const torch::Tensor& pose_dirs,     // [ 36, 5023 * 3]
    const torch::Tensor& lbs_weights,   // [5023, 5]
    const int num_expressions
) {
    const int dim_x = dim_x_for(num_expressions);
    if (v_canonical.dim() != 2 || v_canonical.size(1) != 3) {
        throw std::runtime_error("flame_identity_jacobian: v_canonical must be [V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(v_canonical.size(0), "flame_identity_jacobian");
    if (
        shape_dirs.dim() != 2 || shape_dirs.size(0) != NUM_FLAME_IDENTITY_BASIS || shape_dirs.size(1) != num_vertices * 3 ||
        expr_dirs.dim() != 2 || expr_dirs.size(0) != num_expressions || expr_dirs.size(1) != num_vertices * 3 ||
        pose_dirs.dim() != 2 || pose_dirs.size(1) != num_vertices * 3 ||
        lbs_weights.dim() != 2 || lbs_weights.size(0) != num_vertices || lbs_weights.size(1) != NUM_FLAME_JOINTS
    ) {
        throw std::runtime_error("flame_identity_jacobian: invalid FLAME asset tensor shape.");
    }
    int batch_size = 1.0;
    torch::Tensor batched_v_output, batched_j_output, batched_v_jacobian, batched_j_jacobian;
    if (batched_x.dim() == 1) {
        if (batched_x.size(0) != dim_x) {
            throw std::runtime_error("flame_identity_jacobian: batched_x size mismatch.");
        }
        batched_v_output = torch::zeros({num_vertices, 3}, v_canonical.options()); // [V, 3]
        batched_j_output = torch::zeros({NUM_FLAME_JOINTS, 3}, v_canonical.options()); // [J, 3]
        batched_v_jacobian = torch::zeros({num_vertices, NUM_FLAME_IDENTITY_BASIS, 3}, v_canonical.options()); // [V, 300, 3]
        batched_j_jacobian = torch::zeros({NUM_FLAME_JOINTS, NUM_FLAME_IDENTITY_BASIS, 3}, v_canonical.options()); // [J, 300, 3]
    } else if (batched_x.dim() == 2) {
        if (batched_x.size(1) != dim_x) {
            throw std::runtime_error("flame_identity_jacobian: batched_x size mismatch.");
        }
        batch_size = batched_x.size(0);
        batched_v_output = torch::zeros({batch_size, num_vertices, 3}, v_canonical.options()); // [B, V, 3]
        batched_j_output = torch::zeros({batch_size, NUM_FLAME_JOINTS, 3}, v_canonical.options()); // [B, J, 3]
        batched_v_jacobian = torch::zeros({batch_size, num_vertices, NUM_FLAME_IDENTITY_BASIS, 3}, v_canonical.options()); // [B, V, 300, 3]
        batched_j_jacobian = torch::zeros({batch_size, NUM_FLAME_JOINTS, NUM_FLAME_IDENTITY_BASIS, 3}, v_canonical.options()); // [B, J, 300, 3]
    } else {
        throw std::runtime_error("flame_identity_jacobian: invalid batched_x shape.");
    }

    const int work_size = batch_size * (
        NUM_FLAME_POSEFEAT_BASIS +
        NUM_FLAME_JOINTS * 4 * 4 +
        NUM_FLAME_JOINTS * 4 * 4
    );
    torch::Tensor work_tensor = torch::zeros({work_size}, v_canonical.options());
    float* work_ptr = work_tensor.data_ptr<float>();
    int offset = 0;
    float* batched_pose_feat_ptr = work_ptr + offset;
    offset += batch_size * NUM_FLAME_POSEFEAT_BASIS;
    glm::mat4* batched_rel_affine_mats = reinterpret_cast<glm::mat4*>(work_ptr + offset);
    offset += batch_size * NUM_FLAME_JOINTS * 4 * 4;
    glm::mat4* batched_affine_mats = reinterpret_cast<glm::mat4*>(work_ptr + offset);

    FlameTracking::flame_identity_jacobian(
        num_vertices,
        num_expressions,
        batch_size,
        batched_x.contiguous().data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(v_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(j_canonical.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(ref_joints_dirs.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(shape_dirs.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(expr_dirs.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(pose_dirs.contiguous().data_ptr<float>()),
        lbs_weights.contiguous().data_ptr<float>(),
        batched_pose_feat_ptr,
        batched_rel_affine_mats,
        batched_affine_mats,
        reinterpret_cast<glm::vec3*>(batched_v_output.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_j_output.contiguous().data_ptr<float>()),
        batched_v_jacobian.contiguous().data_ptr<float>(),
        batched_j_jacobian.contiguous().data_ptr<float>()
    );
    return std::make_tuple(batched_v_output, batched_j_output, batched_v_jacobian, batched_j_jacobian);
}


std::tuple<torch::Tensor, torch::Tensor>
assemble_x_jacobian(
    const torch::Tensor& view_mat,
    const torch::Tensor& viewproj_mat,
    const torch::Tensor& batched_vertices,
    const torch::Tensor& batched_joints,
    const torch::Tensor& gt_projected_vertices,
    const torch::Tensor& gt_vertex_rel_depths,
    const torch::Tensor& gt_projected_log_vars,
    const torch::Tensor& gt_vertex_rel_depth_log_vars,

    const torch::Tensor& batched_x,

    const torch::Tensor& batched_v_jacobian,
    const torch::Tensor& batched_j_jacobian,

    const float alignment_weight,
    const float rel_depth_weight,
    const float exp_reg_weight,
    const float pose_reg_weight,
    const torch::Tensor& vertex_indices,
    const int num_expressions
) {
    const int dim_x = dim_x_for(num_expressions);
    if (batched_vertices.dim() != 3 || batched_vertices.size(2) != 3) {
        throw std::runtime_error("assemble_x_jacobian: batched_vertices must be [T,V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(batched_vertices.size(1), "assemble_x_jacobian");

    // Camera matrices: [C, 4, 4] (shared) or [T, C, 4, 4] (per-batch)
    int num_views, camera_stride;
    torch::Tensor view_mat_flat, viewproj_mat_flat;
    if (view_mat.dim() == 4) {
        num_views = view_mat.size(1);
        camera_stride = num_views;
        view_mat_flat = view_mat.reshape({-1, 4, 4}).contiguous();
        viewproj_mat_flat = viewproj_mat.reshape({-1, 4, 4}).contiguous();
    } else if (view_mat.dim() == 3) {
        num_views = view_mat.size(0);
        camera_stride = 0;
        view_mat_flat = view_mat.contiguous();
        viewproj_mat_flat = viewproj_mat.contiguous();
    } else {
        throw std::runtime_error("assemble_x_jacobian: view_mat must be [C,4,4] or [T,C,4,4].");
    }

    if (
        view_mat.size(-2) != 4 || view_mat.size(-1) != 4 ||
        viewproj_mat.size(-2) != 4 || viewproj_mat.size(-1) != 4 ||
        batched_vertices.dim() != 3 || batched_vertices.size(1) != num_vertices || batched_vertices.size(2) != 3 ||
        batched_joints.dim() != 3 || batched_joints.size(1) != NUM_FLAME_JOINTS || batched_joints.size(2) != 3 ||
        gt_projected_vertices.dim() != 4 || gt_projected_vertices.size(3) != 2 ||
        gt_vertex_rel_depths.dim() != 4 || gt_vertex_rel_depths.size(3) != 1 ||
        batched_x.dim() != 2 || batched_x.size(1) != dim_x ||
        batched_v_jacobian.dim() != 4 || batched_v_jacobian.size(1) != num_vertices || batched_v_jacobian.size(2) != dim_x || batched_v_jacobian.size(3) != 3 ||
        batched_j_jacobian.dim() != 4 || batched_j_jacobian.size(1) != NUM_FLAME_JOINTS || batched_j_jacobian.size(2) != dim_x || batched_j_jacobian.size(3) != 3 ||
        vertex_indices.dim() != 1
    ) {
        throw std::runtime_error("assemble_x_jacobian: invalid tensor shape.");
    }
    const int batch_size = batched_x.size(0);
    if (view_mat.dim() == 4 && view_mat.size(0) != batch_size) {
        throw std::runtime_error("assemble_x_jacobian: view_mat [T,C,4,4] T must match batch_size.");
    }
    const bool has_projected_log_var = gt_projected_log_vars.numel() > 0;
    const bool has_depth_log_var = gt_vertex_rel_depth_log_vars.numel() > 0;
    if (has_projected_log_var != has_depth_log_var) {
        throw std::runtime_error("assemble_x_jacobian: projected/depth log_var tensors must both be empty or both be non-empty.");
    }
    if (
        has_projected_log_var && (
            gt_projected_log_vars.dim() != 4 || gt_projected_log_vars.size(3) != 1 ||
            gt_projected_log_vars.size(0) != batch_size ||
            gt_projected_log_vars.size(1) != num_views ||
            gt_projected_log_vars.size(2) != num_vertices
        )
    ) {
        throw std::runtime_error("assemble_x_jacobian: gt_projected_log_vars must be [T,C,V,1] when provided.");
    }
    if (
        has_depth_log_var && (
            gt_vertex_rel_depth_log_vars.dim() != 4 || gt_vertex_rel_depth_log_vars.size(3) != 1 ||
            gt_vertex_rel_depth_log_vars.size(0) != batch_size ||
            gt_vertex_rel_depth_log_vars.size(1) != num_views ||
            gt_vertex_rel_depth_log_vars.size(2) != num_vertices
        )
    ) {
        throw std::runtime_error("assemble_x_jacobian: gt_vertex_rel_depth_log_vars must be [T,C,V,1] when provided.");
    }
    if (
        batched_vertices.size(0) != batch_size ||
        batched_joints.size(0) != batch_size ||
        gt_projected_vertices.size(0) != batch_size ||
        gt_projected_vertices.size(1) != num_views ||
        gt_projected_vertices.size(2) != num_vertices ||
        gt_vertex_rel_depths.size(0) != batch_size ||
        gt_vertex_rel_depths.size(1) != num_views ||
        gt_vertex_rel_depths.size(2) != num_vertices ||
        batched_v_jacobian.size(0) != batch_size ||
        batched_j_jacobian.size(0) != batch_size
    ) {
        throw std::runtime_error("assemble_x_jacobian: inconsistent tensor shape.");
    }

    const int num_items = vertex_indices.numel() > 0 ? vertex_indices.numel() : num_vertices;
    const bool use_rel_depth = rel_depth_weight > 0.0f;
    const int obs_dim = use_rel_depth ? 3 : 2;
    const int num_residuals = num_views * num_items * obs_dim + num_expressions + (NUM_FLAME_JOINTS - 1) * 3;
    torch::Tensor residual = torch::zeros({batch_size, num_residuals}, batched_vertices.options());
    torch::Tensor jacobian = torch::zeros({batch_size, num_residuals, dim_x}, batched_vertices.options());

    FlameTracking::assemble_x_jacobian(
        num_vertices,
        num_expressions,
        batch_size, num_views, num_items, camera_stride,
        vertex_indices.numel() > 0 ?  vertex_indices.contiguous().data_ptr<int64_t>() : nullptr,
        alignment_weight, rel_depth_weight,
        batched_x.contiguous().data_ptr<float>(),
        exp_reg_weight, pose_reg_weight,
        reinterpret_cast<glm::mat4*>(view_mat_flat.data_ptr<float>()),
        reinterpret_cast<glm::mat4*>(viewproj_mat_flat.data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_vertices.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_joints.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec2*>(gt_projected_vertices.contiguous().data_ptr<float>()),
        gt_vertex_rel_depths.contiguous().data_ptr<float>(),
        has_projected_log_var ? gt_projected_log_vars.contiguous().data_ptr<float>() : nullptr,
        has_depth_log_var ? gt_vertex_rel_depth_log_vars.contiguous().data_ptr<float>() : nullptr,
        reinterpret_cast<glm::vec3*>(batched_v_jacobian.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_j_jacobian.contiguous().data_ptr<float>()),
        residual.contiguous().data_ptr<float>(),
        jacobian.contiguous().data_ptr<float>()
    );
    return std::make_tuple(residual, jacobian);
}


std::tuple<torch::Tensor, torch::Tensor>
assemble_identity_jacobian(
    const torch::Tensor& view_mat,
    const torch::Tensor& viewproj_mat,
    const torch::Tensor& batched_vertices,
    const torch::Tensor& batched_joints,
    const torch::Tensor& gt_projected_vertices,
    const torch::Tensor& gt_vertex_rel_depths,
    const torch::Tensor& gt_projected_log_vars,
    const torch::Tensor& gt_vertex_rel_depth_log_vars,

    const torch::Tensor& identity,

    const torch::Tensor& batched_v_jacobian,
    const torch::Tensor& batched_j_jacobian,

    const float alignment_weight,
    const float rel_depth_weight,
    const float identity_reg_weight,
    const torch::Tensor& vertex_indices
) {
    if (batched_vertices.dim() != 3 || batched_vertices.size(2) != 3) {
        throw std::runtime_error("assemble_identity_jacobian: batched_vertices must be [T,V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(batched_vertices.size(1), "assemble_identity_jacobian");

    // Camera matrices: [C, 4, 4] (shared) or [T, C, 4, 4] (per-batch)
    int num_views, camera_stride;
    torch::Tensor view_mat_flat, viewproj_mat_flat;
    if (view_mat.dim() == 4) {
        num_views = view_mat.size(1);
        camera_stride = num_views;
        view_mat_flat = view_mat.reshape({-1, 4, 4}).contiguous();
        viewproj_mat_flat = viewproj_mat.reshape({-1, 4, 4}).contiguous();
    } else if (view_mat.dim() == 3) {
        num_views = view_mat.size(0);
        camera_stride = 0;
        view_mat_flat = view_mat.contiguous();
        viewproj_mat_flat = viewproj_mat.contiguous();
    } else {
        throw std::runtime_error("assemble_identity_jacobian: view_mat must be [C,4,4] or [T,C,4,4].");
    }

    if (
        view_mat.size(-2) != 4 || view_mat.size(-1) != 4 ||
        viewproj_mat.size(-2) != 4 || viewproj_mat.size(-1) != 4 ||
        batched_vertices.dim() != 3 || batched_vertices.size(1) != num_vertices || batched_vertices.size(2) != 3 ||
        batched_joints.dim() != 3 || batched_joints.size(1) != NUM_FLAME_JOINTS || batched_joints.size(2) != 3 ||
        gt_projected_vertices.dim() != 4 || gt_projected_vertices.size(3) != 2 ||
        gt_vertex_rel_depths.dim() != 4 || gt_vertex_rel_depths.size(3) != 1 ||
        identity.dim() != 1 || identity.size(0) != NUM_FLAME_IDENTITY_BASIS ||
        batched_v_jacobian.dim() != 4 || batched_v_jacobian.size(1) != num_vertices || batched_v_jacobian.size(2) != NUM_FLAME_IDENTITY_BASIS || batched_v_jacobian.size(3) != 3 ||
        batched_j_jacobian.dim() != 4 || batched_j_jacobian.size(1) != NUM_FLAME_JOINTS || batched_j_jacobian.size(2) != NUM_FLAME_IDENTITY_BASIS || batched_j_jacobian.size(3) != 3 ||
        vertex_indices.dim() != 1
    ) {
        throw std::runtime_error("assemble_identity_jacobian: invalid tensor shape.");
    }
    const int batch_size = batched_vertices.size(0);
    if (view_mat.dim() == 4 && view_mat.size(0) != batch_size) {
        throw std::runtime_error("assemble_identity_jacobian: view_mat [T,C,4,4] T must match batch_size.");
    }
    const bool has_projected_log_var = gt_projected_log_vars.numel() > 0;
    const bool has_depth_log_var = gt_vertex_rel_depth_log_vars.numel() > 0;
    if (has_projected_log_var != has_depth_log_var) {
        throw std::runtime_error("assemble_identity_jacobian: projected/depth log_var tensors must both be empty or both be non-empty.");
    }
    if (
        has_projected_log_var && (
            gt_projected_log_vars.dim() != 4 || gt_projected_log_vars.size(3) != 1 ||
            gt_projected_log_vars.size(0) != batch_size ||
            gt_projected_log_vars.size(1) != num_views ||
            gt_projected_log_vars.size(2) != num_vertices
        )
    ) {
        throw std::runtime_error("assemble_identity_jacobian: gt_projected_log_vars must be [T,C,V,1] when provided.");
    }
    if (
        has_depth_log_var && (
            gt_vertex_rel_depth_log_vars.dim() != 4 || gt_vertex_rel_depth_log_vars.size(3) != 1 ||
            gt_vertex_rel_depth_log_vars.size(0) != batch_size ||
            gt_vertex_rel_depth_log_vars.size(1) != num_views ||
            gt_vertex_rel_depth_log_vars.size(2) != num_vertices
        )
    ) {
        throw std::runtime_error("assemble_identity_jacobian: gt_vertex_rel_depth_log_vars must be [T,C,V,1] when provided.");
    }
    if (
        batched_joints.size(0) != batch_size ||
        gt_projected_vertices.size(0) != batch_size ||
        gt_projected_vertices.size(1) != num_views ||
        gt_projected_vertices.size(2) != num_vertices ||
        gt_vertex_rel_depths.size(0) != batch_size ||
        gt_vertex_rel_depths.size(1) != num_views ||
        gt_vertex_rel_depths.size(2) != num_vertices ||
        batched_v_jacobian.size(0) != batch_size ||
        batched_j_jacobian.size(0) != batch_size
    ) {
        throw std::runtime_error("assemble_identity_jacobian: inconsistent tensor shape.");
    }

    const int num_items = vertex_indices.numel() > 0 ? vertex_indices.numel() : num_vertices;
    const bool use_rel_depth = rel_depth_weight > 0.0f;
    const int obs_dim = use_rel_depth ? 3 : 2;
    const int num_residuals = batch_size * num_views * num_items * obs_dim + NUM_FLAME_IDENTITY_BASIS;
    torch::Tensor residual = torch::zeros({num_residuals}, batched_vertices.options());
    torch::Tensor jacobian = torch::zeros({num_residuals, NUM_FLAME_IDENTITY_BASIS}, batched_vertices.options());

    FlameTracking::assemble_identity_jacobian(
        num_vertices,
        batch_size, num_views, num_items, camera_stride,
        vertex_indices.numel() > 0 ?  vertex_indices.contiguous().data_ptr<int64_t>() : nullptr,
        alignment_weight, rel_depth_weight,
        identity.contiguous().data_ptr<float>(),
        identity_reg_weight,
        reinterpret_cast<glm::mat4*>(view_mat_flat.data_ptr<float>()),
        reinterpret_cast<glm::mat4*>(viewproj_mat_flat.data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_vertices.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_joints.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec2*>(gt_projected_vertices.contiguous().data_ptr<float>()),
        gt_vertex_rel_depths.contiguous().data_ptr<float>(),
        has_projected_log_var ? gt_projected_log_vars.contiguous().data_ptr<float>() : nullptr,
        has_depth_log_var ? gt_vertex_rel_depth_log_vars.contiguous().data_ptr<float>() : nullptr,
        reinterpret_cast<glm::vec3*>(batched_v_jacobian.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_j_jacobian.contiguous().data_ptr<float>()),
        residual.contiguous().data_ptr<float>(),
        jacobian.contiguous().data_ptr<float>()
    );
    return std::make_tuple(residual, jacobian);
}



std::tuple<torch::Tensor, torch::Tensor>
assemble_x_direct_jacobian(
    const torch::Tensor& gt_vertices,
    const torch::Tensor& batched_x,
    const torch::Tensor& batched_vertices,
    const torch::Tensor& batched_v_jacobian,
    const float vertex_weight,
    const float exp_reg_weight,
    const float pose_reg_weight,
    const torch::Tensor& vertex_indices,
    const int num_expressions
) {
    const int dim_x = dim_x_for(num_expressions);
    if (batched_vertices.dim() != 3 || batched_vertices.size(2) != 3) {
        throw std::runtime_error("assemble_x_direct_jacobian: batched_vertices must be [T,V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(batched_vertices.size(1), "assemble_x_direct_jacobian");
    if (
        gt_vertices.dim() != 3 || gt_vertices.size(1) != num_vertices || gt_vertices.size(2) != 3 ||
        batched_x.dim() != 2 || batched_x.size(1) != dim_x ||
        batched_vertices.dim() != 3 || batched_vertices.size(1) != num_vertices || batched_vertices.size(2) != 3 ||
        batched_v_jacobian.dim() != 4 || batched_v_jacobian.size(1) != num_vertices || batched_v_jacobian.size(2) != dim_x || batched_v_jacobian.size(3) != 3 ||
        vertex_indices.dim() != 1
    ) {
        throw std::runtime_error("assemble_x_direct_jacobian: invalid tensor shape.");
    }

    const int batch_size = batched_x.size(0);
    if (
        gt_vertices.size(0) != batch_size ||
        batched_vertices.size(0) != batch_size ||
        batched_v_jacobian.size(0) != batch_size
    ) {
        throw std::runtime_error("assemble_x_direct_jacobian: inconsistent tensor shape.");
    }

    const int num_items = vertex_indices.numel() > 0 ? vertex_indices.numel() : num_vertices;
    const int num_residuals = num_items * 3 + num_expressions + (NUM_FLAME_JOINTS - 1) * 3;
    torch::Tensor residual = torch::zeros({batch_size, num_residuals}, batched_vertices.options());
    torch::Tensor jacobian = torch::zeros({batch_size, num_residuals, dim_x}, batched_vertices.options());

    FlameTracking::assemble_x_direct_jacobian(
        num_vertices,
        num_expressions,
        batch_size,
        num_items,
        vertex_indices.numel() > 0 ? vertex_indices.contiguous().data_ptr<int64_t>() : nullptr,
        vertex_weight,
        batched_x.contiguous().data_ptr<float>(),
        exp_reg_weight,
        pose_reg_weight,
        reinterpret_cast<glm::vec3*>(batched_vertices.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(gt_vertices.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_v_jacobian.contiguous().data_ptr<float>()),
        residual.contiguous().data_ptr<float>(),
        jacobian.contiguous().data_ptr<float>()
    );
    return std::make_tuple(residual, jacobian);
}


std::tuple<torch::Tensor, torch::Tensor>
assemble_identity_direct_jacobian(
    const torch::Tensor& gt_vertices,
    const torch::Tensor& identity,
    const torch::Tensor& batched_vertices,
    const torch::Tensor& batched_v_jacobian,
    const float vertex_weight,
    const float identity_reg_weight,
    const torch::Tensor& vertex_indices
) {
    if (batched_vertices.dim() != 3 || batched_vertices.size(2) != 3) {
        throw std::runtime_error("assemble_identity_direct_jacobian: batched_vertices must be [T,V,3].");
    }
    const int num_vertices = infer_supported_num_vertices(batched_vertices.size(1), "assemble_identity_direct_jacobian");
    if (
        gt_vertices.dim() != 3 || gt_vertices.size(1) != num_vertices || gt_vertices.size(2) != 3 ||
        identity.dim() != 1 || identity.size(0) != NUM_FLAME_IDENTITY_BASIS ||
        batched_vertices.dim() != 3 || batched_vertices.size(1) != num_vertices || batched_vertices.size(2) != 3 ||
        batched_v_jacobian.dim() != 4 || batched_v_jacobian.size(1) != num_vertices || batched_v_jacobian.size(2) != NUM_FLAME_IDENTITY_BASIS || batched_v_jacobian.size(3) != 3 ||
        vertex_indices.dim() != 1
    ) {
        throw std::runtime_error("assemble_identity_direct_jacobian: invalid tensor shape.");
    }

    const int batch_size = batched_vertices.size(0);
    if (
        gt_vertices.size(0) != batch_size ||
        batched_v_jacobian.size(0) != batch_size
    ) {
        throw std::runtime_error("assemble_identity_direct_jacobian: inconsistent tensor shape.");
    }

    const int num_items = vertex_indices.numel() > 0 ? vertex_indices.numel() : num_vertices;
    const int num_residuals = batch_size * num_items * 3 + NUM_FLAME_IDENTITY_BASIS;
    torch::Tensor residual = torch::zeros({num_residuals}, batched_vertices.options());
    torch::Tensor jacobian = torch::zeros({num_residuals, NUM_FLAME_IDENTITY_BASIS}, batched_vertices.options());

    FlameTracking::assemble_identity_direct_jacobian(
        num_vertices,
        batch_size,
        num_items,
        vertex_indices.numel() > 0 ? vertex_indices.contiguous().data_ptr<int64_t>() : nullptr,
        vertex_weight,
        identity.contiguous().data_ptr<float>(),
        identity_reg_weight,
        reinterpret_cast<glm::vec3*>(batched_vertices.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(gt_vertices.contiguous().data_ptr<float>()),
        reinterpret_cast<glm::vec3*>(batched_v_jacobian.contiguous().data_ptr<float>()),
        residual.contiguous().data_ptr<float>(),
        jacobian.contiguous().data_ptr<float>()
    );
    return std::make_tuple(residual, jacobian);
}



torch::Tensor cholesky_solve(
    const int64_t& cublas_handle,
    const int64_t& cusolver_handle,
    const torch::Tensor& jacobian,  // [R, X] or [B, R, X]
    const torch::Tensor& residual,  // [R] or [B, R]
    const float lambda = 0.0f
) {
    if (jacobian.dim() == 2 && residual.dim() == 1) {
        const int m = jacobian.size(0);
        const int n = jacobian.size(1);
        torch::Tensor x = torch::zeros({n}, residual.options());
        torch::Tensor mem = torch::empty({0}, residual.options().dtype(torch::kUInt8));
        std::function<void*(size_t N)> mem_func = create_byte_allocator(mem);

        FlameTracking::cholesky_solve(
            reinterpret_cast<cublasHandle_t>(cublas_handle),
            reinterpret_cast<cusolverDnHandle_t>(cusolver_handle),
            mem_func,
            jacobian.contiguous().data_ptr<float>(),
            residual.contiguous().data_ptr<float>(),
            x.contiguous().data_ptr<float>(),
            m, n, lambda
        );
        return x;
    } else if (jacobian.dim() == 3 && residual.dim() == 2) {
        const int batch_size = jacobian.size(0);
        const int m = jacobian.size(1);
        const int n = jacobian.size(2);
        torch::Tensor x = torch::zeros({batch_size, n}, residual.options());
        torch::Tensor mem = torch::empty({0}, residual.options().dtype(torch::kUInt8));
        std::function<void*(size_t N)> mem_func = create_byte_allocator(mem);

        FlameTracking::cholesky_solve_batched(
            reinterpret_cast<cublasHandle_t>(cublas_handle),
            reinterpret_cast<cusolverDnHandle_t>(cusolver_handle),
            mem_func,
            jacobian.contiguous().data_ptr<float>(),
            residual.contiguous().data_ptr<float>(),
            x.contiguous().data_ptr<float>(),
            m, n, batch_size, lambda
        );
        return x;
    } else {
        throw std::runtime_error("cholesky_solve: invalid tensor shape.");
    }
}


int64_t create_cublas_handle() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);
    return reinterpret_cast<int64_t>(handle);
}


void destroy_cublas_handle(const int64_t& handle) {
    cublasDestroy(reinterpret_cast<cublasHandle_t>(handle));
}


int64_t create_cusolver_handle() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cusolverDnSetStream(handle, stream);
    return reinterpret_cast<int64_t>(handle);
}

void destroy_cusolver_handle(const int64_t& handle) {
    cusolverDnDestroy(reinterpret_cast<cusolverDnHandle_t>(handle));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flame_calc_canonical", &flame_calc_canonical, "FLAME calculate canonical vertices and joints");
    m.def("flame_expression_jacobian", &flame_expression_jacobian, "FLAME expression jacobian");
    m.def("flame_expression_forward", &flame_expression_forward, "FLAME expression forward");
    m.def("flame_identity_jacobian", &flame_identity_jacobian, "FLAME identity jacobian");
    m.def("assemble_x_jacobian", &assemble_x_jacobian, "Assemble jacobian matrix for optimizing x");
    m.def("assemble_identity_jacobian", &assemble_identity_jacobian, "Assemble jacobian matrix for optimizing identity");
    m.def("assemble_x_direct_jacobian", &assemble_x_direct_jacobian, "Assemble direct 3D jacobian matrix for optimizing x");
    m.def("assemble_identity_direct_jacobian", &assemble_identity_direct_jacobian, "Assemble direct 3D jacobian matrix for optimizing identity");
    m.def("cholesky_solve", &cholesky_solve, "Solve update vector with Cholesky");

    m.def("create_cublas_handle", &create_cublas_handle, "Create cublas handle");
    m.def("destroy_cublas_handle", &destroy_cublas_handle, "Destroy cublas handle");
    m.def("create_cusolver_handle", &create_cusolver_handle, "Create cusolver handle");
    m.def("destroy_cusolver_handle", &destroy_cusolver_handle, "Destroy cusolver handle");
}
