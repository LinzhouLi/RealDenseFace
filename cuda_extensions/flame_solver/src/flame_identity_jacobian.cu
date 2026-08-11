#include "utils.h"
#include "flame_tracking.h"


namespace IdentityJacobian {

// Templated on N_EXPR because the kernel reads pose/translation slots inside
// the x layout (which depend on N_EXPR), but the rest of its work is identity-only.
template<int N_EXPR>
__global__ void SkeletonTreeCuda(
    const int batch_size,
    const glm::vec3* __restrict__ j_canonical,
    const float* __restrict__ batched_x,

    float* __restrict__ batched_pose_feat,
    glm::mat4* __restrict__ batched_rel_affine_mats,
    glm::mat4* __restrict__ batched_affine_mats,
    glm::vec3* __restrict__ batched_j_output
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float* pose_ptr = batched_x + batch_idx * DIM_X + N_EXPR;
    float* pose_feat = batched_pose_feat + batch_idx * NUM_FLAME_POSEFEAT_BASIS;
    glm::mat4* rel_affine_mats = batched_rel_affine_mats + batch_idx * NUM_FLAME_JOINTS;
    glm::mat4* affine_mats = batched_affine_mats + batch_idx * NUM_FLAME_JOINTS;
    glm::vec3* j_output = batched_j_output + batch_idx * NUM_FLAME_JOINTS;

    const glm::vec3 translation = glm::vec3(
        pose_ptr[NUM_FLAME_JOINTS * 3 + 0],
        pose_ptr[NUM_FLAME_JOINTS * 3 + 1],
        pose_ptr[NUM_FLAME_JOINTS * 3 + 2]
    );

    glm::vec3 rel_joints[NUM_FLAME_JOINTS];
    rel_joints[0] = j_canonical[0];
    rel_joints[1] = j_canonical[1] - j_canonical[0];
    rel_joints[2] = j_canonical[2] - j_canonical[1];
    rel_joints[3] = j_canonical[3] - j_canonical[1];
    rel_joints[4] = j_canonical[4] - j_canonical[1];

    glm::mat3 rot_mats[NUM_FLAME_JOINTS];
    for (int i = 0; i < NUM_FLAME_JOINTS; i++) {
        glm::vec3 pose(pose_ptr[i * 3 + 0], pose_ptr[i * 3 + 1], pose_ptr[i * 3 + 2]);
        rot_mats[i] = rotvec_to_rotmat(pose);
        rel_affine_mats[i] = make_affine(rot_mats[i], rel_joints[i]);
        if (i > 0) {
            glm::mat3 pose_feat_mat = rot_mats[i] - glm::mat3(1.0f);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    pose_feat[(i - 1) * 9 + r * 3 + c] = pose_feat_mat[c][r];
        }
    }

    affine_mats[0] = rel_affine_mats[0];
    affine_mats[1] = affine_mats[0] * rel_affine_mats[1];
    affine_mats[2] = affine_mats[1] * rel_affine_mats[2];
    affine_mats[3] = affine_mats[1] * rel_affine_mats[3];
    affine_mats[4] = affine_mats[1] * rel_affine_mats[4];

    // j output
    for (int j = 0; j < NUM_FLAME_JOINTS; j++) {
        j_output[j] = glm::vec3(affine_mats[j][3]) + translation;
    }
}

template<int N_VERTS, int N_EXPR>
__global__ void FlameExpressionForwardCuda(
    const int batch_size,
    const float* __restrict__ batched_x,
    const float* __restrict__ batched_pose_feat,
    const glm::mat4* __restrict__ batched_affine_mats,
    const glm::vec3* __restrict__ v_canonical,
    const glm::vec3* __restrict__ j_canonical,

    const glm::vec3* __restrict__ expr_dirs,
    const glm::vec3* __restrict__ pose_dirs,
    const float* __restrict__ lbs_weights,

    glm::vec3* __restrict__ batched_v_output
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (v_idx >= N_VERTS || batch_idx >= batch_size) return;

    glm::vec3 vertex = v_canonical[v_idx];
    const float* expression = batched_x + batch_idx * DIM_X;
    const glm::mat4* affine_mats = batched_affine_mats + batch_idx * NUM_FLAME_JOINTS;
    const float* pose_feat = batched_pose_feat + batch_idx * NUM_FLAME_POSEFEAT_BASIS;

    const float* translation_ptr = batched_x + batch_idx * DIM_X + (DIM_X - 3);
    glm::vec3 translation = glm::vec3(translation_ptr[0], translation_ptr[1], translation_ptr[2]);

    // compute blendshape & skinning
    for (int l = 0; l < N_EXPR; l++) {
        vertex += expression[l] * expr_dirs[l * N_VERTS + v_idx];
    }

    if (USE_POSE_BS) {
        for (int l = 0; l < NUM_FLAME_POSEFEAT_BASIS; l++) {
            vertex += pose_feat[l] * pose_dirs[l * N_VERTS + v_idx];
        }
    }

    glm::vec3 v_skinned = glm::vec3(0.0f);
    // glm::mat3 affine_skinned = glm::mat3(0.0f);

    for (int j = 0; j < NUM_FLAME_JOINTS; j++) {
        float weight = lbs_weights[v_idx * NUM_FLAME_JOINTS + j];
        glm::vec4 v_delta = glm::vec4(vertex - j_canonical[j], 1.0f);
        glm::vec4 v_transformed = affine_mats[j] * v_delta;
        v_skinned += weight * glm::vec3(v_transformed);
        // affine_skinned += weight * glm::mat3(affine_mats[j]);
    }

    batched_v_output[batch_idx * N_VERTS + v_idx] = v_skinned + translation;
}

// Identity-only — does not depend on N_EXPR.
__global__ void PrecomputeJointJacobian(
    const int batch_size,
    const glm::mat4* __restrict__ batched_rel_affine_mats,
    const glm::mat4* __restrict__ batched_affine_mats,
    const glm::vec3* __restrict__ joints_dirs,  // [300, J, 3]

    float* __restrict__ batched_j_jacobian
) {
    int batch_idx = blockIdx.x;
    int s_idx = threadIdx.x;
    if (batch_idx >= batch_size || s_idx >= NUM_FLAME_IDENTITY_BASIS) return;

    glm::vec3 joints_dir[NUM_FLAME_JOINTS];
    for (int k = 0; k < NUM_FLAME_JOINTS; k++) {
        joints_dir[k] = joints_dirs[s_idx * NUM_FLAME_JOINTS + k];
    }

    glm::vec3 ref_joints_dir[NUM_FLAME_JOINTS];
    ref_joints_dir[0] = joints_dir[0];
    ref_joints_dir[1] = joints_dir[1] - joints_dir[0];
    ref_joints_dir[2] = joints_dir[2] - joints_dir[1];
    ref_joints_dir[3] = joints_dir[3] - joints_dir[1];
    ref_joints_dir[4] = joints_dir[4] - joints_dir[1];

    glm::mat4 d_rel_affine_mats[NUM_FLAME_JOINTS];
    for (int k = 0; k < NUM_FLAME_JOINTS; k++) {
        glm::vec3 ref_joint_dir = ref_joints_dir[k];
        d_rel_affine_mats[k] = glm::mat4(0.0f);
        d_rel_affine_mats[k][3][0] = ref_joint_dir.x;
        d_rel_affine_mats[k][3][1] = ref_joint_dir.y;
        d_rel_affine_mats[k][3][2] = ref_joint_dir.z;
    }

    glm::mat4 affine_mats[NUM_FLAME_JOINTS];
    glm::mat4 rel_affine_mats[NUM_FLAME_JOINTS];
    for (int k = 0; k < NUM_FLAME_JOINTS; k++) {
        affine_mats[k] = batched_affine_mats[batch_idx * NUM_FLAME_JOINTS + k];
        rel_affine_mats[k] = batched_rel_affine_mats[batch_idx * NUM_FLAME_JOINTS + k];
    }

    glm::mat4 d_affine_mats[NUM_FLAME_JOINTS];
    d_affine_mats[0] = d_rel_affine_mats[0];
    d_affine_mats[1] = d_affine_mats[0] * rel_affine_mats[1] + affine_mats[0] * d_rel_affine_mats[1];
    d_affine_mats[2] = d_affine_mats[1] * rel_affine_mats[2] + affine_mats[1] * d_rel_affine_mats[2];
    d_affine_mats[3] = d_affine_mats[1] * rel_affine_mats[3] + affine_mats[1] * d_rel_affine_mats[3];
    d_affine_mats[4] = d_affine_mats[1] * rel_affine_mats[4] + affine_mats[1] * d_rel_affine_mats[4];

    for(int j = 0; j < 5; ++j) {
        glm::vec3 J_joint = glm::vec3(d_affine_mats[j][3]);

        // size_t to keep batch_idx * J * 300 * 3 from overflowing int32 at very large batch.
        const size_t out_offset = ((size_t)batch_idx * NUM_FLAME_JOINTS * NUM_FLAME_IDENTITY_BASIS + j * NUM_FLAME_IDENTITY_BASIS + s_idx) * 3;
        batched_j_jacobian[out_offset + 0] = J_joint.x;
        batched_j_jacobian[out_offset + 1] = J_joint.y;
        batched_j_jacobian[out_offset + 2] = J_joint.z;
    }
}

// Identity-only — does not depend on N_EXPR.
template<int N_VERTS>
__global__ void FlameIdentityJacobianCuda(
    const int batch_size,
    const glm::mat4* __restrict__ batched_affine_mats,
    const glm::vec3* __restrict__ shape_dirs,       // [300, V, 3] TODO: better memory layout
    const glm::vec3* __restrict__ joints_dirs,  // [300, J, 3]
    const float* __restrict__ lbs_weights,          // [V, J]
    const float* __restrict__ batched_j_jacobian,   // [B, J, 300, 3]

    float* __restrict__ batched_v_jacobian
) {
    int batch_idx = blockIdx.x;
    int s_idx = threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (v_idx >= N_VERTS || batch_idx >= batch_size || s_idx >= NUM_FLAME_IDENTITY_BASIS) return;

    glm::mat4 affine_mats[NUM_FLAME_JOINTS];
    for (int k = 0; k < NUM_FLAME_JOINTS; k++) {
        affine_mats[k] = batched_affine_mats[batch_idx * NUM_FLAME_JOINTS + k];
    }

    glm::vec3 joints_dir[NUM_FLAME_JOINTS];
    for (int k = 0; k < NUM_FLAME_JOINTS; k++) {
        joints_dir[k] = joints_dirs[s_idx * NUM_FLAME_JOINTS + k];
    }

    glm::vec3 shape_dir = shape_dirs[s_idx * N_VERTS + v_idx];

    glm::vec3 J = glm::vec3(0.0f);
    for (int k = 0; k < NUM_FLAME_JOINTS; k++) {
        const size_t out_offset = ((size_t)batch_idx * NUM_FLAME_JOINTS * NUM_FLAME_IDENTITY_BASIS + k * NUM_FLAME_IDENTITY_BASIS + s_idx) * 3;
        glm::vec3 d_affine_trans = glm::vec3(
            batched_j_jacobian[out_offset + 0],
            batched_j_jacobian[out_offset + 1],
            batched_j_jacobian[out_offset + 2]
        );
        float weight = lbs_weights[v_idx * NUM_FLAME_JOINTS + k];
        glm::vec3 J_K = glm::mat3(affine_mats[k]) * (shape_dir - joints_dir[k]) + d_affine_trans;
        J += weight * J_K;
    }

    // batch_idx * V * 300 * 3 can become large, so cast to size_t to
    // future-proof against larger batch sizes.
    const size_t out_offset = ((size_t)batch_idx * N_VERTS * NUM_FLAME_IDENTITY_BASIS + v_idx * NUM_FLAME_IDENTITY_BASIS + s_idx) * 3;
    batched_v_jacobian[out_offset + 0] = J.x;
    batched_v_jacobian[out_offset + 1] = J.y;
    batched_v_jacobian[out_offset + 2] = J.z;
}

}  // namespace IdentityJacobian


// ---------- host-side templated launcher + dispatch ----------

template<int N_VERTS, int N_EXPR>
static void launch_flame_identity_jacobian_impl(
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,

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
) {
    dim3 block_a(256);
    dim3 grid_a((batch_size + block_a.x - 1) / block_a.x);
    IdentityJacobian::SkeletonTreeCuda<N_EXPR><<<grid_a, block_a>>>(
        batch_size, j_canonical, batched_x,
        batched_pose_feat, batched_rel_affine_mats, batched_affine_mats, batched_j_output
    );

    dim3 block_b(1, 256);
    dim3 grid_b(batch_size, (N_VERTS + block_b.y - 1) / block_b.y);
    IdentityJacobian::FlameExpressionForwardCuda<N_VERTS, N_EXPR><<<grid_b, block_b>>>(
        batch_size,
        batched_x,
        batched_pose_feat, batched_affine_mats,
        v_canonical, j_canonical,
        expr_dirs, pose_dirs, lbs_weights,
        batched_v_output
    );

    dim3 block_c(NUM_FLAME_IDENTITY_BASIS);
    dim3 grid_c(batch_size);
    IdentityJacobian::PrecomputeJointJacobian<<<grid_c, block_c>>>(
        batch_size,
        batched_rel_affine_mats,
        batched_affine_mats,
        ref_joints_dirs,
        batched_j_jacobian
    );

    dim3 block_d(NUM_FLAME_IDENTITY_BASIS, 1);
    dim3 grid_d(batch_size, (N_VERTS + block_d.y - 1) / block_d.y);
    IdentityJacobian::FlameIdentityJacobianCuda<N_VERTS><<<grid_d, block_d>>>(
        batch_size,
        batched_affine_mats,
        shape_dirs,
        ref_joints_dirs,
        lbs_weights,
        batched_j_jacobian,
        batched_v_jacobian
    );
}

template<int N_VERTS>
static void dispatch_flame_identity_jacobian_by_expr(
    const int num_expressions,
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,
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
) {
    switch (num_expressions) {
#define DISPATCH_CASE(N) \
        case N: \
            launch_flame_identity_jacobian_impl<N_VERTS, N>( \
                batch_size, batched_x, v_canonical, j_canonical, \
                ref_joints_dirs, shape_dirs, expr_dirs, pose_dirs, lbs_weights, \
                batched_pose_feat, batched_rel_affine_mats, batched_affine_mats, \
                batched_v_output, batched_j_output, \
                batched_v_jacobian, batched_j_jacobian \
            ); \
            break;
        FLAME_FOREACH_NEXPR(DISPATCH_CASE)
#undef DISPATCH_CASE
        default:
            throw std::runtime_error("flame_identity_jacobian: unsupported num_expressions");
    }
}


void FlameTracking::flame_identity_jacobian(
    const int num_vertices,
    const int num_expressions,
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,

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
) {
#define DISPATCH_VERTS_CASE(N_VERTS) \
        case N_VERTS: \
            dispatch_flame_identity_jacobian_by_expr<N_VERTS>( \
                num_expressions, batch_size, batched_x, v_canonical, j_canonical, \
                ref_joints_dirs, shape_dirs, expr_dirs, pose_dirs, lbs_weights, \
                batched_pose_feat, batched_rel_affine_mats, batched_affine_mats, \
                batched_v_output, batched_j_output, batched_v_jacobian, batched_j_jacobian \
            ); \
            break;
    switch (num_vertices) {
        FLAME_FOREACH_NVERTS(DISPATCH_VERTS_CASE)
        default:
            throw std::runtime_error("flame_identity_jacobian: unsupported num_vertices");
    }
#undef DISPATCH_VERTS_CASE
}
