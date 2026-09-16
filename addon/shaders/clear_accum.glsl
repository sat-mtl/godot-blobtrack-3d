#[compute]
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 7, std430) buffer AccumBuffer {
  float data[];
} accum_buffer;

layout(push_constant) uniform Parameters {
  // should be the number of floats in the buffer.
  // should be 10* max cluster
  int num_floats;
  int max_clusters;
} params;

const uint l_size_x = 64;
const uint max_workgroup_idx = 65535;
const uint max_x_idx = max_workgroup_idx * l_size_x;

void main() {
  uint global_idx =
      uint((gl_WorkGroupID.x + gl_LocalInvocationID.x + ((l_size_x - 1) * gl_WorkGroupID.x)) +
       gl_WorkGroupID.y * max_x_idx +
       gl_WorkGroupID.z * max_x_idx * max_workgroup_idx);
  if (global_idx < params.num_floats) {
    if (global_idx < params.max_clusters * 4) {
      accum_buffer.data[global_idx] = 0.0;
    }
    else if (global_idx < params.max_clusters * 7) {
      accum_buffer.data[global_idx] = uintBitsToFloat(0x7F7FFFFFu);
    } else {
      accum_buffer.data[global_idx] = uintBitsToFloat(0xFF7FFFFFu);
    }
  }
}
