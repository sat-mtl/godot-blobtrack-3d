#[compute]
#version 450

layout(local_size_x = 1) in;

// passes over the sorted points (flatten, propagate, accumulate)
layout(set = 0, binding = 0, std430) buffer GridPoints {
  int x;
  int y;
  int z;
} grid_points;

layout(set = 0, binding = 1, std430) buffer GridMerge {
  int x;
  int y;
  int z;
} grid_merge;

// passes over the input segments (bbox/histogram, scatter): one row of
// workgroups per segment, sized for the largest one.
layout(set = 0, binding = 2, std430) buffer GridInput {
  int x;
  int y;
  int z;
} grid_input;

layout(set = 0, binding = 11, std430) readonly buffer SegmentSizes {
  uint sizes[];
} segment_sizes;

layout(set = 0, binding = 12, std430) readonly buffer SegmentIds {
  uint ids[];
} segment_ids;

layout(push_constant) uniform Parameters {
  int u_numSegments;
} params;

const int max_workgroup_idx = 65535;
const int points_local_size = 256;
const int merge_local_size = 128;

int grid_for(int n, int local_size) {
  if (n <= 0) {
    return 1;
  }
  int groups = (n + local_size - 1) / local_size;
  return min(groups, max_workgroup_idx);
}

void main() {
  int total = 0;
  int largest = 0;
  for (int k = 0; k < params.u_numSegments; k++) {
    int size = int(segment_sizes.sizes[segment_ids.ids[k]]);
    total += size;
    largest = max(largest, size);
  }

  grid_input.x = grid_for(largest, points_local_size);
  grid_input.y = min(params.u_numSegments, max_workgroup_idx);
  grid_input.z = 1;

  grid_points.x = grid_for(total, points_local_size);
  grid_points.y = 1;
  grid_points.z = 1;

  grid_merge.x = grid_for(total, merge_local_size);
  grid_merge.y = 1;
  grid_merge.z = 1;
}
