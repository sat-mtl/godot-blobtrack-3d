#[compute]
#version 450

layout(local_size_x = 1) in;

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

layout(set = 0, binding = 11, std430) buffer NumPoints {
  int num;
} num_points;

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
  grid_points.x = grid_for(num_points.num, points_local_size);
  grid_points.y = 1;
  grid_points.z = 1;

  grid_merge.x = grid_for(num_points.num, merge_local_size);
  grid_merge.y = 1;
  grid_merge.z = 1;
}
