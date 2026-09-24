#[compute]
#version 450
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 5) readonly buffer ParentBuf  { int parent[]; };
layout(std430, binding = 6)          buffer ClusterBuf { int clusterId[]; };
layout(std430, binding = 8) readonly buffer MetaBuf    { int meta[]; };

layout(set = 0, binding = 11, std430) buffer NumPoints {
  int num;
} num_points;

void main()
{
	int nValid = meta[12];

	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint idx = gl_GlobalInvocationID.x; idx < uint(num_points.num); idx += stride)
	{
		int i = int(idx);
		if (i >= nValid) continue;
		clusterId[i] = clusterId[parent[i]];
	}
}
