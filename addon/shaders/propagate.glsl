#[compute]
#version 450
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 5) readonly buffer ParentBuf  { int parent[]; };
layout(std430, binding = 6)          buffer ClusterBuf { int clusterId[]; };
layout(std430, binding = 8) readonly buffer MetaBuf    { int meta[]; };

layout(push_constant) uniform Parameters {
  // number of points
  int u_n;
} params;

void main()
{
	int nValid = meta[12];

	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint idx = gl_GlobalInvocationID.x; idx < uint(params.u_n); idx += stride)
	{
		int i = int(idx);
		if (i >= nValid) continue;
		clusterId[i] = clusterId[parent[i]];
	}
}
