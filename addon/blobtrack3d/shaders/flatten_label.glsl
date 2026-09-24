#[compute]
#version 450
// Collapses every path to its root and hands each root a compact id. Only the
// root itself allocates, so no two invocations can race for one cluster's id.
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 5) coherent buffer ParentBuf  { int parent[]; };
layout(std430, binding = 6)          buffer ClusterBuf { int clusterId[]; };
layout(std430, binding = 8) coherent buffer MetaBuf    { int meta[]; };

layout(push_constant) uniform Parameters {
  int u_maxClusters;
} params;

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

		int r = i;
		while (parent[r] != r)
		{
			int p  = parent[r];
			int gp = parent[p];
			parent[r] = gp;
			r = gp;
		}
		parent[i] = r;

		if (r != i) continue;

		int id = atomicAdd(meta[13], 1);
		if (id < params.u_maxClusters)
		{
			clusterId[i] = id;
		}
		else
		{
			clusterId[i] = -1;
			meta[15] = 1;
		}
	}
}
