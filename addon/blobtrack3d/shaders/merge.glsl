#[compute]
#version 450
// y'avait pas de commentaire par dessus celui là, who knows ?
#include "prelude.glsl.inc"
layout(local_size_x = 128) in;

layout(std430, binding = 1) readonly buffer SortedBuf { float posSorted[]; };
layout(std430, binding = 3) readonly buffer StartBuf  { uint  cellStart[]; };
layout(std430, binding = 5) coherent buffer ParentBuf { int   parent[]; };
layout(std430, binding = 8) readonly buffer MetaBuf   { int   meta[]; };

layout(push_constant) uniform Parameters {
  uint u_tableSize;
  float u_invCellSize;
  float u_clusterDist2;
} params;

layout(set = 0, binding = 11, std430) buffer NumPoints {
  int num;
} num_points;

int ufFind(int x)
{
	int r = x;
	while (parent[r] != r)
	{
		int p  = parent[r];
		int gp = parent[p];
		atomicCompSwap(parent[r], p, gp);
		r = gp;
	}
	return r;
}

// Always reparents the higher index onto the lower one, so parent indices
// strictly decrease along any path and ufFind cannot cycle.
void ufUnite(int a, int b)
{
	for (;;)
	{
		a = ufFind(a);
		b = ufFind(b);
		if (a == b) return;
		if (a > b) { int t = a; a = b; b = t; }
		if (atomicCompSwap(parent[b], b, a) == b) return;
	}
}

void main()
{
	int nValid = meta[12];

	uint stride = gl_NumWorkGroups.x * 128u;
	for (uint idx = gl_GlobalInvocationID.x; idx < uint(num_points.num); idx += stride)
	{
		int i = int(idx);
		if (i >= nValid) continue;

		vec3  p = vec3(posSorted[idx * 3u], posSorted[idx * 3u + 1u], posSorted[idx * 3u + 2u]);
		ivec3 c = cellOf(p, params.u_invCellSize);

		for (int dx = -1; dx <= 1; dx++)
		for (int dy = -1; dy <= 1; dy++)
		for (int dz = -1; dz <= 1; dz++)
		{
			uint b     = bucketOf(c + ivec3(dx, dy, dz), params.u_tableSize);
			uint first = cellStart[b];
			uint last  = cellStart[b + 1u];

			for (uint j = first; j < last; j++)
			{
				// Each unordered pair is tested once. Cell adjacency is
				// symmetric, so the higher index always sees the lower one.
				if (int(j) <= i) continue;

				vec3 q = vec3(posSorted[j * 3u], posSorted[j * 3u + 1u], posSorted[j * 3u + 2u]);
				vec3 d = p - q;
				if (dot(d, d) < params.u_clusterDist2)
					ufUnite(i, int(j));
			}
		}
	}
}
