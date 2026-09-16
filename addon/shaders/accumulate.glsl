#[compute]
#version 450
// The accumulator is struct-of-arrays over maxClusters M so the host can reset
// it with three buffer clears instead of a dispatch:
//   [0, 3M)   quantised centroid sums, axis major, cleared to 0
//   [3M, 4M)  point counts, cleared to 0
//   [4M, 7M)  bbox minima as float bits, cleared to +FLT_MAX
//   [7M, 10M) bbox maxima as float bits, cleared to -FLT_MAX
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 1) readonly buffer SortedBuf  { float posSorted[]; };
layout(std430, binding = 6) readonly buffer ClusterBuf { int   clusterId[]; };
layout(std430, binding = 7) coherent buffer AccumBuf   { int   accum[]; };
layout(std430, binding = 8) readonly buffer MetaBuf    { int   meta[]; };

layout(push_constant) uniform Parameters {
  // number of points
  int u_n;
  int u_maxClusters;
} params;

void atomicMinF(uint idx, float v)
{
	int e = accum[idx];
	for (;;)
	{
		int d = floatBitsToInt(min(v, intBitsToFloat(e)));
		int o = atomicCompSwap(accum[idx], e, d);
		if (o == e) return;
		e = o;
	}
}

void atomicMaxF(uint idx, float v)
{
	int e = accum[idx];
	for (;;)
	{
		int d = floatBitsToInt(max(v, intBitsToFloat(e)));
		int o = atomicCompSwap(accum[idx], e, d);
		if (o == e) return;
		e = o;
	}
}

void main()
{
	int  nValid = meta[12];
	uint M      = uint(params.u_maxClusters);

	vec3 lo    = vec3(intBitsToFloat(meta[6]), intBitsToFloat(meta[7]),  intBitsToFloat(meta[8]));
	vec3 scale = vec3(intBitsToFloat(meta[9]), intBitsToFloat(meta[10]), intBitsToFloat(meta[11]));

	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint i = gl_GlobalInvocationID.x; i < uint(params.u_n); i += stride)
	{
		if (int(i) >= nValid) continue;

		int cid = clusterId[i];
		if (cid < 0) continue;

		vec3  p = vec3(posSorted[i * 3u], posSorted[i * 3u + 1u], posSorted[i * 3u + 2u]);
		ivec3 q = ivec3(clamp((p - lo) * scale, vec3(0.0), vec3(2.1e9)));
		uint  c = uint(cid);

		atomicAdd(accum[c],          q.x);
		atomicAdd(accum[M + c],      q.y);
		atomicAdd(accum[2u * M + c], q.z);
		atomicAdd(accum[3u * M + c], 1);

		atomicMinF(4u * M + c, p.x);
		atomicMinF(5u * M + c, p.y);
		atomicMinF(6u * M + c, p.z);
		atomicMaxF(7u * M + c, p.x);
		atomicMaxF(8u * M + c, p.y);
		atomicMaxF(9u * M + c, p.z);
	}
}
