#[compute]
#version 450
// Bounding box reduction fused with the cell histogram: both want one pass over
// the raw positions, so they share it.
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf   { float positions[]; };
layout(std430, binding = 2) coherent buffer CountBuf { uint  cellCount[]; };
layout(std430, binding = 8) coherent buffer MetaBuf  { int   meta[]; };

layout(push_constant) uniform Parameters {
  uint u_tableSize;
  float u_invCellSize;
} params;

layout(set = 0, binding = 11, std430) buffer NumPoints {
  int num;
} num_points;

shared float sMin[3][256];
shared float sMax[3][256];

void atomicMinF(uint idx, float v)
{
	int e = meta[idx];
	for (;;)
	{
		int d = floatBitsToInt(min(v, intBitsToFloat(e)));
		int o = atomicCompSwap(meta[idx], e, d);
		if (o == e) return;
		e = o;
	}
}

void atomicMaxF(uint idx, float v)
{
	int e = meta[idx];
	for (;;)
	{
		int d = floatBitsToInt(max(v, intBitsToFloat(e)));
		int o = atomicCompSwap(meta[idx], e, d);
		if (o == e) return;
		e = o;
	}
}

void main()
{
	uint tid = gl_LocalInvocationID.x;
	vec3 lo = vec3( 3.402823466e+38);
	vec3 hi = vec3(-3.402823466e+38);

	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint i = gl_GlobalInvocationID.x; i < uint(num_points.num); i += stride)
	{
		vec3 p = vec3(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
		if (!posValid(p)) continue;

		lo = min(lo, p);
		hi = max(hi, p);
		atomicAdd(cellCount[bucketOf(cellOf(p, params.u_invCellSize), params.u_tableSize)], 1u);
	}

	sMin[0][tid] = lo.x; sMin[1][tid] = lo.y; sMin[2][tid] = lo.z;
	sMax[0][tid] = hi.x; sMax[1][tid] = hi.y; sMax[2][tid] = hi.z;
	barrier();

	for (uint s = 128u; s > 0u; s >>= 1u)
	{
		if (tid < s)
		{
			for (int a = 0; a < 3; a++)
			{
				sMin[a][tid] = min(sMin[a][tid], sMin[a][tid + s]);
				sMax[a][tid] = max(sMax[a][tid], sMax[a][tid + s]);
			}
		}
		barrier();
	}

	if (tid == 0u)
	{
		atomicMinF(0u, sMin[0][0]); atomicMinF(1u, sMin[1][0]); atomicMinF(2u, sMin[2][0]);
		atomicMaxF(3u, sMax[0][0]); atomicMaxF(4u, sMax[1][0]); atomicMaxF(5u, sMax[2][0]);
	}
}
