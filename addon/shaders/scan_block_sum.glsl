#[compute]
#version 450
// Stage 2: one workgroup scans the block totals in place. This is also the
// first point where the bounding box and the valid-point count are both final,
// so it derives the fixed-point accumulation frame here rather than paying for
// another dispatch.
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 3)          buffer StartBuf { uint cellStart[]; };
layout(std430, binding = 4)          buffer BlockBuf { uint blockSums[]; };
layout(std430, binding = 8) coherent buffer MetaBuf  { int  meta[]; };

layout(push_constant) uniform Parameters {
  int u_numBlocks;
  int u_tableSize;
} params;


shared uint sPart[256];
shared uint sRunning;

void main()
{
	uint tid = gl_LocalInvocationID.x;

	if (tid == 0u) sRunning = 0u;
	barrier();

	for (uint tile = 0u; tile < params.u_numBlocks; tile += 256u)
	{
		uint idx = tile + tid;
		uint v   = (idx < params.u_numBlocks) ? blockSums[idx] : 0u;

		sPart[tid] = v;
		barrier();

		for (uint off = 1u; off < 256u; off <<= 1u)
		{
			uint add = (tid >= off) ? sPart[tid - off] : 0u;
			barrier();
			sPart[tid] += add;
			barrier();
		}

		uint run = sRunning;
		if (idx < params.u_numBlocks)
			blockSums[idx] = run + sPart[tid] - v;
		barrier();

		if (tid == 255u) sRunning = run + sPart[255u];
		barrier();
	}

	if (tid != 0u) return;

	uint total = sRunning;
	cellStart[params.u_tableSize] = total;
	meta[12] = int(total);

	vec3 lo  = vec3(intBitsToFloat(meta[0]), intBitsToFloat(meta[1]), intBitsToFloat(meta[2]));
	vec3 hi  = vec3(intBitsToFloat(meta[3]), intBitsToFloat(meta[4]), intBitsToFloat(meta[5]));
	vec3 ext = max(hi - lo, vec3(0.0));

	// Centroids accumulate as int32 sums of offsets from lo, quantised by
	// scale. The worst case is every valid point joining one cluster, so
	// capping the per-point quantum at 2e9 / count makes overflow impossible.
	// The rounding error is uniform and averages out across the cluster.
	float budget = 2.0e9 / max(float(total), 1.0);
	vec3  scale  = vec3(ext.x > 1.0e-20 ? budget / ext.x : 1.0,
						ext.y > 1.0e-20 ? budget / ext.y : 1.0,
						ext.z > 1.0e-20 ? budget / ext.z : 1.0);

	meta[6]  = floatBitsToInt(lo.x);
	meta[7]  = floatBitsToInt(lo.y);
	meta[8]  = floatBitsToInt(lo.z);
	meta[9]  = floatBitsToInt(scale.x);
	meta[10] = floatBitsToInt(scale.y);
	meta[11] = floatBitsToInt(scale.z);
}
