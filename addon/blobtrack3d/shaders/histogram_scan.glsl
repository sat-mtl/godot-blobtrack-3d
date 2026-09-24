#[compute]
#version 450
// Exclusive scan of the histogram, stage 1: scan within each block of 1024 and
// publish the block total. The table size is a power of two of at least 1024,
// so blocks are always full and need no bounds checks.
layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer CountBuf { uint cellCount[]; };
layout(std430, binding = 3)          buffer StartBuf { uint cellStart[]; };
layout(std430, binding = 4)          buffer BlockBuf { uint blockSums[]; };

shared uint sPart[256];

void main()
{
	uint tid  = gl_LocalInvocationID.x;
	uint base = gl_WorkGroupID.x * 1024u + tid * 4u;

	uvec4 v = uvec4(cellCount[base], cellCount[base + 1u],
					cellCount[base + 2u], cellCount[base + 3u]);

	uint s0 = 0u;
	uint s1 = v.x;
	uint s2 = s1 + v.y;
	uint s3 = s2 + v.z;
	uint total = s3 + v.w;

	sPart[tid] = total;
	barrier();

	for (uint off = 1u; off < 256u; off <<= 1u)
	{
		uint add = (tid >= off) ? sPart[tid - off] : 0u;
		barrier();
		sPart[tid] += add;
		barrier();
	}

	uint excl = sPart[tid] - total;
	cellStart[base]      = excl + s0;
	cellStart[base + 1u] = excl + s1;
	cellStart[base + 2u] = excl + s2;
	cellStart[base + 3u] = excl + s3;

	if (tid == 255u)
		blockSums[gl_WorkGroupID.x] = sPart[255u];
}
