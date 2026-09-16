#[compute]
#version 450
#include "prelude.glsl.inc"
// Stage 3: fold each block's offset into its slice of the scan. Zeroing the
// histogram here too turns it into the scatter cursor without a separate clear.
layout(local_size_x = 256) in;

layout(std430, binding = 2)          buffer CountBuf { uint cellCount[]; };
layout(std430, binding = 3)          buffer StartBuf { uint cellStart[]; };
layout(std430, binding = 4) readonly buffer BlockBuf { uint blockSums[]; };

layout(push_constant) uniform Parameters {
  int u_tableSize;
} params;

void main()
{
	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint i = gl_GlobalInvocationID.x; i < params.u_tableSize; i += stride)
	{
		cellStart[i] += blockSums[i / 1024u];
		cellCount[i] = 0u;
	}
}
