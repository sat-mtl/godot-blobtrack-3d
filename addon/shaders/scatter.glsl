#[compute]
#version 450
// Reorders points into cell order. Everything downstream reads posSorted, so
// the neighbour walk touches contiguous memory instead of chasing a linked
// list. Union-find roots are seeded here too, saving a dispatch.
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf    { float positions[]; };
layout(std430, binding = 1)          buffer SortedBuf { float posSorted[]; };
layout(std430, binding = 2) coherent buffer CursorBuf { uint  cursor[]; };
layout(std430, binding = 3) readonly buffer StartBuf  { uint  cellStart[]; };
layout(std430, binding = 5)          buffer ParentBuf { int   parent[]; };

layout(push_constant) uniform Parameters {
  // number of points
  int u_n;
  uint u_tableSize;
  float u_invCellSize;
} params;


void main()
{
	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint i = gl_GlobalInvocationID.x; i < uint(params.u_n); i += stride)
	{
		vec3 p = vec3(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
		if (!posValid(p)) continue;

		uint b    = bucketOf(cellOf(p, params.u_invCellSize), params.u_tableSize);
		uint slot = cellStart[b] + atomicAdd(cursor[b], 1u);

		posSorted[slot * 3u]      = p.x;
		posSorted[slot * 3u + 1u] = p.y;
		posSorted[slot * 3u + 2u] = p.z;
		parent[slot]              = int(slot);
	}
}
