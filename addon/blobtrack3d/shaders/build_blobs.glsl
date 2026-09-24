#[compute]
#version 450
#include "prelude.glsl.inc"

layout(local_size_x = 256) in;

layout(std430, binding = 7) readonly buffer AccumBuf { int accum[]; };
layout(std430, binding = 8) coherent buffer MetaBuf  { int meta[]; };

struct BlobDet
{
	float cx, cy, cz;
	float bmnx, bmny, bmnz;
	float bmxx, bmxy, bmxz;
	int   pointCount;
};
layout(std430, binding = 9) buffer BlobBuf { BlobDet blobs[]; };

layout(push_constant) uniform Parameters {
  int u_minPoints;
  int u_maxClusters;
  int u_maxBlobs;
} params;

void main()
{
	uint M = uint(params.u_maxClusters);

	vec3 lo    = vec3(intBitsToFloat(meta[6]), intBitsToFloat(meta[7]),  intBitsToFloat(meta[8]));
	vec3 scale = vec3(intBitsToFloat(meta[9]), intBitsToFloat(meta[10]), intBitsToFloat(meta[11]));

	uint stride = gl_NumWorkGroups.x * 256u;
	for (uint i = gl_GlobalInvocationID.x; i < M; i += stride)
	{
		int count = accum[3u * M + i];
		if (count < params.u_minPoints) continue;

		int idx = atomicAdd(meta[14], 1);
		if (idx >= params.u_maxBlobs) continue;

		vec3 sums = vec3(float(accum[i]), float(accum[M + i]), float(accum[2u * M + i]));
		vec3 c    = lo + (sums / float(count)) / scale;

		blobs[idx].cx = c.x;
		blobs[idx].cy = c.y;
		blobs[idx].cz = c.z;

		blobs[idx].bmnx = intBitsToFloat(accum[4u * M + i]);
		blobs[idx].bmny = intBitsToFloat(accum[5u * M + i]);
		blobs[idx].bmnz = intBitsToFloat(accum[6u * M + i]);
		blobs[idx].bmxx = intBitsToFloat(accum[7u * M + i]);
		blobs[idx].bmxy = intBitsToFloat(accum[8u * M + i]);
		blobs[idx].bmxz = intBitsToFloat(accum[9u * M + i]);

		blobs[idx].pointCount = count;
	}
}
