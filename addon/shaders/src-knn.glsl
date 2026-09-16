#[compute]
#version 450
// Distance to the k-th nearest neighbour for a subsample, which the host turns
// into a cluster distance by taking a percentile.
//
// Candidates come from the sample's own 27-cell neighbourhood in the grid the
// pipeline has already built, so every neighbour that could be among the
// nearest k is actually examined. Subsampling the candidates instead would
// inflate the measured spacing by a factor that depends on how many points were
// skipped, and silently rescale auto-scale with the size of the cloud.
//
// Points are already in cell order, so striding the sorted index is a spatially
// even sample of the cloud.
#include "prelude.glsl.inc"
layout(local_size_x = 256) in;

layout(std430, binding = 1)  readonly buffer SortedBuf { float posSorted[]; };
layout(std430, binding = 3)  readonly buffer StartBuf  { uint  cellStart[]; };
layout(std430, binding = 8)  readonly buffer MetaBuf   { int   meta[]; };
layout(std430, binding = 10)          buffer KnnBuf    { float knnDist[]; };

layout(push_constant) uniform Parameters {
  // number of points
  int u_k;
  float u_invCellSize;
  int u_sampleCount;
  uint u_tableSize;
} params;

// Only reached while the cluster distance is far too large for the cloud, which
// lasts a frame or two before the estimate settles.
const int MAX_CANDIDATES = 2048;

void main()
{
	uint tid = gl_GlobalInvocationID.x;
	if (tid >= uint(params.u_sampleCount)) return;

	knnDist[tid] = -1.0;

	int nValid = meta[12];
	if (nValid <= params.u_k) return;

	uint idx = tid * uint(nValid) / uint(params.u_sampleCount);
        // if you go over, go to the last valid index
        if (idx >= uint(nValid)) idx = uint(nValid) - 1u;

	vec3  p = vec3(posSorted[idx * 3u], posSorted[idx * 3u + 1u], posSorted[idx * 3u + 2u]);
	ivec3 c = cellOf(p, params.u_invCellSize);

	float topK[8];
	for (int t = 0; t < 8; t++) topK[t] = 3.402823466e+38;

	int found   = 0;
	int scanned = 0;

	for (int n = 0; n < 27 && scanned < MAX_CANDIDATES; n++)
	{
		// Visit the sample's own cell first, so the nearest neighbours are seen
		// before the candidate cap can bite.
		int m = (n == 0) ? 13 : ((n <= 13) ? n - 1 : n);
		ivec3 d = ivec3(m % 3 - 1, (m / 3) % 3 - 1, m / 9 - 1);

		uint b     = bucketOf(c + d, params.u_tableSize);
		uint first = cellStart[b];
		uint last  = cellStart[b + 1u];

		for (uint j = first; j < last && scanned < MAX_CANDIDATES; j++)
		{
			if (j == idx) continue;
			scanned++;
			found++;

			vec3  q  = vec3(posSorted[j * 3u], posSorted[j * 3u + 1u], posSorted[j * 3u + 2u]);
			vec3  e  = q - p;
			float d2 = dot(e, e);
			if (d2 >= topK[params.u_k - 1]) continue;

			topK[params.u_k - 1] = d2;
			for (int t = params.u_k - 2; t >= 0; t--)
			{
				if (topK[t + 1] >= topK[t]) break;
				float tmp = topK[t]; topK[t] = topK[t + 1]; topK[t + 1] = tmp;
			}
		}
	}

	// Fewer than k neighbours within reach means the cell is smaller than the
	// real spacing. Reporting the cell size grows the estimate, so the next
	// frame searches wider and the loop converges upward instead of stalling.
	knnDist[tid] = (found >= params.u_k) ? sqrt(topK[params.u_k - 1]) : (1.0 / params.u_invCellSize);
}
