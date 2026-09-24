#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include "godot_cpp/variant/vector3.hpp"
#include "godot_cpp/variant/aabb.hpp"
#include "godot_cpp/classes/ref_counted.hpp"

constexpr int blob_data_size = 10 * 4;
using namespace godot;

class Blob {

 public:
  int32_t id                  = -1;
  float   cx = 0, cy = 0, cz = 0;
  float   bmnx = 0, bmny = 0, bmnz = 0;
  float   bmxx = 0, bmxy = 0, bmxz = 0;
  int32_t pointCount          = 0;
  int32_t framesSinceLastSeen = 0;
  float   vx = 0, vy = 0, vz = 0;

  void populate_from_bytes(PackedByteArray& blob_bytes, int blob_num) {
    int current_offset = blob_num*blob_data_size;
    cx = blob_bytes.decode_float(current_offset);
    cy = blob_bytes.decode_float(current_offset + 4);
    cz = blob_bytes.decode_float(current_offset + 8);
    current_offset += 12;
    bmnx = blob_bytes.decode_float(current_offset);
    bmny = blob_bytes.decode_float(current_offset + 4);
    bmnz = blob_bytes.decode_float(current_offset + 8);
    current_offset += 12;
    bmxx = blob_bytes.decode_float(current_offset);
    bmxy = blob_bytes.decode_float(current_offset + 4);
    bmxz = blob_bytes.decode_float(current_offset + 8);
    current_offset += 12;

    pointCount = blob_bytes.decode_u32(current_offset);
  }
  float radius() const
  {
    return std::max(std::max(bmxx - bmnx, bmxy - bmny), bmxz - bmnz) * 0.5f;
  }

  float volume() const
  {
    return (bmxx - bmnx) * (bmxy - bmny) * (bmxz - bmnz);
  }
};


class GodotBlob : public RefCounted {
 GDCLASS(GodotBlob, RefCounted)
 protected:
  static void _bind_methods() {
    godot::ClassDB::bind_method(D_METHOD("set_centroid", "p_centroid"), &GodotBlob::set_centroid);
    godot::ClassDB::bind_method(D_METHOD("get_centroid"), &GodotBlob::get_centroid);
    godot::ClassDB::bind_method(D_METHOD("set_bounding_box", "p_bounding_box"), &GodotBlob::set_bounding_box);
    godot::ClassDB::bind_method(D_METHOD("get_bounding_box"), &GodotBlob::get_bounding_box);
    godot::ClassDB::bind_method(D_METHOD("set_blob_id", "p_blob_id"), &GodotBlob::set_blob_id);
    godot::ClassDB::bind_method(D_METHOD("get_blob_id"), &GodotBlob::get_blob_id);
    godot::ClassDB::bind_method(D_METHOD("set_velocity", "p_velocity"), &GodotBlob::set_velocity);
    godot::ClassDB::bind_method(D_METHOD("get_velocity"), &GodotBlob::get_velocity);

    ADD_PROPERTY(PropertyInfo(Variant::AABB, "bounding_box"), "set_bounding_box", "get_bounding_box");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "centroid"), "set_centroid", "get_centroid");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "blob_id"), "set_blob_id", "get_blob_id");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "velocity"), "set_velocity", "get_velocity");
  }
 public:

  int id = -1;
  void set_bounding_box(AABB aabb) {
    // should not manually set bounding box.
  }
  void set_from_blob(const Blob& blob) {
    bmin.x = blob.bmnx;
    bmin.y = blob.bmny;
    bmin.z = blob.bmnz;
    bmax.x = blob.bmxx;
    bmax.y = blob.bmxy;
    bmax.z = blob.bmxz;
    velocity.x = blob.vx;
    velocity.y = blob.vy;
    velocity.z = blob.vz;
    centroid.x = blob.cx;
    centroid.y = blob.cy;
    centroid.z = blob.cz;
    id = blob.id;
    bounding_box.set_position(bmin);
    bounding_box.set_size(bmax - bmin);
  }
  const AABB& get_bounding_box() {
    // TODO: change all the blob class to use godot's Vector3 internally instead of
    // updating stuff here
    return bounding_box;
  }
  void set_centroid(Vector3 cent){
    // should not manually set either
    centroid = cent;
  }
  const Vector3& get_centroid(){
    return centroid;
  }
  void set_velocity(Vector3 vel){
    velocity = vel;
  }
  const Vector3& get_velocity(){
    return velocity;
  }
  void set_blob_id(int the_id) {
    id = the_id;
  }
  int get_blob_id() {
    return id;
  }
 private:
  AABB bounding_box;
  Vector3 centroid;
  Vector3 bmin;
  Vector3 bmax;
  Vector3 velocity;

};

inline float blobDist(const Blob& a, const Blob& b)
{
  const float dx = a.cx - b.cx;
  const float dy = a.cy - b.cy;
  const float dz = a.cz - b.cz;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Centroid distance discounted by box overlap, so a detection that overlaps a
// prediction is preferred over an equally distant one that does not. Optionally
// penalises a mismatch in point count, which separates two people walking past
// each other when their boxes overlap but their densities differ.
inline float blobMatchCost(const Blob& det, const Blob& pred,
                           bool usePointCount, float pointCountWeight)
{
  const float dist = blobDist(det, pred);

  const float ix = std::max(0.0f, std::min(det.bmxx, pred.bmxx) - std::max(det.bmnx, pred.bmnx));
  const float iy = std::max(0.0f, std::min(det.bmxy, pred.bmxy) - std::max(det.bmny, pred.bmny));
  const float iz = std::max(0.0f, std::min(det.bmxz, pred.bmxz) - std::max(det.bmnz, pred.bmnz));

  const float inter    = ix * iy * iz;
  const float unionVol = det.volume() + pred.volume() - inter;
  const float iou      = (unionVol > 1e-10f) ? (inter / unionVol) : 0.0f;

  float cost = dist * (1.0f - iou * 0.5f);

  if (usePointCount && pointCountWeight > 0.0f &&
      det.pointCount > 0 && pred.pointCount > 0)
  {
    const float ratio = (float)std::min(det.pointCount, pred.pointCount) /
        (float)std::max(det.pointCount, pred.pointCount);
    cost += (1.0f - ratio) * pointCountWeight * dist;
  }

  return cost;
}

// Scalar Kalman filter on [position, velocity] with a constant-velocity model
// and a unit time step, so predict() folds down to x += v.
struct KalmanFilter1D
{
  float x[2];
  float P[2][2];
  float Q[2][2];
  float R;

  void init(float pos, float procNoise, float measNoise)
  {
    x[0] = pos;   x[1] = 0.0f;
    P[0][0] = 10.0f; P[0][1] = 0.0f;
    P[1][0] = 0.0f;  P[1][1] = 1000.0f;
    Q[0][0] = procNoise; Q[0][1] = 0.0f;
    Q[1][0] = 0.0f;      Q[1][1] = procNoise * 0.1f;
    R = measNoise;
  }

  float predict()
  {
    x[0] += x[1];

    const float p00 = P[0][0] + P[1][0] + P[0][1] + P[1][1] + Q[0][0];
    const float p01 = P[0][1] + P[1][1] + Q[0][1];
    const float p10 = P[1][0] + P[1][1] + Q[1][0];
    const float p11 = P[1][1] + Q[1][1];

    P[0][0] = p00; P[0][1] = p01;
    P[1][0] = p10; P[1][1] = p11;
    return x[0];
  }

  void update(float z)
  {
    const float y = z - x[0];
    float S = P[0][0] + R;
    if (std::fabs(S) < 1e-12f)
      S = 1e-12f;

    const float K0 = P[0][0] / S;
    const float K1 = P[1][0] / S;

    x[0] += K0 * y;
    x[1] += K1 * y;

    const float p00 = (1.0f - K0) * P[0][0];
    const float p01 = (1.0f - K0) * P[0][1];
    const float p10 = P[1][0] - K1 * P[0][0];
    const float p11 = P[1][1] - K1 * P[0][1];

    P[0][0] = p00; P[0][1] = p01;
    P[1][0] = p10; P[1][1] = p11;
  }
};

// Six independent scalar filters: centroid xyz and box half-extents xyz. The
// axes are treated as uncorrelated, which costs nothing in accuracy here and
// keeps the update free of matrix inversions.
struct KalmanTracker3D
{
  KalmanFilter1D kfCx, kfCy, kfCz;
  KalmanFilter1D kfHx, kfHy, kfHz;

  int32_t id              = -1;
  int32_t timeSinceUpdate = 0;
  int32_t hits            = 0;
  int32_t hitStreak       = 0;
  int32_t age             = 0;
  int32_t lastPointCount  = 0;

  void init(const Blob& det, int32_t trackID)
  {
    kfCx.init(det.cx, 0.01f, 1.0f);
    kfCy.init(det.cy, 0.01f, 1.0f);
    kfCz.init(det.cz, 0.01f, 1.0f);
    kfHx.init((det.bmxx - det.bmnx) * 0.5f, 0.005f, 5.0f);
    kfHy.init((det.bmxy - det.bmny) * 0.5f, 0.005f, 5.0f);
    kfHz.init((det.bmxz - det.bmnz) * 0.5f, 0.005f, 5.0f);

    id              = trackID;
    timeSinceUpdate = 0;
    hits            = 1;
    hitStreak       = 1;
    age             = 1;
    lastPointCount  = det.pointCount;
  }

  Blob predict()
  {
    const float cx = kfCx.predict();
    const float cy = kfCy.predict();
    const float cz = kfCz.predict();
    const float hx = std::max(kfHx.predict(), 0.001f);
    const float hy = std::max(kfHy.predict(), 0.001f);
    const float hz = std::max(kfHz.predict(), 0.001f);

    age++;
    if (timeSinceUpdate > 0)
      hitStreak = 0;
    timeSinceUpdate++;

    return box(cx, cy, cz, hx, hy, hz);
  }

  void update(const Blob& det)
  {
    timeSinceUpdate = 0;
    hits++;
    hitStreak++;
    lastPointCount = det.pointCount;

    kfCx.update(det.cx);
    kfCy.update(det.cy);
    kfCz.update(det.cz);
    kfHx.update((det.bmxx - det.bmnx) * 0.5f);
    kfHy.update((det.bmxy - det.bmny) * 0.5f);
    kfHz.update((det.bmxz - det.bmnz) * 0.5f);
  }
  Blob state;
  const Blob& getState()
  {
    return box(kfCx.x[0], kfCy.x[0], kfCz.x[0],
               std::max(kfHx.x[0], 0.001f),
               std::max(kfHy.x[0], 0.001f),
               std::max(kfHz.x[0], 0.001f));
  }

 private:
  const Blob& box(float cx, float cy, float cz, float hx, float hy, float hz)
  {
    state.id = id;
    state.cx = cx; state.cy = cy; state.cz = cz;
    state.bmnx = cx - hx; state.bmny = cy - hy; state.bmnz = cz - hz;
    state.bmxx = cx + hx; state.bmxy = cy + hy; state.bmxz = cz + hz;
    state.vx = kfCx.x[1]; state.vy = kfCy.x[1]; state.vz = kfCz.x[1];
    state.pointCount = lastPointCount;
    state.framesSinceLastSeen = timeSinceUpdate;
    return state;
  }
};

// Remembers a track that was absorbed into another so its id can be handed back
// when the pair separates again.
struct MergeMemory
{
  int32_t storedID       = -1;
  float   lastCx = 0, lastCy = 0, lastCz = 0;
  float   lastVx = 0, lastVy = 0, lastVz = 0;
  float   halfExtent[3]  = { 0, 0, 0 };
  int32_t lastPointCount = 0;
  int32_t absorberID     = -1;
  int32_t age            = 0;
};

// Jonker-Volgenant shortest augmenting path over a square padded matrix. Holds
// its scratch across calls so a per-frame solve allocates nothing.
class HungarianSolver
{
 public:
  // cost is numRows x numCols, row major. Afterwards rowMatch()[r] is the
  // column assigned to row r, or -1.
  void solve(const float* cost, int numRows, int numCols)
  {
    myRowMatch.assign(std::max(numRows, 0), -1);
    if (numRows <= 0 || numCols <= 0)
      return;

    const int n = std::max(numRows, numCols);

    myCost.assign((size_t)n * n, 0.0f);
    for (int r = 0; r < numRows; r++)
      for (int c = 0; c < numCols; c++)
        myCost[(size_t)r * n + c] = cost[(size_t)r * numCols + c];

    myU.assign(n + 1, 0.0f);
    myV.assign(n + 1, 0.0f);
    myP.assign(n + 1, 0);
    myWay.assign(n + 1, 0);
    myMinv.resize(n + 1);
    myUsed.resize(n + 1);

    for (int row = 1; row <= n; row++)
    {
      myP[0] = row;
      int j0 = 0;

      std::fill(myMinv.begin(), myMinv.end(), FLT_MAX);
      std::fill(myUsed.begin(), myUsed.end(), (char)0);

      do
      {
        myUsed[j0] = 1;

        const int i0 = myP[j0];
        float delta  = FLT_MAX;
        int   j1     = 0;

        for (int j = 1; j <= n; j++)
        {
          if (myUsed[j])
            continue;

          const float cur = myCost[(size_t)(i0 - 1) * n + (j - 1)] - myU[i0] - myV[j];
          if (cur < myMinv[j])
          {
            myMinv[j] = cur;
            myWay[j]  = j0;
          }
          if (myMinv[j] < delta)
          {
            delta = myMinv[j];
            j1    = j;
          }
        }

        for (int j = 0; j <= n; j++)
        {
          if (myUsed[j])
          {
            myU[myP[j]] += delta;
            myV[j]      -= delta;
          }
          else
          {
            myMinv[j] -= delta;
          }
        }

        j0 = j1;
      } while (myP[j0] != 0);

      do
      {
        const int j1 = myWay[j0];
        myP[j0] = myP[j1];
        j0 = j1;
      } while (j0);
    }

    // Drop pairings that only involve the padding.
    for (int j = 1; j <= numCols; j++)
    {
      const int row = myP[j];
      if (row >= 1 && row <= numRows)
        myRowMatch[row - 1] = j - 1;
    }
  }

  const std::vector<int>& rowMatch() const { return myRowMatch; }

 private:
  std::vector<float> myCost;
  std::vector<float> myU, myV, myMinv;
  std::vector<int>   myP, myWay, myRowMatch;
  std::vector<char>  myUsed;
};
