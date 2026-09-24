#pragma once

#include "godot_cpp/variant/packed_byte_array.hpp"
#include "godot_cpp/variant/array.hpp"

#include "godot_cpp/classes/ref.hpp"
#include "godot_cpp/classes/ref_counted.hpp"
#include "tracking.hpp"

using namespace godot;
constexpr int max_blobs = 4046;
class BlobTrack3D : public RefCounted {
  GDCLASS(BlobTrack3D, RefCounted)

protected:
  static void _bind_methods();
public:
  BlobTrack3D();
  ~BlobTrack3D() override = default;
  const Array& track_blobs(PackedByteArray blobs, int num_detected);
  float current_cluster_dist = 0.1;
  float current_match_dist = 5.0;
  int32_t current_min_points = 100;
  int32_t current_max_blobs = 100;
  int32_t current_max_age = 120;
  int32_t current_min_hits = 10;
  // this should probably be a time in seconds otherwise this will
  // cause the blobtrack behaviour to depend on the framerate.
  int32_t current_merge_memory_frames = 960;
  float current_smoothing = 0.5;
  bool current_use_point_count = true;
  float current_point_count_weight = 3.25;
  float get_cluster_dist() {
    return current_cluster_dist;
  }
  void set_cluster_dist(float cluster_dist) {
    current_cluster_dist = cluster_dist;
  }
  float get_match_dist() {
    return current_match_dist;
  }
  void set_match_dist(float match_dist) {
    current_match_dist = match_dist;
  }
  int32_t get_min_points() {
    return current_min_points;
  }
  void set_min_points(int32_t min_points) {
    current_min_points = min_points;
  }
  int32_t get_max_blobs() {
    return current_max_blobs;
  }
  void set_max_blobs(int32_t max_blobs) {
    current_max_blobs = max_blobs;
  }
  int32_t get_max_age() {
    return current_max_age;
  }
  void set_max_age(int32_t max_age) {
    current_max_age = max_age;
  }
  int32_t get_min_hits() {
    return current_min_hits;
  }
  void set_min_hits(int32_t min_hits) {
    current_min_hits = min_hits;
  }
  int32_t get_merge_memory_frames() {
    return current_merge_memory_frames;
  }
  void set_merge_memory_frames(int32_t merge_memory_frames) {
    current_merge_memory_frames = merge_memory_frames;
  }
  float get_smoothing() {
    return current_smoothing;
  }
  void set_smoothing(float smoothing) {
    current_smoothing = smoothing;
  }
  bool get_use_point_count() {
    return current_use_point_count;
  }
  void set_use_point_count(bool use_point_count) {
    current_use_point_count = use_point_count;
  }
  float get_point_count_weight() {
    return current_point_count_weight;
  }
  void set_point_count_weight(float point_count_weight) {
    current_point_count_weight = point_count_weight;
  }
  void reset();

private:
  int current_frame_count{};
  std::vector<Blob>& apply_smoothing(float smoothing);
  void absorb_unmatched_tracks();
  void recover_split_tracks(int num_detected);
  void track_frame(int num_detected);
  void build_output(const std::vector<Blob>& pre_out);
  std::vector<KalmanTracker3D> trackers{};
  std::vector<MergeMemory> merge_memories{};
  std::vector<Blob> tracked_blobs{};
  std::vector<Blob> smoothed_blobs{};
  int32_t next_blob_id = 0;
  std::array<Blob, max_blobs> detections{};
  std::vector<Blob> predictions{};
  std::vector<float> cost_matrix{};
  std::vector<char> detections_matched{};
  std::vector<char> tracks_matched{};
  std::vector<std::pair<int, int>> matches{};
  std::array<Ref<GodotBlob>, max_blobs> output_blobs;
  Array blob_output;
  Array blob_slice;
  HungarianSolver solver{};
};
