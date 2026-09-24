#include "blobtrack3d.hpp"
#include <stdlib.h>
#include <unordered_map>
void BlobTrack3D::_bind_methods() {
  godot::ClassDB::bind_method(D_METHOD("track_blobs"), &BlobTrack3D::track_blobs);
  godot::ClassDB::bind_method(D_METHOD("set_cluster_dist", "p_cluster_dist"), &BlobTrack3D::set_cluster_dist);
  godot::ClassDB::bind_method(D_METHOD("get_cluster_dist"), &BlobTrack3D::get_cluster_dist);
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cluster_dist"), "set_cluster_dist", "get_cluster_dist");
  godot::ClassDB::bind_method(D_METHOD("set_match_dist", "p_match_dist"), &BlobTrack3D::set_match_dist);
  godot::ClassDB::bind_method(D_METHOD("get_match_dist"), &BlobTrack3D::get_match_dist);
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "match_dist"), "set_match_dist", "get_match_dist");
  godot::ClassDB::bind_method(D_METHOD("set_min_points", "p_min_points"), &BlobTrack3D::set_min_points);
  godot::ClassDB::bind_method(D_METHOD("get_min_points"), &BlobTrack3D::get_min_points);
  ADD_PROPERTY(PropertyInfo(Variant::INT, "min_points"), "set_min_points", "get_min_points");
  godot::ClassDB::bind_method(D_METHOD("set_max_blobs", "p_max_blobs"), &BlobTrack3D::set_max_blobs);
  godot::ClassDB::bind_method(D_METHOD("get_max_blobs"), &BlobTrack3D::get_max_blobs);
  ADD_PROPERTY(PropertyInfo(Variant::INT, "max_blobs"), "set_max_blobs", "get_max_blobs");
  godot::ClassDB::bind_method(D_METHOD("set_max_age", "p_max_age"), &BlobTrack3D::set_max_age);
  godot::ClassDB::bind_method(D_METHOD("get_max_age"), &BlobTrack3D::get_max_age);
  ADD_PROPERTY(PropertyInfo(Variant::INT, "max_age"), "set_max_age", "get_max_age");
  godot::ClassDB::bind_method(D_METHOD("set_min_hits", "p_min_hits"), &BlobTrack3D::set_min_hits);
  godot::ClassDB::bind_method(D_METHOD("get_min_hits"), &BlobTrack3D::get_min_hits);
  ADD_PROPERTY(PropertyInfo(Variant::INT, "min_hits"), "set_min_hits", "get_min_hits");
  godot::ClassDB::bind_method(D_METHOD("set_merge_memory_frames", "p_merge_memory_frames"), &BlobTrack3D::set_merge_memory_frames);
  godot::ClassDB::bind_method(D_METHOD("get_merge_memory_frames"), &BlobTrack3D::get_merge_memory_frames);
  ADD_PROPERTY(PropertyInfo(Variant::INT, "merge_memory_frames"), "set_merge_memory_frames", "get_merge_memory_frames");
  godot::ClassDB::bind_method(D_METHOD("set_smoothing", "p_smoothing"), &BlobTrack3D::set_smoothing);
  godot::ClassDB::bind_method(D_METHOD("get_smoothing"), &BlobTrack3D::get_smoothing);
  ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing"), "set_smoothing", "get_smoothing");
  godot::ClassDB::bind_method(D_METHOD("set_use_point_count", "p_use_point_count"), &BlobTrack3D::set_use_point_count);
  godot::ClassDB::bind_method(D_METHOD("get_use_point_count"), &BlobTrack3D::get_use_point_count);
  ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_point_count"), "set_use_point_count", "get_use_point_count");
  godot::ClassDB::bind_method(D_METHOD("set_point_count_weight", "p_point_count_weight"), &BlobTrack3D::set_point_count_weight);
  godot::ClassDB::bind_method(D_METHOD("get_point_count_weight"), &BlobTrack3D::get_point_count_weight);
  ADD_PROPERTY(PropertyInfo(Variant::BOOL, "point_count_weight"), "set_point_count_weight", "get_point_count_weight");
}

const Array& BlobTrack3D::track_blobs(PackedByteArray blobs, int num_detected) {
  for (int i = 0; i < num_detected; i++) {
    detections[i].populate_from_bytes(blobs, i);
  }
  track_frame(num_detected);
  auto pre_out = apply_smoothing(current_smoothing);
  build_output(pre_out);
  return blob_slice;
}

BlobTrack3D::BlobTrack3D() {
  // makes the link between the std::array of output blobs and the godot output.
  for (int i=0; i<max_blobs; i++) {
    output_blobs[i].instantiate();
    blob_output.append(*output_blobs[i]);
  }
}

void BlobTrack3D::build_output(const std::vector<Blob>& pre_out) {
  for (int i=0; i<pre_out.size(); i++) {
    GodotBlob* out_blob = dynamic_cast<GodotBlob*>(blob_output[i].get_validated_object());
    out_blob->set_from_blob(pre_out[i]);
  }
  blob_slice = blob_output.slice(0, pre_out.size());
}

void BlobTrack3D::track_frame(int num_detected) {
  // TODO: preallocate and don't clear
  predictions.clear();
  // no auto incrementation of counter because
  // we erase stuff from the structure during the loop
  for (int t = 0; t < trackers.size();) {
    const Blob pred = trackers[t].predict();
    // clears trackers lost to nan propagation
    if (std::isnan(pred.cx) || std::isnan(pred.cy) || std::isnan(pred.cz))
    {
      trackers.erase(trackers.begin() + t);
      continue;
    }
    predictions.push_back(pred);
    t++;
  }
  const int num_dets = num_detected;
  const int num_tracks = (int)trackers.size();
  detections_matched.assign(std::max(num_dets, 0), 0);
  tracks_matched.assign(std::max(num_tracks, 0), 0);
  matches.clear();
  if (num_dets > 0 && num_tracks > 0)
  {
    cost_matrix.resize((size_t)num_dets * num_tracks);
    for (int d = 0; d < num_dets; d++)
      for (int t = 0; t < num_tracks; t++)
        cost_matrix[(size_t)d * num_tracks + t] =
            blobMatchCost(detections[d], predictions[t],
                          current_use_point_count, current_point_count_weight);

    solver.solve(cost_matrix.data(), num_dets, num_tracks);

    // The assignment is over a padded square matrix, so a pairing is only
    // real if the two are actually close enough.
    const std::vector<int>& row_match = solver.rowMatch();
    for (int d = 0; d < num_dets; d++)
    {
      const int t = row_match[d];
      if (t < 0 || t >= num_tracks)
        continue;
      if (blobDist(detections[d], predictions[t]) > current_match_dist)
        continue;

      matches.push_back({ d, t });
      detections_matched[d] = 1;
      tracks_matched[t] = 1;
    }
  }

  for (const auto& m : matches) {
    trackers[m.second].update(detections[m.first]);
  }

  if (current_merge_memory_frames > 0) {
    absorb_unmatched_tracks();
  }

  recover_split_tracks(num_detected);

  // create a kalman tracker for detections that
  // did not already have one
  for (int d = 0; d < num_dets; d++)
  {
    if (detections_matched[d])
      continue;
    KalmanTracker3D trk;
    trk.init(detections[d], next_blob_id++);
    trackers.push_back(trk);
  }
  // delete merge memories that are too old
  for (size_t i = 0; i < merge_memories.size();)
  {
    if (++merge_memories[i].age > current_merge_memory_frames)
      merge_memories.erase(merge_memories.begin() + i);
    else
      i++;
  }
  // delete trackers that are too old
  trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                [&](const KalmanTracker3D& t)
                                { return t.timeSinceUpdate > current_max_age; }),
                 trackers.end());

  // TODO: use the blobs already allocated here instead of
  // clearing and making new blobs.
  tracked_blobs.clear();
  for (auto& trk : trackers)
  {
    const bool updated_this_frame = trk.timeSinceUpdate == 0;
    // not sure I understand this condition ... a blob is established if its frame_count
    // is smaller than min_hits ?
    const bool established = trk.hitStreak >= current_min_hits || current_frame_count <= current_min_hits;
    if (updated_this_frame && established)
      tracked_blobs.push_back(trk.getState());
  }
  current_frame_count++;
}

void BlobTrack3D::absorb_unmatched_tracks() {
  const int num_trks = (int)trackers.size();

  for (int t = 0; t < num_trks; t++)
  {
    // if the tracker has a blob matched to it, don't iterate over it.
    if (tracks_matched[t])
      continue;

    const Blob& pred = predictions[t];
    float best_dist   = FLT_MAX;
    int best_tracker = -1;
    // tries to find an id-matched blob near
    // where the unmatched tracker is and mark it as
    // the best match
    for (const auto& m : matches)
    {
      const Blob& absorbed = detections[m.first];
      const float threshold = (pred.radius() + absorbed.radius()) * 2.0f;
      const float dist = blobDist(pred, absorbed);
      if (dist < threshold && dist < best_dist)
      {
        best_dist    = dist;
        best_tracker = m.second;
      }
    }
    // if nothing is even close just give up on this tracker.
    if (best_tracker < 0)
      continue;

    // look to see if we already knew that blob
    const int32_t id = trackers[t].id;

    bool alreadyStored = false;
    for (const auto& existing : merge_memories)
    {
      if (existing.storedID == id)
      {
        alreadyStored = true;
        break;
      }
    }
    // if it doesn't exist create it
    if (!alreadyStored)
    {
      MergeMemory mm;
      mm.storedID = id;
      mm.lastCx = pred.cx; mm.lastCy = pred.cy; mm.lastCz = pred.cz;
      mm.lastVx = pred.vx; mm.lastVy = pred.vy; mm.lastVz = pred.vz;
      mm.halfExtent[0] = std::max(trackers[t].kfHx.x[0], 0.001f);
      mm.halfExtent[1] = std::max(trackers[t].kfHy.x[0], 0.001f);
      mm.halfExtent[2] = std::max(trackers[t].kfHz.x[0], 0.001f);
      mm.lastPointCount = trackers[t].lastPointCount;
      mm.absorberID = trackers[best_tracker].id;
      merge_memories.push_back(mm);
    }

    // TODO: use time instead of frames.
    trackers[t].timeSinceUpdate = current_max_age + 1;
  }

}

// The other half of merge handling: an unmatched detection near a remembered
// track is most likely that track reappearing, so give it its old id back.
void BlobTrack3D::recover_split_tracks(int num_detected) {
  const int num_dets = num_detected;

  for (int d = 0; d < num_dets; d++)
  {
    if (detections_matched[d])
      continue;

    int   best_mem   = -1;
    float best_score = FLT_MAX;

    for (int mi = 0; mi < (int)merge_memories.size(); mi++)
    {
      const MergeMemory& mem = merge_memories[mi];

      Blob remembered;
      remembered.cx = mem.lastCx;
      remembered.cy = mem.lastCy;
      remembered.cz = mem.lastCz;

      KalmanTracker3D* absorber = nullptr;
      for (auto& trk : trackers)
      {
        if (trk.id == mem.absorberID)
        {
          absorber = &trk;
          break;
        }
      }
      // TODO : documenter et extraire les constantes utilisées ici.
      // peut être ajouter en paramètre mais ça va être fiddly.

      // Search around the absorber while it is alive, since the pair have
      // been moving together; fall back to the last known position once
      // the absorber is gone too.
      if (absorber)
      {
        const Blob state = absorber->getState();
        if (blobDist(detections[d], state) > state.radius() * 3.0f)
          continue;
      }
      else if (blobDist(detections[d], remembered) > current_match_dist * 2.0f)
      {
        continue;
      }

      const float memVol = std::max(mem.halfExtent[0] * mem.halfExtent[1] *
                                    mem.halfExtent[2] * 8.0f, 1e-10f);
      const float volRatio = detections[d].volume() / memVol;
      if (volRatio < 0.2f || volRatio > 5.0f)
        continue;

      float score = std::fabs(std::log(std::max(volRatio, 0.01f)))
          + (float)mem.age * 0.01f
          + blobDist(detections[d], remembered) * 0.5f;

      if (current_use_point_count && current_point_count_weight > 0.0f &&
          mem.lastPointCount > 0 && detections[d].pointCount > 0)
      {
        const float ratio =
            (float)std::min(detections[d].pointCount, mem.lastPointCount) /
            (float)std::max(detections[d].pointCount, mem.lastPointCount);
        score += (1.0f - ratio) * current_point_count_weight;
      }

      if (score < best_score)
      {
        best_score = score;
        best_mem   = mi;
      }
    }
    // pourquoi 10.0 ?
    // A loose match is worse than a new id, so only recover a confident one.
    if (best_mem < 0 || best_score >= 10.0f)
      continue;

    const int32_t recovered_id = merge_memories[best_mem].storedID;

    bool id_in_use = false;
    for (const auto& trk : trackers)
    {
      if (trk.id == recovered_id && trk.timeSinceUpdate <= current_max_age)
      {
        id_in_use = true;
        break;
      }
    }

    if (!id_in_use)
    {
      KalmanTracker3D trk;
      trk.init(detections[d], recovered_id);
      trackers.push_back(trk);
      detections_matched[d] = 1;
    }

    merge_memories.erase(merge_memories.begin() + best_mem);

  }
}

std::vector<Blob>& BlobTrack3D::apply_smoothing(float smoothing) {
  if (smoothing <= 0.0f)
  {
    return tracked_blobs;
  }
  // TODO: don't allocate here
  std::unordered_map<int32_t, const Blob*> previous;
  previous.reserve(smoothed_blobs.size());
  for (const auto& sb : smoothed_blobs) {
    previous.emplace(sb.id, &sb);
  }

  const float t = std::max(1.0f - smoothing, 0.02f);
  std::vector<Blob> next{};
  for (const auto& tracked : tracked_blobs)
  {
    // I think this needs to do a copy of the blobs because
    // it wants to keep the blob predictions and merge memory free
    // of smoothing.
    Blob b = tracked;
    const auto it = previous.find(b.id);
    if (it != previous.end())
    {
      const Blob& p = *it->second;
      const float k = 1.0f - t;
      b.cx   = p.cx   * k + b.cx   * t;
      b.cy   = p.cy   * k + b.cy   * t;
      b.cz   = p.cz   * k + b.cz   * t;
      b.bmnx = p.bmnx * k + b.bmnx * t;
      b.bmny = p.bmny * k + b.bmny * t;
      b.bmnz = p.bmnz * k + b.bmnz * t;
      b.bmxx = p.bmxx * k + b.bmxx * t;
      b.bmxy = p.bmxy * k + b.bmxy * t;
      b.bmxz = p.bmxz * k + b.bmxz * t;
      b.vx   = p.vx   * k + b.vx   * t;
      b.vy   = p.vy   * k + b.vy   * t;
      b.vz   = p.vz   * k + b.vz   * t;
    }
    next.push_back(b);
  }
  smoothed_blobs.swap(next);
  return smoothed_blobs;
}
