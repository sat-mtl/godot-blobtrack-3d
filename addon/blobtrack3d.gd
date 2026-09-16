extends Node
const local_size := 256
const merge_local_size := 128
# Elements one workgroup scans in a pass: kLocalSize threads, 4 each.
const scan_block := local_size * 4

var rd: RenderingDevice

# Hash table bounds. A power of two turns the modulo into a mask; the ceiling
# keeps the block-sum scan to a single workgroup pass (4M / 1024 = 4096 blocks,
# which 256 threads cover in 16 tiles).
const min_table_size := 1 << 10
const max_table_size := 1 << 22

func table_size_for(num_points:int) -> int:
	var ts = min_table_size
	while (ts < num_points and ts < max_table_size):
		ts <<= 1
	return ts

const float_size := 4
const uint_size := 4
const floats_per_point := 3
const max_workgroup_count := 65536
# corresponds to the max workgroup in one dimension. we could potentially change
# this if we want to track more stuff
const max_clusters := max_workgroup_count
const meta_uint_size := 16
# 10 fields of 4 bytes for blob detection data
const blob_data_size := 10 * 4
const max_blobs := 4046
#const knn_samples := 256
#const knn_max_k := 8
#const knn_read_size := knn_samples * float_size
const blob_read_size := max_blobs * blob_data_size

class Params extends RefCounted:
	var cluster_dist:= 0.1
	var auto_scale:= false
	var min_points := 100
	# not sure this is a decent default ?
	var max_blobs := 4000
#	var knnk := 0
class BufferResources extends RefCounted:
		var buffer := RID()
		var uniform := RDUniform.new()
		var rd:RenderingDevice
		static var empty_buffer:= PackedByteArray()
		func create_buffer_with_device_address(buffer_size:int) -> RID:
			empty_buffer.resize(buffer_size)
			if not rd.has_feature(RenderingDevice.Features.SUPPORTS_BUFFER_DEVICE_ADDRESS):
				return rd.storage_buffer_create(buffer_size, empty_buffer)
			else:
				return rd.storage_buffer_create(buffer_size, empty_buffer, 0, RenderingDevice.BUFFER_CREATION_DEVICE_ADDRESS_BIT)

		func _init(byte_size:int, binding, rendering_device:RenderingDevice):
			rd = rendering_device
			buffer = create_buffer_with_device_address(byte_size)
			uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
			uniform.binding = binding
			uniform.add_id(buffer)
class ShaderResources extends RefCounted:
		var shader:RID
		var pipeline:RID
		var uniform_set:RID
		var rd:RenderingDevice
		func _init(path, rendering_device:RenderingDevice) -> void:
			rd = rendering_device
			var shader_file := load(path)
			print(shader_file)
			var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()
			shader = rd.shader_create_from_spirv(shader_spirv)
			pipeline = rd.compute_pipeline_create(shader)

		func set_push_constants(compute_list, constants_array:Array):
			## NOTE: only works with 4 bytes constants.
			var constants_bytes:= PackedByteArray()
			var idx:= 0
			constants_bytes.resize(constants_array.size()*4)
			for constant in constants_array:
				if constant is float:
					constants_bytes.encode_float(idx*4, constant)
				else:
					constants_bytes.encode_s32(idx*4, constant)
				idx+=1
			rd.compute_list_set_push_constant(compute_list, constants_bytes, constants_bytes.size())
		func create_and_bind_uniform_set(compute_list, buffer_res:Array[RDUniform]):
			uniform_set = rd.uniform_set_create(buffer_res, shader, 0)
			rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		func bind_pipeline(compute_list):
			rd.compute_list_bind_compute_pipeline(compute_list, pipeline)


class GPUResource extends RefCounted:

	var rd: RenderingDevice
	var initialized := false
	var max_points := 0
	var table_size := 0
	var blocks := 0
	var error := ""
	var device := ""
	var bbox_count_shader : ShaderResources
	var scan_blocks_shader : ShaderResources
	var scan_sum_shader : ShaderResources
	var scan_add_shader : ShaderResources
	var scatter_shader : ShaderResources
	var merge_shader : ShaderResources
	var flatten_shader : ShaderResources
	var propagate_shader : ShaderResources
	var accumulate_shader : ShaderResources
	var build_blobs_shader : ShaderResources
	#var knn_shader : ShaderResources
	var accum_init_shader : ShaderResources
	func get_all_shader_resources():
		return [bbox_count_shader, scan_blocks_shader, scan_sum_shader, scan_add_shader,
			scatter_shader, merge_shader, flatten_shader, propagate_shader,
			accumulate_shader, build_blobs_shader,
			#knn_shader,
			accum_init_shader,]

	var accum_init_pipeline : RID
	var points_uniform : RDUniform
	var sorted_points_resources : BufferResources
	var cell_count_resources : BufferResources
	var cell_start_resources : BufferResources
	var block_sum_resources : BufferResources
	var parent_resources : BufferResources
	var cluster_id_resources : BufferResources
	var accum_resources : BufferResources
	var meta_resources : BufferResources
	var blobs_resources : BufferResources
	#var knn_resources : BufferResources

	func get_xyz_invocations(required_invocations:int, threads_per_workgroup:int = 64) -> Vector3:
		# number of workgroups in X,Y,Z.
		# Algorithm adapted from https://github.com/ossia/score/blob/master/src/plugins/score-plugin-gfx/Gfx/Graph/RenderedCSFNode.cpp#L1372
		# to compute the right size for the workgroup dimensions.
		# example : if we have threads_per_workgroup=64 and 65 invocations, we need 2 workgroups. (65 + 63)/64 == 2
		@warning_ignore("integer_division")
		var total_workgroups := (required_invocations + threads_per_workgroup - 1) / threads_per_workgroup;
		const max_workgroups := 65535
		var dispatch = Vector3.ZERO
		# the logic of those steps is that everytime we overflow max_workgroups on one dimension,
		if(total_workgroups > max_workgroups * max_workgroups):
			dispatch.x = max_workgroups;
			@warning_ignore("integer_division")
			var remaining := (total_workgroups + max_workgroups - 1) / max_workgroups;
			dispatch.y = min(remaining, max_workgroups);
			@warning_ignore("integer_division")
			dispatch.z = (remaining + (max_workgroups - 1)) / max_workgroups;
		elif(total_workgroups > max_workgroups):
			dispatch.x = min(total_workgroups, max_workgroups);
			@warning_ignore("integer_division")
			dispatch.y = (total_workgroups + max_workgroups - 1) / max_workgroups;
			dispatch.z = 1;
		else:
			dispatch.x = total_workgroups;
			dispatch.y = 1;
			dispatch.z = 1;
		return dispatch
	var init_buffer_uniform_set:= RID()

	func reinitialize_buffers():
		if init_buffer_uniform_set.is_valid():
			rd.free_rid(init_buffer_uniform_set)
		rd.buffer_clear(cell_count_resources.buffer, 0, table_size*uint_size)
		var meta_buffer = PackedInt32Array()
		# TODO: using binary float min and float max here instead of something more
		# reasonnable seems weird to me.
		var float_max = 0x7F7FFFFF
		var float_min = 0xFF7FFFFF
		meta_buffer.resize(meta_uint_size)
		for i in range(meta_uint_size):
			if i < 3:
				meta_buffer[i] = float_max
			elif i < 6:
				meta_buffer[i] = float_min
			else:
				meta_buffer[i] = 0

		var meta_bytes = meta_buffer.to_byte_array()
		rd.buffer_update(meta_resources.buffer, 0, meta_bytes.size(), meta_bytes)
		init_buffer_uniform_set = rd.uniform_set_create([
			accum_resources.uniform,
			], accum_init_shader.shader, 0
		)
		var compute_list := rd.compute_list_begin()

		rd.compute_list_bind_compute_pipeline(compute_list, accum_init_shader.pipeline)
		var accum_buffer_size = 10*max_clusters
		rd.compute_list_set_push_constant(compute_list, PackedInt32Array([accum_buffer_size, max_clusters]).to_byte_array(), 8)
		rd.compute_list_bind_uniform_set(compute_list, init_buffer_uniform_set, 0)
		# we need to be called for max_points because we need to clear unused points.
		var xyz_invoc = get_xyz_invocations(max_clusters * 10)
		rd.compute_list_dispatch(compute_list, xyz_invoc.x, xyz_invoc.y, xyz_invoc.z)
		rd.compute_list_end()

	func _init(max_points, table_size, blocks, rendering_device:RenderingDevice) -> void:
		rd = rendering_device
		# compile the shaders
		bbox_count_shader = ShaderResources.new("res://addons/shaders/boundingbox_and_histogram.glsl", rd)
		scan_blocks_shader = ShaderResources.new("res://addons/shaders/histogram_scan.glsl", rd)
		scan_sum_shader = ShaderResources.new("res://addons/shaders/scan_block_sum.glsl", rd)
		scan_add_shader = ShaderResources.new("res://addons/shaders/scan_add.glsl", rd)
		scatter_shader = ShaderResources.new("res://addons/shaders/scatter.glsl", rd)
		merge_shader = ShaderResources.new("res://addons/shaders/merge.glsl", rd)
		flatten_shader = ShaderResources.new("res://addons/shaders/flatten_label.glsl", rd)
		propagate_shader = ShaderResources.new("res://addons/shaders/propagate.glsl", rd)
		accumulate_shader = ShaderResources.new("res://addons/shaders/accumulate.glsl", rd)
		build_blobs_shader = ShaderResources.new("res://addons/shaders/build_blobs.glsl", rd)
		#knn_shader = ShaderResources.new("res://addons/shaders/src-knn.glsl", rd)
		accum_init_shader = ShaderResources.new("res://addons/shaders/clear_accum.glsl", rd)
		initialized = true
		sorted_points_resources = BufferResources.new(max_points * floats_per_point * float_size, 1, rd)
		cell_count_resources = BufferResources.new(table_size * uint_size, 2, rd)
		cell_start_resources = BufferResources.new((table_size+1) * uint_size, 3, rd)
		block_sum_resources = BufferResources.new(blocks * uint_size, 4, rd)
		parent_resources = BufferResources.new(max_points * uint_size, 5, rd)
		# why max_points and not max_clusters ?
		cluster_id_resources = BufferResources.new(max_points * uint_size, 6, rd)
		accum_resources = BufferResources.new(max_clusters * 10 * uint_size, 7, rd)
		meta_resources = BufferResources.new(meta_uint_size * uint_size, 8, rd)
		blobs_resources = BufferResources.new(blob_read_size, 9, rd)
		#knn_resources = BufferResources.new(knn_read_size, 10, rd)

# returns the number of invocation for a given n with
# a certain local size. This is limited to the x dimension of a dispatch
func grid_for(n:int, local_size:int):
	if n <= 0:
		return 1
	var groups := (n + local_size -1) / local_size
	return min(groups, max_workgroup_count)

var gpu_res : GPUResource
var params := Params.new()
func initialize_gpu_resources(max_points:int, rendering_device:RenderingDevice):
	rd = rendering_device
	var table_size := table_size_for(max_points)
	var blocks := table_size/scan_block
	gpu_res = GPUResource.new(max_points, table_size, blocks, rendering_device)
	gpu_res.table_size = table_size
	gpu_res.blocks = blocks


func update_compute_shader_buffers(num_pts):
	# reset error string
	gpu_res.error = ""
	# don't know where 1e-8 comes from.
	gpu_res.reinitialize_buffers()
	for shader_resources: ShaderResources in gpu_res.get_all_shader_resources():
		if shader_resources.uniform_set.is_valid():
			# we need to cleanup our uniform set, there seems to be no way to update it
			# so we need to create one every frame and if we don't free we eventually crash
			rd.free_rid(shader_resources.uniform_set)

#var current_knnk: int

func add_dispatches_to_compute_list(compute_list, points_buffer_rid:RID, points_uniform:RDUniform, num_pts):
	var current_cluster_distance :float = max(params.cluster_dist, 1e-8)
	var current_max_blobs := clamp(params.max_blobs,1, max_blobs)
	#current_knnk = clamp(params.knnk, 0, knn_max_k)
	var cell_size := 1.0 / current_cluster_distance;
	# TODO: indirect dispatch here
	var grid_points  = grid_for(num_pts, local_size);
	var grid_merge   = grid_for(num_pts, merge_local_size);
	var grid_table   = grid_for(gpu_res.table_size, local_size);
	var grid_cluster = grid_for(max_clusters, local_size);
	var scan_blocks  = gpu_res.table_size / scan_block;

	# 1. adds dispatch for bounding boxes
	gpu_res.bbox_count_shader.bind_pipeline(compute_list)
	gpu_res.bbox_count_shader.create_and_bind_uniform_set(compute_list, [
		points_uniform, gpu_res.cell_count_resources.uniform, gpu_res.meta_resources.uniform
	])
	gpu_res.bbox_count_shader.set_push_constants(compute_list, [num_pts, gpu_res.table_size, cell_size])
	rd.compute_list_dispatch(compute_list, grid_points, 1, 1)
	rd.compute_list_add_barrier(compute_list)
	# 2. adds dispatch for scan blocks
	gpu_res.scan_blocks_shader.bind_pipeline(compute_list)
	gpu_res.scan_blocks_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.cell_count_resources.uniform, gpu_res.cell_start_resources.uniform,  gpu_res.block_sum_resources.uniform
	])
	gpu_res.scan_blocks_shader.set_push_constants(compute_list, [])

	rd.compute_list_dispatch(compute_list, scan_blocks, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#3.
	gpu_res.scan_sum_shader.bind_pipeline(compute_list)
	gpu_res.scan_sum_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.cell_start_resources.uniform, gpu_res.block_sum_resources.uniform, gpu_res.meta_resources.uniform
	])
	gpu_res.scan_sum_shader.set_push_constants(compute_list, [scan_blocks, gpu_res.table_size])
	rd.compute_list_dispatch(compute_list, 1,1,1)
	rd.compute_list_add_barrier(compute_list)

	#4.
	gpu_res.scan_add_shader.bind_pipeline(compute_list)
	gpu_res.scan_add_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.cell_count_resources.uniform, gpu_res.cell_start_resources.uniform,  gpu_res.block_sum_resources.uniform
	])
	gpu_res.scan_add_shader.set_push_constants(compute_list, [gpu_res.table_size])
	rd.compute_list_dispatch(compute_list, grid_table, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#5.
	gpu_res.scatter_shader.bind_pipeline(compute_list)
	gpu_res.scatter_shader.create_and_bind_uniform_set(compute_list, [
		# the binding 2 ( cell_count ) is called "cursor" in the shader ???
		points_uniform, gpu_res.sorted_points_resources.uniform, gpu_res.cell_count_resources.uniform, gpu_res.cell_start_resources.uniform, gpu_res.parent_resources.uniform
	])
	gpu_res.scatter_shader.set_push_constants(compute_list, [num_pts, gpu_res.table_size, cell_size])
	rd.compute_list_dispatch(compute_list, grid_points, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#6.
	gpu_res.merge_shader.bind_pipeline(compute_list)
	gpu_res.merge_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.sorted_points_resources.uniform, gpu_res.cell_start_resources.uniform, gpu_res.parent_resources.uniform, gpu_res.meta_resources.uniform
	])
	gpu_res.merge_shader.set_push_constants(compute_list, [num_pts, gpu_res.table_size, cell_size, current_cluster_distance*current_cluster_distance])
	rd.compute_list_dispatch(compute_list, grid_merge, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#7.
	gpu_res.flatten_shader.bind_pipeline(compute_list)
	gpu_res.flatten_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.parent_resources.uniform, gpu_res.cluster_id_resources.uniform, gpu_res.meta_resources.uniform
	])
	gpu_res.flatten_shader.set_push_constants(compute_list, [num_pts, max_clusters])
	rd.compute_list_dispatch(compute_list, grid_points, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#8.
	gpu_res.propagate_shader.bind_pipeline(compute_list)
	gpu_res.propagate_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.parent_resources.uniform, gpu_res.cluster_id_resources.uniform, gpu_res.meta_resources.uniform
	])
	gpu_res.propagate_shader.set_push_constants(compute_list, [num_pts])
	rd.compute_list_dispatch(compute_list, grid_points, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#9.
	gpu_res.accumulate_shader.bind_pipeline(compute_list)
	gpu_res.accumulate_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.sorted_points_resources.uniform, gpu_res.cluster_id_resources.uniform, gpu_res.accum_resources.uniform, gpu_res.meta_resources.uniform
	])
	gpu_res.accumulate_shader.set_push_constants(compute_list, [num_pts, max_clusters])
	rd.compute_list_dispatch(compute_list, grid_points, 1, 1)
	rd.compute_list_add_barrier(compute_list)

	#10.
	gpu_res.build_blobs_shader.bind_pipeline(compute_list)
	gpu_res.build_blobs_shader.create_and_bind_uniform_set(compute_list, [
		gpu_res.accum_resources.uniform, gpu_res.meta_resources.uniform, gpu_res.blobs_resources.uniform
	])
	gpu_res.build_blobs_shader.set_push_constants(compute_list, [params.min_points, max_clusters, current_max_blobs])
	rd.compute_list_dispatch(compute_list, grid_cluster, 1, 1)

	# no barrier between blobs and knn they can be run in parallel ?
	#if current_knnk > 0:
		##11.
		#gpu_res.knn_shader.bind_pipeline(compute_list)
		#gpu_res.knn_shader.create_and_bind_uniform_set(compute_list, [
			#gpu_res.sorted_points_resources.uniform, gpu_res.cell_start_resources.uniform, gpu_res.meta_resources.uniform, gpu_res.knn_resources.uniform
		#])
		#gpu_res.knn_shader.set_push_constants(compute_list, [current_knnk, cell_size, knn_samples, gpu_res.table_size])
		#rd.compute_list_dispatch(compute_list, grid_for(knn_samples, local_size), 1, 1)

enum meta {num_valid=12, cluster_count=13, blob_count=14, overflow=15}

class ClusterResult extends RefCounted:
	var num_blobs:=0
	var valid_points:= 0
	var num_cluster:= 0
	var overflow:= false
	#var knn_spacing:= 0.0

class BlobResult extends RefCounted:
	var centroid := Vector3.ZERO
	var bounding_box_min := Vector3.ZERO
	var bounding_box_max := Vector3.ZERO
	var point_count := 0

func _ready() -> void:
	blob_results = []
	for i in range(max_blobs):
		blob_results.append(BlobResult.new())
	print("ready")


var cluster_result := ClusterResult.new()
## only contains cluster_result.num_blobs valid blobs at any given time.
var blob_results : Array[BlobResult]
func read_rest_of_results(data:PackedByteArray):
	var meta_results = data.to_int32_array()
	meta_results = rd.buffer_get_data(gpu_res.meta_resources.buffer, 0, meta_uint_size*uint_size).to_int32_array()
	print(meta_results)
	cluster_result.num_blobs = meta_results[meta.blob_count]
	cluster_result.valid_points = meta_results[meta.num_valid]
	cluster_result.num_cluster = meta_results[meta.cluster_count]
	cluster_result.overflow = bool(meta_results[meta.overflow])
	print("blob count ", cluster_result.num_blobs)
	print("num_valid ", cluster_result.valid_points)
	print("overflow ", cluster_result.overflow)
	print("cluster_count ", cluster_result.num_cluster)
	var blob_bytes = rd.buffer_get_data(gpu_res.blobs_resources.buffer, 0, blob_read_size)
	for i in range(cluster_result.num_blobs):
		var current_offset := i*blob_data_size
		blob_results[i].centroid = Vector3(
			blob_bytes.decode_float(current_offset),
			blob_bytes.decode_float(current_offset + float_size),
			blob_bytes.decode_float(current_offset + float_size*2)
		)
		current_offset += float_size*3
		blob_results[i].bounding_box_min = Vector3(
			blob_bytes.decode_float(current_offset),
			blob_bytes.decode_float(current_offset + float_size),
			blob_bytes.decode_float(current_offset + float_size*2)
		)
		current_offset += float_size*3
		blob_results[i].bounding_box_max = Vector3(
			blob_bytes.decode_float(current_offset),
			blob_bytes.decode_float(current_offset + float_size),
			blob_bytes.decode_float(current_offset + float_size*2)
		)
		current_offset += float_size*3
		blob_results[i].point_count = blob_bytes.decode_u32(current_offset)
		print("blob ", i)
		print("centroid", blob_results[i].centroid)

func read_result():
	#rd.buffer_get_data_async(gpu_res.meta_resources.buffer, read_rest_of_results, 0, meta_uint_size*uint_size)
	read_rest_of_results(PackedByteArray())
