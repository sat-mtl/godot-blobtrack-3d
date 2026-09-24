extends MultiMeshInstance3D
class_name TestMesh
var points: PackedVector3Array
var redraw:bool = false
var points_buffer := RID()
var points_uniform := RDUniform.new()
var rd := RenderingServer.get_rendering_device()
static var width := 13
static var count := 100
# initialize with the maximum points that the blob tracker can track.
var pts := 8_000_000
var label3ds : Array[Label3D] = []

class PointsFrame extends RefCounted:
	var rd := RenderingServer.get_rendering_device()
	var points_buffer:=RID()
	var points_uniform:=RDUniform.new()
	var points:PackedVector3Array
	var pts := 0
	var multimesh_buffer:PackedFloat32Array
	func _init() -> void:

		var vec:= PackedVector3Array()
		vec.resize(TestMesh.count*TestMesh.width*TestMesh.width*TestMesh.width)
		var bytes := vec.to_byte_array()
		points_buffer = rd.storage_buffer_create(bytes.size(), bytes)
		points_uniform.binding = 0
		points_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		points_uniform.add_id(points_buffer)

var num_frames := 100
var pts_frames : Array[PointsFrame] = []

var points_num_buffer:=RID()
var points_num_uniform:=RDUniform.new()

func _ready():
	for i in range(1000):
		var l3d = Label3D.new()
		add_child(l3d)
		l3d.position = Vector3.ONE * -666
		l3d.text = str(i)
		l3d.font_size = 100
		label3ds.append(l3d)
		l3d.billboard = true
	var bytes := PackedByteArray()
	bytes.resize(12)
	points_num_buffer = rd.storage_buffer_create(12, bytes, RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT)
	points_num_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	# points num binding should always be 11
	points_num_uniform.binding = 11
	points_num_uniform.add_id(points_num_buffer)

	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D

	var pmesh := PointMesh.new()
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.albedo_color=Color(1,0,1)
	material.point_size=10
	pmesh.material=material
	multimesh.mesh=pmesh
	%blobtrack3d.initialize_gpu_resources(pts, rd)
	var init_poss := PackedVector3Array()
	var scs := PackedFloat32Array()
	var amplitudes:=PackedVector3Array()
	for i in range(count):
		init_poss.append(Vector3(randf_range(0.0, 10), randf_range(0.0, 10), randf_range(0.0, 10)))
		scs.append(randf_range(0.001, 0.1))
		amplitudes.append(Vector3(randf_range(-3, 3), randf_range(-3, 3), randf_range(-3, 3)))
	for i in range(num_frames):
		var pts_frm := PointsFrame.new()
		pts_frames.append(pts_frm)
		create_points(init_poss, scs, i, amplitudes, pts_frm)

func set_points(pts_frame:PointsFrame):
	multimesh.instance_count = pts_frame.pts
	multimesh.buffer = pts_frame.multimesh_buffer

func create_points(init_poss:PackedVector3Array, scs:PackedFloat32Array, frame_num:int, amplitudes:PackedVector3Array,pts_frame:PointsFrame):
	var tf := Transform3D()
	var point_idx := 0
	pts_frame.points.clear()
	multimesh.instance_count = count*width*width*width
	var thinning := randf_range(0, 0.5)
	for i in range(count):
		var pos := init_poss[i] + (Vector3(sin(float(frame_num) / num_frames * 2.0*PI), cos(float(frame_num)/num_frames * 2*PI), -sin(float(frame_num)/num_frames * 2.0*PI))) * amplitudes[i]
		var sc := scs[i]
		for x in range(width):
			for y in range(width):
				for z in range(width):
					if randf() < thinning:
						continue
					var vec :=Vector3(x*sc, y*sc, z*sc) + pos
					pts_frame.points.append(vec)
					multimesh.set_instance_transform(point_idx, tf.translated(vec))
					point_idx+=1
	var pt_bytes := pts_frame.points.to_byte_array()
	pts_frame.pts = point_idx
	rd.buffer_update(pts_frame.points_buffer, 0, pt_bytes.size(), pt_bytes)
	pts_frame.multimesh_buffer = multimesh.buffer.slice(0, point_idx*12)

var frame_idx:=0

func _process(_delta: float) -> void:
	var current_frame := pts_frames[frame_idx%num_frames]
	frame_idx+=1
	set_points(current_frame)
	%blobtrack3d.cluster_dist = 0.1
	%blobtrack3d.min_points = 100
	%blobtrack3d.update_compute_shader_buffers(current_frame.pts)
	rd.buffer_update(points_num_buffer, 0, 12, PackedInt32Array([current_frame.pts, 1, 1]).to_byte_array())
	var compute_list := rd.compute_list_begin()
	print(current_frame.pts)
	# direct dispatch dispatch if you know exactly the number of points there are in the frame
	#%blobtrack3d.add_dispatches_to_compute_list(compute_list, current_frame.points_buffer, current_frame.points_uniform, current_frame.pts)
	# indirect dispatch, you would populate the points_count uniform with the number of points in the
	# frame from another compute shader.
	%blobtrack3d.add_dispatch_indirect(compute_list, current_frame.points_buffer, current_frame.points_uniform, points_num_buffer, points_num_uniform)
	rd.compute_list_end()
	var blobs=  %blobtrack3d.read_result()
	var cnt := 0
	for blob in blobs:
		DebugDraw.draw_box_aabb(blob.bounding_box, Color(1,1,1,1))
		DebugDraw.draw_line_3d(blob.centroid, blob.centroid + blob.velocity*3, Color(1,1,1,1))
		label3ds[cnt].position = blob.centroid
		label3ds[cnt].text = str(blob.blob_id)
		cnt+=1
	while cnt < %blobtrack3d.max_blobs:
		label3ds[cnt].position = Vector3.ONE*-666
		cnt += 1
