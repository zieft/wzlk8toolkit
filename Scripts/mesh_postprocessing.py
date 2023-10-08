# import pydevd
# pydevd.settrace('127.0.0.1', port=1090, stdoutToServer=True, stderrToServer=True)

import bpy
import sys
import os

sys.path.append(os.path.curdir)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(
    os.path.dirname(r'C:\Users\zieft\PycharmProjects\wzlk8toolkit\Scripts\mesh_postprocessing.py'))

os.chdir(BASE_DIR)
# output_dir='/storage/blender/output/mesh_postprocessed.obj'
output_dir = r'C:\Users\zieft\Desktop\final_test\mesh_postprocessed.obj'
# filePath = '/storage/blender/input/texturedMesh.obj'
filePath = r'C:\Users\zieft\Desktop\final_test\texturedMesh.obj'

from core.bpycore import *

## Delete initial objects.
if "Cube" in bpy.data.meshes:
    mesh = bpy.data.meshes["Cube"]
    print("removing mesh", mesh)
    bpy.data.meshes.remove(mesh)

if "Camera" in bpy.data.cameras:
    camera = bpy.data.cameras["Camera"]
    print("removing camera", camera)
    bpy.data.cameras.remove(camera)

if "Light" in bpy.data.lights:
    light = bpy.data.lights["Light"]
    print("removing light", light)
    bpy.data.lights.remove(light)

## Change ViewPoint Shading into 'Rendered'.
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'RENDERED'

bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))

## import .obj file
bpy.ops.import_scene.obj(filepath=filePath)

## set texturedMesh to the world origin
objectToSelect = bpy.data.objects['texturedMesh']
objectToSelect.select_set(True)
bpy.context.view_layer.objects.active = objectToSelect
bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

# Step 1: Define a sphere
cameraCoordinates, cameraNumbers = BlenderCameraOperation.generateDomeCoor(10, 10, 4)
lightCoordinates, lightNumbers = BlenderCameraOperation.generateDomeCoor(3, 3, 1)

# Step 2: place a camera on the sphere and track the world origin
cameraList = BlenderCameraOperation.addCamera(cameraCoordinates, cameraNumbers)
BlenderCameraOperation.addLightSources(lightCoordinates, lightNumbers)

# Step 3-5:
for camera in cameraList:
    BlenderCameraOperation.render_through_camera(camera)

max_len = 0
cameras_with_id_length = []
for camera in cameraList:
    img = ImageTransformProcess.readImageBIN(work_dir + '{}.png'.format(camera['name']))
    aruco_info = {}
    best_camera_angle_name = ''
    try:
        corners, ids, _ = ArucoInfoDetection.detect_save_aruco_info_image(camera, img)
        if ids is not None:
            best_camera_angle_name = camera['name']
            aruco_info['corners'] = corners
            aruco_info['ids'] = ids
            cameras_with_id_length.append([best_camera_angle_name, aruco_info])
            print(cameras_with_id_length)
            if len(ids) > max_len:
                max_len = len(ids)

    except cv2.error:
        pass

print(max_len)

# print('Selected Camera: {}'.format(best_camera_angle_name))

selected_cameras = []
for i in cameras_with_id_length:
    if len(i[1]['ids']) >= max_len - 1:
        selected_cameras.append(i)

plane_coefficients = np.array([0, 0, 0], dtype='f2')


all_list_detected_markers_obj = []
for camera in selected_cameras:
    # Step 6: Move the camera slightly to build a stereo pair
    bpy.ops.object.select_all(action='DESELECT')
    best_camera_angle_co = BlenderCameraOperation.add_co_camera(camera[0])
    BlenderCameraOperation.render_through_camera(best_camera_angle_co)
    co_img = ImageTransformProcess.readImageBIN(work_dir + '{}.png'.format(best_camera_angle_co))
    corners_co, ids_co, _ = ArucoInfoDetection.detect_save_aruco_info_image(best_camera_angle_co, co_img)
    aruco_info_co = {'corners': corners_co, 'ids': ids_co}
    # Step 7: Calculate the coordinates of the markers
    iml = ImageTransformProcess.readImageBIN(work_dir + '{}.png'.format(camera[0]), BIN=False)
    imr = ImageTransformProcess.readImageBIN(work_dir + '{}.png'.format(best_camera_angle_co), BIN=False)
    height, width = iml.shape[0:2]
    stereo_config = stereoCamera(camera[0], best_camera_angle_co)
    ## Stereo Rectify
    map1x, map1y, map2x, map2y, Q, cameraRecMat = ImageTransformProcess.getRectifyTransform(height, width,
                                                                                            stereo_config)
    iml_rectified, imr_rectified = ImageTransformProcess.rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    line = ImageTransformProcess.draw_line(iml_rectified, imr_rectified)
    plt.figure()
    plt.imshow(line)
    plt.savefig(work_dir + 'validation.png', dpi=1000)
    ## Stereo Match
    iml_, imr_ = ImageTransformProcess.preprocess(iml, imr)
    disp, _ = ImageTransformProcess.stereoMatchSGBM(iml_rectified, imr_rectified, True)
    plt.figure()
    plt.imshow(disp)
    plt.savefig(work_dir + 'z_disparity_map.png', dpi=1000)
    ## Reproject image to 3D World
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    points_3d[:, :, 1:3] = -points_3d[:, :, 1:3]  # y, z direction in OpenCV and Blender are different
    ## New algrithm
    aruco_info_better = ArucoInfoDetection.better_aruco_info(camera[1])
    aruco_info_co_better = ArucoInfoDetection.better_aruco_info(aruco_info_co)
    if not aruco_info_better or not aruco_info_co_better:
        continue
    aruco_info_common, aruco_info_co_common, common_ids = DetectedArUcoMarker_world.detected_markers_common(
        aruco_info_better, aruco_info_co_better)
    allMarkers, allCorners = StereoPointObject.generate_dict_of_stereo_point_pairs_obj_allMarker(aruco_info_common,
                                                                                                 aruco_info_co_common,
                                                                                                 common_ids,
                                                                                                 stereo_config)
    list_detected_markers_obj = []
    for id in common_ids:
        exec('DetectedAruco_id_{} = DetectedArUcoMarker_world(allMarkers[id], id)'.format(id))
        exec('list_detected_markers_obj.append(DetectedAruco_id_{})'.format(id))
    all_list_detected_markers_obj.append(list_detected_markers_obj)
    A, B, C = DetectedArUcoMarker_world.plane_from_least_square(allCorners)
    coefficient = np.array([A, B, C], dtype='f2')
    print(coefficient)
    plane_coefficients = np.row_stack((plane_coefficients, coefficient))


plane_coefficients = plane_coefficients[1:, :]  # delete the initial line [0, 0, 0]

# get rid of abnormal
array_of_deviations = abs(plane_coefficients - plane_coefficients.mean(axis=0))
line_index = 0
for line in array_of_deviations:
    # TODO: 这里可以改成try，从小的数目开始，逐渐增加阈值
    bool_array = line > 2.3  # scale of abnormal
    if bool_array.any():
        array_of_deviations = np.delete(array_of_deviations, line_index, axis=0)
        plane_coefficients = np.delete(plane_coefficients, line_index, axis=0)
        del all_list_detected_markers_obj[line_index]
        line_index -= 1
    line_index += 1

A, B, C = tuple(plane_coefficients.mean(axis=0).tolist())


verts, edges, faces = DetectedArUcoMarker_world.vertices_from_plane(A, B, C)
surfaceName = StereoPointObject.addSurface(verts, edges, faces, surfaceName='reference')

for i in cameraList:
    bpy.data.cameras.remove(bpy.data.cameras[i['name']])

## select reference surface and create orientation
bpy.context.scene.objects["reference"].select_set(True)
bpy.context.view_layer.objects.active = bpy.context.scene.objects["reference"]
bpy.ops.object.editmode_toggle()
areas = [area for area in bpy.context.window.screen.areas if area.type == 'VIEW_3D']
if areas:
    override = {'area': areas[0]}
    bpy.ops.transform.create_orientation(override, name="Reference", use=True)

## toggle edit mode and change Transformation Orientation to the custom orientation('Reference')
bpy.ops.object.editmode_toggle()
bpy.context.scene.transform_orientation_slots[0].type = 'Reference'

## Align local coordinate system of textruedMesh to the custom orientation
bpy.context.scene.objects["texturedMesh"].select_set(True)
bpy.context.scene.tool_settings.use_transform_data_origin = True
bpy.ops.transform.transform(mode='ALIGN', orient_type='Reference')
bpy.context.scene.tool_settings.use_transform_data_origin = False

## Eliminate initial orientation
bpy.ops.object.rotation_clear(clear_delta=False)

bpy.ops.object.select_all(action='DESELECT')
bpy.context.scene.objects["reference"].select_set(True)
bpy.ops.object.delete()

## Rescale
bpy.context.scene.objects["texturedMesh"].select_set(True)

rescale_factors = 0
loop_counter = 0
for list_detected_markers_obj in all_list_detected_markers_obj:
    rescale_factor = 0
    for i in list_detected_markers_obj:
        rescale_factor += i.rescale_factor
    rescale_factor = rescale_factor / len(list_detected_markers_obj)
    rescale_factors += rescale_factor
    loop_counter += 1

rescale_factor = rescale_factors / loop_counter
bpy.ops.transform.resize(value=(rescale_factor, rescale_factor, rescale_factor))
bpy.ops.object.select_all(action='DESELECT')

## Verify direction
view_layer = bpy.context.view_layer
camera_data_verify1 = bpy.data.cameras.new(name='verifyCamera1')
camera_object_verify1 = bpy.data.objects.new(name='verifyCamera1', object_data=camera_data_verify1)
view_layer.active_layer_collection.collection.objects.link(camera_object_verify1)
camera_object_verify1.location = (0, 0, 1)
camera_data_verify2 = bpy.data.cameras.new(name='verifyCamera2')
camera_object_verify2 = bpy.data.objects.new(name='verifyCamera2', object_data=camera_data_verify2)
view_layer.active_layer_collection.collection.objects.link(camera_object_verify2)
camera_object_verify2.location = (0, 0, -1)

camera_object_verify2.select_set(True)
view_layer.objects.active = camera_object_verify2
bpy.ops.object.constraint_add(type='TRACK_TO')
bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'
bpy.ops.object.select_all(action='DESELECT')

BlenderCameraOperation.render_through_camera('verifyCamera1')
verifyCamera_img1 = ImageTransformProcess.readImageBIN(work_dir + 'verifyCamera1.png')
BlenderCameraOperation.render_through_camera('verifyCamera2')
verifyCamera_img2 = ImageTransformProcess.readImageBIN(work_dir + 'verifyCamera2.png')
verify_corners1, verify_ids1, _ = ArucoInfoDetection.detect_save_aruco_info_image('verifyCamera1', verifyCamera_img1)
verify_corners2, verify_ids2, _ = ArucoInfoDetection.detect_save_aruco_info_image('verifyCamera2', verifyCamera_img2)

if len(verify_corners1) < len(verify_corners2):
    bpy.context.scene.objects["texturedMesh"].select_set(True)
    bpy.ops.transform.rotate(value=math.pi, orient_axis='Y')

## delete unnesessary infomation
bpy.context.view_layer.objects.active = bpy.context.scene.objects["texturedMesh"]
obj = bpy.context.active_object
bpy.ops.object.mode_set(mode='OBJECT')
me = bpy.context.active_object.data
for face in me.polygons:
    face.select = False

for edge in me.edges:
    edge.select = False

for vert in me.vertices:
    vert.select = False

for vert in me.vertices:
    if (sqrt(vert.co.x ** 2 + vert.co.y ** 2) > 0.50) or ((vert.co.z < -0.63) or (vert.co.z > -0.015)):
        vert.select = True

bpy.ops.object.select_all(action='DESELECT')
BlenderCameraOperation.debug_vertices(me.vertices)

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.delete(type='VERT')
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.export_scene.obj(filepath=output_dir, axis_forward='-Z', axis_up='Y')

bpy.ops.object.select_all(action='DESELECT')
obj = bpy.data.objects['texturedMesh']
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

image_name = obj.name + '_BakedTexture'
img = bpy.data.images.new(image_name, 4096, 4096)

# Due to the presence of any multiple materials, it seems necessary to iterate on all the materials, and assign them a node + the image to bake.
for mat in obj.data.materials:
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.name = 'Bake_node'
    texture_node.select = True
    nodes.active = texture_node
    texture_node.image = img  # Assign the image to the node

for s in bpy.data.screens:
    for a in s.areas:
        if a.type == 'VIEW_3D':
            a.spaces[0].shading.use_scene_world_render = False
            a.spaces[0].shading.use_scene_lights_render = True

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.view_layers[0].cycles.use_denoising = True
scene.cycles.samples = 20
scene.cycles.preview_sample = 0
scene.render.resolution_percentage = 100
scene.render.use_border = False
scene.render.bake.use_pass_direct = False
scene.render.bake.use_pass_indirect = False
scene.render.bake.margin = 1
bpy.context.view_layer.objects.active = obj
bpy.ops.object.bake(type='DIFFUSE', save_mode='EXTERNAL')

img.save_render(filepath='C:\\TEMP\\final_test.png')
