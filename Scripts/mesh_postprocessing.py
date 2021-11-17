# import pydevd
# pydevd.settrace('127.0.0.1', port=1090, stdoutToServer=True, stderrToServer=True)
import platform
if platform.system().lower() == 'windows':
    script_position = r'C:\Users\zieft\PycharmProjects\wzlk8toolkit'
else:
    script_position = '/opt/scripts/wzlk8toolkit'

import bpy
import sys
import os
sys.path.append(os.path.curdir)
os.chdir(script_position)
from core.bpycore import *



print('## Delete initial objects.')
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

print("## Change ViewPoint Shading into 'Rendered'.")
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'RENDERED'


bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))

print('## import .obj file')

bpy.ops.import_scene.obj(filepath=filePath)

print('## set texturedMesh to the world origin')
objectToSelect = bpy.data.objects['texturedMesh']
objectToSelect.select_set(True)
bpy.context.view_layer.objects.active = objectToSelect
bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')


print('# Step 1: Define a sphere')
cameraCoordinates, cameraNumbers = BlenderCameraOperation.generateDomeCoor(10, 10, 4)
lightCoordinates, lightNumbers = BlenderCameraOperation.generateDomeCoor(3, 3, 1)

print('# Step 2: place a camera on the sphere and track the world origin')
cameraList = BlenderCameraOperation.addCamera(cameraCoordinates, cameraNumbers)
BlenderCameraOperation.addLightSources(lightCoordinates, lightNumbers)

print('# Step 3-5:')
for camera in cameraList:
    BlenderCameraOperation.render_through_camera(camera)

numbers_of_markers_detected = 0
best_camera_angle = ''
aruco_info = {}
for camera in cameraList:
    img = ImageTransformProcess.readImageBIN(work_dir+'{}.png'.format(camera['name']))
    try:
        corners, ids, _ = ArucoInfoDetection.detect_save_aruco_info_image(camera, img)
        if ids is not None:
            if len(ids) > numbers_of_markers_detected:
                numbers_of_markers_detected = len(ids)
                best_camera_angle = camera['name']
                aruco_info['corners'] = corners
                aruco_info['ids'] = ids
    except cv2.error:
        pass

print('Selected Camera: {}'.format(best_camera_angle))

print('# Step 6: Move the camera slightly to build a stereo pair')
bpy.ops.object.select_all(action='DESELECT')
best_camera_angle_co = BlenderCameraOperation.add_co_camera(best_camera_angle)
BlenderCameraOperation.render_through_camera(best_camera_angle_co)
co_img = ImageTransformProcess.readImageBIN(work_dir+'{}.png'.format(best_camera_angle_co))
corners_co, ids_co, _ = ArucoInfoDetection.detect_save_aruco_info_image(best_camera_angle_co, co_img)
aruco_info_co = {'corners': corners_co, 'ids':ids_co}


print('# Step 7: Calculate the coordinates of the markers')
iml = ImageTransformProcess.readImageBIN(work_dir+'{}.png'.format(best_camera_angle), BIN=False)
imr = ImageTransformProcess.readImageBIN(work_dir+'{}.png'.format(best_camera_angle_co), BIN=False)
height, width = iml.shape[0:2]

stereo_config = stereoCamera(best_camera_angle, best_camera_angle_co)

print('## Stereo Rectify')
map1x, map1y, map2x, map2y, Q, cameraRecMat = ImageTransformProcess.getRectifyTransform(height, width, stereo_config)
iml_rectified, imr_rectified = ImageTransformProcess.rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

line = ImageTransformProcess.draw_line(iml_rectified, imr_rectified)
plt.figure()
plt.imshow(line)
plt.savefig(work_dir+'validation.png', dpi=1000)


print('## Stereo Match')
iml_, imr_ = ImageTransformProcess.preprocess(iml, imr)
disp, _ = ImageTransformProcess.stereoMatchSGBM(iml_rectified, imr_rectified, True)
plt.figure()
plt.imshow(disp)
plt.savefig(work_dir+'z_disparity_map.png', dpi=1000)

print('## Reproject image to 3D World')
points_3d = cv2.reprojectImageTo3D(disp, Q)
points_3d[:,:,1:3] = -points_3d[:,:, 1:3] # y, z direction in OpenCV and Blender are different

## New algrithm
aruco_info_better = ArucoInfoDetection.better_aruco_info(aruco_info)
aruco_info_co_better = ArucoInfoDetection.better_aruco_info(aruco_info_co)

aruco_info_common, aruco_info_co_common, common_ids = DetectedArUcoMarker_world.detected_markers_common(aruco_info_better, aruco_info_co_better)

allMarkers, allCorners = StereoPointObject.generate_dict_of_stereo_point_pairs_obj_allMarker(aruco_info_common, aruco_info_co_common, common_ids ,stereo_config)

list_detected_markers_obj = []
for id in common_ids:
    exec('DetectedAruco_id_{} = DetectedArUcoMarker_world(allMarkers[id], id)'.format(id))
    exec('list_detected_markers_obj.append(DetectedAruco_id_{})'.format(id))


verts, edges, faces = DetectedArUcoMarker_world.plane_from_all_corners(allCorners)
surfaceName = StereoPointObject.addSurface(verts, edges, faces, surfaceName='reference')

for i in cameraList:
    bpy.data.cameras.remove(bpy.data.cameras[i['name']])

print('## select reference surface and create orientation')

bpy.context.scene.objects["reference"].select_set(True)
bpy.context.view_layer.objects.active = bpy.context.scene.objects["reference"]
bpy.ops.object.editmode_toggle()
areas = [area for area in bpy.context.window.screen.areas if area.type == 'VIEW_3D']
if areas:
    override = {'area' : areas[0]}
    bpy.ops.transform.create_orientation(override, name="Reference", use=True)

print("## toggle edit mode and change Transformation Orientation to the custom orientation('Reference')")
bpy.ops.object.editmode_toggle()
bpy.context.scene.transform_orientation_slots[0].type = 'Reference'

print('## Align local coordinate system of texturedMesh to the custom orientation')
bpy.context.scene.objects["texturedMesh"].select_set(True)
bpy.context.scene.tool_settings.use_transform_data_origin = True
bpy.ops.transform.transform(mode='ALIGN', orient_type='Reference')
bpy.context.scene.tool_settings.use_transform_data_origin = False

print('## Eliminate initial orientation')
bpy.ops.object.rotation_clear(clear_delta=False)

bpy.ops.object.select_all(action='DESELECT')
bpy.context.scene.objects["reference"].select_set(True)
bpy.ops.object.delete()

print('## Rescale')
bpy.context.scene.objects["texturedMesh"].select_set(True)
rescale_factor = 0
for i in list_detected_markers_obj:
    rescale_factor += i.rescale_factor

rescale_factor = rescale_factor / len(list_detected_markers_obj)
bpy.ops.transform.resize(value=(rescale_factor, rescale_factor, rescale_factor))
bpy.ops.object.select_all(action='DESELECT')


print('## Verify direction')
view_layer = bpy.context.view_layer
camera_data_verify = bpy.data.cameras.new(name='verifyCamera')
camera_object_verify = bpy.data.objects.new(name='verifyCamera', object_data=camera_data_verify)
view_layer.active_layer_collection.collection.objects.link(camera_object_verify)
camera_object_verify.location = (0, 0, 1)
BlenderCameraOperation.render_through_camera('verifyCamera')
verifyCamera_img = ImageTransformProcess.readImageBIN(work_dir+'verifyCamera.png')
verify_corners, verify_ids, _ = ArucoInfoDetection.detect_save_aruco_info_image('verifyCamera', verifyCamera_img)
if verify_corners ==[]:
    bpy.context.scene.objects["texturedMesh"].select_set(True)
    bpy.ops.transform.rotate(value=math.pi, orient_axis='Y')

print('## delete unnesessary infomation')
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
    if (sqrt(vert.co.x ** 2 + vert.co.y ** 2) > 0.50) or ((vert.co.z < -0.90) or (vert.co.z > 0.24)):
        vert.select = True

bpy.ops.object.select_all(action='DESELECT')
BlenderCameraOperation.debug_vertices(me.vertices)

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.delete(type='VERT')
bpy.ops.object.mode_set(mode='OBJECT')

print('## Export wavefront obj file.')


bpy.ops.export_scene.obj(filepath=output_dir, axis_forward='-Z', axis_up='Y')
