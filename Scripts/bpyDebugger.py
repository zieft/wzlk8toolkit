# import pydevd
# pydevd.settrace('127.0.0.1', port=1090, stdoutToServer=True, stderrToServer=True)

import bpy
import sys
import os
sys.path.append(os.path.curdir)
os.chdir(r'C:\Users\zieft\PycharmProjects\wzlk8toolkit')
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
filePath = r"C:\Users\zieft\Desktop\test1\texturedMesh.obj"
bpy.ops.import_scene.obj(filepath=filePath)

## set texturedMesh to the world origin
objectToSelect = bpy.data.objects['texturedMesh']
objectToSelect.select_set(True)
bpy.context.view_layer.objects.active = objectToSelect
bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')


# Step 1: Define a sphere
cameraCoordinates, cameraNumbers = generateDomeCoor(10, 10, 4)
lightCoordinates, lightNumbers = generateDomeCoor(3, 3, 1)

# Step 2: place a camera on the sphere and track the world origin
cameraList = addCamera(cameraCoordinates, cameraNumbers)
addLightSources(lightCoordinates, lightNumbers)

# Step 3-5:
for camera in cameraList:
    render_through_camera(camera)

numbers_of_markers_detected = 0
best_camera_angle = ''
aruco_info = {}
for camera in cameraList:
    img = readImageBIN(work_dir+'{}.png'.format(camera['name']))
    try:
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=aruco_parameters)
        # frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        # plt.figure()
        # plt.imshow(frame_markers)
        corners, ids, _ = detect_save_aruco_info_image(camera, img)
        if ids is not None:
            if len(ids) > numbers_of_markers_detected:
                numbers_of_markers_detected = len(ids)
                best_camera_angle = camera['name']
                aruco_info['corners'] = corners
                aruco_info['ids'] = ids
                # co_corners, co_ids = detect_co_image(best_camera_angle)
                # if len(ids)-len(co_ids) > 0:
                #     numbers_of_markers_detected -= 1
            # for i in range(len(ids)):
            #     c = corners[i][0]
            #     plt.plot([c[:, 0].mean()], [c[:, 1].mean()], label="id={0}".format(ids[i]))
            # plt.legend()
            # plt.savefig(work_dir+'detected_{}.png'.format(camera['name']), dpi=1000)
            # plt.show()
    except cv2.error:
        pass

print('Selected Camera: {}'.format(best_camera_angle))

# Step 6: Move the camera slightly to build a stereo pair
bpy.ops.object.select_all(action='DESELECT')
best_camera_angle_co = add_co_camera(best_camera_angle)
render_through_camera(best_camera_angle_co)
co_img = readImageBIN(work_dir+'{}.png'.format(best_camera_angle_co))
corners_co, ids_co, _ = detect_save_aruco_info_image(best_camera_angle_co, co_img)
aruco_info_co = {'corners': corners_co, 'ids':ids_co}


# Step 7: Calculate the coordinates of the markers
iml = readImageBIN(work_dir+'{}.png'.format(best_camera_angle), BIN=False)
imr = readImageBIN(work_dir+'{}.png'.format(best_camera_angle_co), BIN=False)
height, width = iml.shape[0:2]

stereo_config = stereoCamera(best_camera_angle, best_camera_angle_co)

## Stereo Rectify
map1x, map1y, map2x, map2y, Q, cameraRecMat = getRectifyTransform(height, width, stereo_config)
iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
print(Q)

line = draw_line(iml_rectified, imr_rectified)
plt.figure()
plt.imshow(line)
plt.savefig(work_dir+'validation.png', dpi=1000)


## Stereo Match
iml_, imr_ = preprocess(iml, imr)
disp, _ = stereoMatchSGBM(iml_rectified, imr_rectified, True)
plt.figure()
plt.imshow(disp)
plt.savefig(work_dir+'z_disparity_map.png', dpi=1000)

## Reproject image to 3D World
points_3d = cv2.reprojectImageTo3D(disp, Q)
points_3d[:,:,1:3] = -points_3d[:,:, 1:3] # y, z direction in OpenCV and Blender are different

## New algrithm
aruco_info_better = better_aruco_info(aruco_info)
aruco_info_co_better = better_aruco_info(aruco_info_co)

aruco_info_common, aruco_info_co_common, common_ids = detected_markers_common(aruco_info_better, aruco_info_co_better)

allMarkers = generate_dict_of_stereo_point_pairs_obj_allMarker(aruco_info_common, aruco_info_co_common, common_ids ,stereo_config)

list_detected_markers_obj = []
for id in common_ids:
    exec('DetectedAruco_id_{} = DetectedArUcoMarker_world(allMarkers[id], id)'.format(id))
    exec('list_detected_markers_obj.append(DetectedAruco_id_{})'.format(id))


# for area in bpy.context.screen.areas:
#     if area.type == 'VIEW_3D':
#         ctx = bpy.context.copy()
#         ctx['area'] = area
#         ctx['region'] = area.regions[-1]
#         # TODO: select camera
#         bpy.ops.view3d.view_selected(ctx)
#         bpy.ops.view3d.snap_cursor_to_selected(ctx)
#
#
# for i in len(aruco_info['ids']):
#     u, v = tuple(aruco_info['corners'][i][0][0].astype(int))
#     coordinate_of_uv = tuple(points_3d[u, v, :])
#