# import pydevd
# pydevd.settrace('127.0.0.1', port=1090, stdoutToServer=True, stderrToServer=True)

import bpy
import sys
import os
sys.path.append(os.path.curdir)
os.chdir('/home/yulin/PycharmProjects/wzlk8toolkit')
from core.bpycore import *


# Delete initial objects.
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

# Change ViewPoint Shading into 'Rendered'.
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'RENDERED'


bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))

# import .obj file
filePath = "/home/yulin/Desktop/ArUcoTest1/texturedMesh.obj"
bpy.ops.import_scene.obj(filepath=filePath)

# set texturedMesh to the world origin
objectToSelect = bpy.data.objects['texturedMesh']
objectToSelect.select_set(True)
bpy.context.view_layer.objects.active = objectToSelect
bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')


cameraCoordinates, cameraNumbers = generateDomeCoor(10, 10, 4)
lightCoordinates, lightNumbers = generateDomeCoor(3, 3, 1)


cameraList = addCamera(cameraCoordinates, cameraNumbers)
addLightSources(lightCoordinates, lightNumbers)


for camera in cameraList:
    render_through_camera(camera)



numbers_of_markers_detected = 0
best_camera_angle = ''
for camera in cameraList:
    img = readImageBIN('/tmp/renders/{}.png'.format(camera['name']))
    try:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        plt.figure()
        plt.imshow(frame_markers)
        if ids is not None:
            if len(ids) > numbers_of_markers_detected:
                numbers_of_markers_detected = len(ids)
                best_camera_angle = camera['name']
            for i in range(len(ids)):
                c = corners[i][0]
                plt.plot([c[:, 0].mean()], [c[:, 1].mean()], label="id={0}".format(ids[i]))
            plt.legend()
            plt.savefig('/tmp/renders/detected_{}.png'.format(camera['name']), dpi=1000)
            plt.show()
    except cv2.error:
        pass

print('Selected Camera: {}'.format(best_camera_angle))
# TODO: unselect all before add co camera
bpy.ops.object.select_all(action='DESELECT')
best_camera_angle_co = add_co_camera(best_camera_angle)
render_through_camera(best_camera_angle_co)


iml = readImageBIN('/tmp/renders/{}.png'.format(best_camera_angle), BIN=False)
imr = readImageBIN('/tmp/renders/{}.png'.format(best_camera_angle_co), BIN=False)
height, width = iml.shape[0:2]

stereo_config = stereoCamera(best_camera_angle, best_camera_angle_co)

# 立体校正
map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, stereo_config)
iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
print(Q)

line = draw_line(iml_rectified, imr_rectified)
# cv2.imwrite('/tmp/renders/validation.png', line)
plt.figure()
plt.imshow(line)
plt.savefig('/tmp/renders/validation.png', dpi=1000)


# 立体匹配
iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
disp, _ = stereoMatchSGBM(iml_rectified, imr_rectified, True)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
# cv2.imwrite('/tmp/renders/视差.png', disp)
plt.figure()
plt.imshow(disp)
plt.savefig('/tmp/renders/z_disparity_map.png', dpi=1000)
