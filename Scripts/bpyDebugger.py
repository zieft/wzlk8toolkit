import pydevd
pydevd.settrace('127.0.0.1', port=1090, stdoutToServer=True, stderrToServer=True)

import bpy
from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.curdir)
import core.bpycore


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
# bpy.ops.object.camera_add()

# import matplotlib.pyplot as plt
# from matplotlib import cm, colors
# from mpl_toolkits.mplot3d import Axes3D

# def generateDomeCoor(numVer, numHor, radius):
#     """
#     Generate coordinates of points landing on a sphere with a given radius.
#     :param numVer: int
#     :param numHor: int
#     :param radius: float
#     :return: list of coordinates.
#     """
#     r = radius
#
#     # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
#     stepPhi = 2*math.pi / numHor
#     stepTheta = 2*math.pi / numVer
#
#     coordinates = []
#     for i in range(numHor+1):
#         for j in range(numVer+1):
#             # x = r * sin(phi) * cos(theta)
#             # y = r * sin(phi) * sin(theta)
#             # z = r * cos(phi)
#             phi = i * stepPhi
#             theta = j * stepTheta
#             coordinate = (r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi))
#             coordinates.append(coordinate)
#
#     cameraNumbers = numHor * numVer
#
#     return coordinates, cameraNumbers

cameraCoordinates, cameraNumbers = core.bpycore.generateDomeCoor(10, 10, 4)
lightCoordinates, lightNumbers = core.bpycore.generateDomeCoor(3, 3, 1)

# def addCamera(coordinates, cameraNumbers):
#     view_layer = bpy.context.view_layer
#     cameraList = []
#     for i in range(cameraNumbers+1):
#         # view_layer = bpy.context.view_layer
#         cameraID = i
#         cameraName = 'camera{}'.format(cameraID)
#         camera_data = bpy.data.cameras.new(name=cameraName)
#         camera_object = bpy.data.objects.new(name=cameraName, object_data=camera_data)
#         view_layer.active_layer_collection.collection.objects.link(camera_object)
#         camera_object.location = coordinates[i]
#         camera_object.select_set(True)
#         view_layer.objects.active = camera_object
#         cameraInfo = {"name": cameraName, "Coordinate": coordinates[i]}
#         cameraList.append(cameraInfo)
#
#         bpy.ops.object.constraint_add(type='TRACK_TO')
#         bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
#         bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
#         bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'
#
#     return cameraList

# def addLightSources(coordinates, Numbers):
#     view_layer = bpy.context.view_layer
#     for i in range(Numbers + 1):
#         cameraID = i
#         lightName = 'light{}'.format(cameraID)
#         light_data = bpy.data.lights.new(name=lightName, type="AREA")
#         light_object = bpy.data.objects.new(name=lightName, object_data=light_data)
#         view_layer.active_layer_collection.collection.objects.link(light_object)
#         light_object.location = coordinates[i]
#         light_object.select_set(True)
#         view_layer.objects.active = light_object
#
#         bpy.ops.object.constraint_add(type='TRACK_TO')
#         bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
#         bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
#         bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'


cameraList = core.bpycore.addCamera(cameraCoordinates, cameraNumbers)
core.bpycore.addLightSources(lightCoordinates, lightNumbers)


# def renderAllCameras(cameraList):
#     scene = bpy.context.scene
#     bpy.context.scene.cycles.samples = 1
#     scene.render.resolution_x = 1920
#     scene.render.resolution_y = 1080
#     scene.render.resolution_percentage = 100
#     scene.render.use_border = False
#     for camera in cameraList:
#         scene.camera = bpy.data.objects[camera['name']]
#         bpy.data.scenes["Scene"].render.filepath = '/tmp/renders/{}.png'.format(camera['name'])
#         bpy.ops.render.render(write_still=True)

core.bpycore.renderAllCameras(cameraList)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

for camera in cameraList:
    img = cv2.imread('/tmp/renders/{}.png'.format(camera['name']))
    try:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)

        plt.figure()
        plt.imshow(frame_markers)
        for i in range(len(ids)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], label="id={0}".format(ids[i]))
        plt.legend()
        plt.savefig('/tmp/renders/ArucoDetect/{}.png'.format(camera['name']), dpi=1000)
        plt.show()
    except cv2.error:
        pass

