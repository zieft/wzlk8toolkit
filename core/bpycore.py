import math
from math import sin, cos
import bpy

def generateDomeCoor(numVer, numHor, radius):
    """
    Generate coordinates of points landing on a sphere with a given radius.
    :param numVer: int
    :param numHor: int
    :param radius: float
    :return: list of coordinates.
    """
    r = radius

    # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    stepPhi = 2*math.pi / numHor
    stepTheta = 2*math.pi / numVer

    coordinates = []
    for i in range(numHor+1):
        for j in range(numVer+1):
            # x = r * sin(phi) * cos(theta)
            # y = r * sin(phi) * sin(theta)
            # z = r * cos(phi)
            phi = i * stepPhi
            theta = j * stepTheta
            coordinate = (r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi))
            coordinates.append(coordinate)

    cameraNumbers = numHor * numVer

    return coordinates, cameraNumbers

def addCamera(coordinates, cameraNumbers):
    view_layer = bpy.context.view_layer
    cameraList = []
    for i in range(cameraNumbers+1):
        # view_layer = bpy.context.view_layer
        cameraID = i
        cameraName = 'camera{}'.format(cameraID)
        camera_data = bpy.data.cameras.new(name=cameraName)
        camera_object = bpy.data.objects.new(name=cameraName, object_data=camera_data)
        view_layer.active_layer_collection.collection.objects.link(camera_object)
        camera_object.location = coordinates[i]
        camera_object.select_set(True)
        view_layer.objects.active = camera_object
        cameraInfo = {"name": cameraName, "Coordinate": coordinates[i]}
        cameraList.append(cameraInfo)

        bpy.ops.object.constraint_add(type='TRACK_TO')
        bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
        bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
        bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'

    return cameraList

def addLightSources(coordinates, Numbers):
    view_layer = bpy.context.view_layer
    for i in range(Numbers + 1):
        cameraID = i
        lightName = 'light{}'.format(cameraID)
        light_data = bpy.data.lights.new(name=lightName, type="AREA")
        light_object = bpy.data.objects.new(name=lightName, object_data=light_data)
        view_layer.active_layer_collection.collection.objects.link(light_object)
        light_object.location = coordinates[i]
        light_object.select_set(True)
        view_layer.objects.active = light_object

        bpy.ops.object.constraint_add(type='TRACK_TO')
        bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
        bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
        bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'


def renderAllCameras(cameraList):
    scene = bpy.context.scene
    bpy.context.scene.cycles.samples = 1
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    for camera in cameraList:
        scene.camera = bpy.data.objects[camera['name']]
        bpy.data.scenes["Scene"].render.filepath = '/tmp/renders/{}.png'.format(camera['name'])
        bpy.ops.render.render(write_still=True)
