import bpy
from math import *
import numpy as np
import math
from mathutils import Matrix
from mathutils import Vector
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from matplotlib import pyplot

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_parameters = aruco.DetectorParameters_create()

camera_baseline_translation = (0.1, 0, 0) # 0.1 meter along x axis
work_dir = r'C:\Users\zieft\Desktop\test1\renders\\'


def get_calibration_matrix_K_from_blender(camd):
    """
    Returns the camera Matrix (intrinsics).
    :param camd:
    :return:
    """
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels
    K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1 * R_world2bcam * cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location
    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam
    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],), R_world2cv[1][:] + (T_world2cv[1],), R_world2cv[2][:] + (T_world2cv[2],)))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT


class cameraObject():
    def __init__(self, cameraName):
        self.cameraName = cameraName
        self.location_world = np.array(bpy.data.objects[cameraName].location)
        self.R_world2cam = np.array(bpy.data.objects[cameraName].matrix_world)[:3, :3]
        self.T_world2cam = np.array(bpy.data.objects[cameraName].matrix_world)[:3, 3].reshape((1, 3))
        self.K = np.array(get_calibration_matrix_K_from_blender(bpy.data.objects[cameraName].data), dtype=float)







# camera22 = cameraObject('camera22')

def verify_point_world(point_coor_world):
    bpy.ops.object.light_add(type='SUN', radius=1, location=point_coor_world)


def info_marker_uv(id, aruco_info):
    uv_corner_0 = aruco_info['corners'][id][0][0].astype('int').tolist()
    uv_corner_1 = aruco_info['corners'][id][0][1].astype('int').tolist()
    uv_corner_2 = aruco_info['corners'][id][0][2].astype('int').tolist()
    uv_corner_3 = aruco_info['corners'][id][0][3].astype('int').tolist()
    return uv_corner_0, uv_corner_1, uv_corner_2, uv_corner_3


def point_3d_camera_2_world(point_coor_camera_np, camera_obj):
    point_coor_camera = point_coor_camera_np.reshape((1, 3))
    R_world2cam = camera_obj.R_world2cam
    T_world2cam = camera_obj.T_world2cam
    point_coor_world_np = np.dot(R_world2cam ,point_coor_camera.T) + T_world2cam.T
    point_coor_world_tuple = (point_coor_world_np[0][0], point_coor_world_np[1][0], point_coor_world_np[2][0])
    return point_coor_world_tuple


def corners_uv_2_3d_world(points_3d, aruco_info, id, cameraObj):
    lenth_of_ids = len(aruco_info['ids'])
    which_marker = aruco_info['ids'].reshape((1,lenth_of_ids)).tolist()[0].index(id)
    uv_corner_0 = tuple(aruco_info['corners'][which_marker][0][0].astype('int').tolist())
    uv_corner_1 = tuple(aruco_info['corners'][which_marker][0][1].astype('int').tolist())
    uv_corner_2 = tuple(aruco_info['corners'][which_marker][0][2].astype('int').tolist())
    uv_corner_3 = tuple(aruco_info['corners'][which_marker][0][3].astype('int').tolist())
    corner0_coor_camera = points_3d[uv_corner_0]
    corner1_coor_camera = points_3d[uv_corner_1]
    corner2_coor_camera = points_3d[uv_corner_2]
    corner3_coor_camera = points_3d[uv_corner_3]
    corners_coor_world_tuple_list = [
        point_3d_camera_2_world(corner0_coor_camera, cameraObj),
        point_3d_camera_2_world(corner1_coor_camera, cameraObj),
        point_3d_camera_2_world(corner2_coor_camera, cameraObj),
        point_3d_camera_2_world(corner3_coor_camera, cameraObj)
    ]
    return corners_coor_world_tuple_list

# corners_uv_2_3d_world(points_3d, aruco_info, 6, camera22)

def Point_3d_camera_2_world(point_coor_camera, camera_obj):
    point_coor_camera = point_coor_camera.reshape((1, 3))
    R_world2cam = camera_obj.R_world2cam
    T_world2cam = camera_obj.T_world2cam
    point_coor_world_np = np.dot(R_world2cam ,point_coor_camera.T) + T_world2cam.T
    point_coor_world_tuple = (point_coor_world_np[0][0], point_coor_world_np[1][0], point_coor_world_np[2][0])
    return point_coor_world_tuple


def detect_save_aruco_info_image(camera, img, work_dir=work_dir):
    try:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=aruco_parameters)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        plt.figure()
        plt.imshow(frame_markers)
        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                plt.plot([c[:, 0].mean()], [c[:, 1].mean()], label="id={0}".format(ids[i]))
            plt.legend()
            if isinstance(camera, dict):
                plt.savefig(work_dir + 'detected_{}.png'.format(camera['name']), dpi=1000)
            elif isinstance(camera, str):
                plt.savefig(work_dir + 'detected_{}.png'.format(camera), dpi=1000)
            plt.show()
            plt.close()
    except cv2.error:
        pass
    return corners, ids, rejectedImgPoints

# def verify_aruco_marker(detectedMarker_obj):
#

# class arucoInfo():
#     def __init__(self, ids, cameraName):
#         camera = self.__dict__
#         camera['camera'+id]
#         # self.camera = cameraName
#         # self.number_of_detection = len(ids)
#         # self.ids = ids.tolist
#         # self.info = dict(zip(ids, corners))
#
#     def get_corner_coor_list(self, id):
#         corner = self.info[id]
#         return corner


# class detectedMarker():
#     def __init__(self, aruco_info_obj):
#         names = self.__dict__
#         for i in range(len(aruco_info_obj.ids)):
#             names['Marker_id'+i] = aruco_info_obj.get_corner_coor_list(i)




# Points_3d_camera_2_world(corner1, camera22)

# class remappedPoints():
#     def __init__(self, disp, Q):
#         self.points_coor_cv2 = cv2.reprojectImageTo3D(disp, Q)
#         self.Points_coor_blender = -self.points_coor_cv2[:,:, 1:3]


class stereoCamera(object):
    def __init__(self, camera_left, camera_right):
        self.cam_matrix_left = np.array(get_calibration_matrix_K_from_blender(bpy.data.objects[camera_left].data), dtype=float)
        self.cam_matrix_right = np.array(get_calibration_matrix_K_from_blender(bpy.data.objects[camera_right].data), dtype=float)
        self.distortion_left = np.zeros((1, 5), dtype=float)
        self.distortion_right = np.zeros((1, 5), dtype=float)
        self.R = np.eye(3, dtype=float)
        self.T = np.array([tuple(-ti for ti in camera_baseline_translation)], dtype=float).T
        self.focal_length = 50  # mm
        self.baseline = 0.1 # meter


class PointPairObject():
    def __init__(self, aruco_info_corner_left, aruco_info_corner_right, stereo_camera_obj):
        """

        :param aruco_info_corner: One uv of a marker in aruco_info, ex.: aruco_info['corners'][0][0][0]
        """
        self.uv_np_l = aruco_info_corner_left
        self.uv_np_r = aruco_info_corner_right
        self.uv_tuple_l = tuple(aruco_info_corner_left)
        self.uv_tuple_r = tuple(aruco_info_corner_right)
        self.u_l = aruco_info_corner_left[0]
        self.u_r = aruco_info_corner_right[0]
        self.v_l = aruco_info_corner_left[1]
        self.v_r = aruco_info_corner_right[1]

        self.disparity = self.u_r - self.u_l

        self.focal_length = stereo_camera_obj.focal_length
        self.cam_matrix_l = stereo_camera_obj.cam_matrix_left

        self.coor_cam_np = None
        self.coor_cam_tuple = None
        self.coor_world_np = None
        self.coor_world_tuple = None

    def _set_uv(self, aruco_info_corner_left, aruco_info_corner_right):
        """

        :param aruco_info_corner: One uv of a marker in aruco_info, ex.: aruco_info['corners'][0][0][0]
        :return:
        """
        self.uv_np_l = aruco_info_corner_left
        self.uv_tuple_l = tuple(aruco_info_corner_left)
        self.uv_np_r = aruco_info_corner_right
        self.uv_tuple_r = tuple(aruco_info_corner_right)

    def uv_to_coor_cam(self):
        f_x = self.cam_matrix_l[0][0]


def preprocess(img1, img2):
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


def undistortion(imgae, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(imgae, camera_matrix, dist_coeff)

    return undistortion_image


def getRectifyTransform(height, width, config):
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_left
    right_distortion = config.distortion_right
    R = config.R
    T = config.T

    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q , P1


def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


def draw_line(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    line_interval = 50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM matching parameters
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 128,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # Build SGBM Instants
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # Compute disparity map
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # real disparity
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


def generateDomeCoor(numVer, numHor, radius):
    """
    Generate coordinates of points landing on a sphere with a given radius.
    :param numVer: int
    :param numHor: int
    :param radius: float
    :return: list of coordinates.
    """
    r = radius
    stepPhi = 2*math.pi / numHor
    stepTheta = 2*math.pi / numVer
    coordinates = []
    for i in range(numHor+1):
        for j in range(numVer+1):
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


def render_through_camera(camera):
    scene = bpy.context.scene
    bpy.context.scene.cycles.samples = 1
    scene.render.resolution_x = 1620
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    if isinstance(camera, dict):
        scene.camera = bpy.data.objects[camera['name']]
        bpy.data.scenes["Scene"].render.filepath = work_dir+'{}.png'.format(camera['name'])
    elif isinstance(camera, str):
        scene.camera = bpy.data.objects[camera]
        bpy.data.scenes["Scene"].render.filepath = work_dir+'{}.png'.format(camera)
    bpy.ops.render.render(write_still=True)


def readImageBIN(path, BIN=True):
    """
    cv2.imread() does not perform well inside blender python interpreter.
    This function is meant to solve the problem.
    :param path:
    :return:
    """
    img = plt.imread(path)[:,:,:3]
    img = (img * 255).astype('uint8')
    if BIN == True:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        ret, img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    return img


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x, y, z


def add_co_camera(best_angle_camera):
    # location_left = bpy.data.objects[best_angle_camera].location
    # rotation_matrix_left = np.array(bpy.data.objects[best_angle_camera].matrix_world)[0:3, 0:3]
    RT_Matrix_left = bpy.data.objects[best_angle_camera].matrix_world
    # rotation_left_euler = rotationMatrixToEulerAngles(rotation_matrix_left)
    # bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=location_left, rotation=rotation_left_euler)
    view_layer = bpy.context.view_layer
    co_cameraName = best_angle_camera + '_co'
    co_camera_data = bpy.data.cameras.new(name=co_cameraName)
    co_camera_object = bpy.data.objects.new(name=co_cameraName, object_data=co_camera_data)
    view_layer.active_layer_collection.collection.objects.link(co_camera_object)
    co_camera_object.matrix_world = RT_Matrix_left
    # co_camera_object.location = location_left
    # co_camera_object.rotation_euler = rotation_left_euler
    co_camera_object.select_set(True)
    view_layer.objects.active = co_camera_object
    bpy.ops.transform.translate(value=camera_baseline_translation, orient_type='LOCAL',
                                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                proportional_edit_falloff='SMOOTH', proportional_size=1,
                                use_proportional_connected=False, use_proportional_projected=False)
    return co_cameraName


def hw3ToN3(points):
    height, width = points.shape[0:2]
    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)
    points_ = np.hstack((points_1, points_2, points_3))
    return points_


def addSurface(verts, edges, faces):
    view_layer = bpy.context.view_layer
    mesh = bpy.data.meshes.new('referentPlane')
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    referencePlane = bpy.data.objects.new('surface', mesh)
    view_layer.active_layer_collection.collection.objects.link(referencePlane)


def detect_co_image(best_camera_angle):
    bpy.ops.object.select_all(action='DESELECT')
    best_camera_angle_co = add_co_camera(best_camera_angle)
    render_through_camera(best_camera_angle_co)
    co_img = readImageBIN(work_dir + '{}.png'.format(best_camera_angle_co))
    co_corners, co_ids, _ = detect_save_aruco_info_image(best_camera_angle_co, co_img)
    return co_corners, co_ids