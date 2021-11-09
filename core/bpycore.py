import bpy
from math import *
import numpy as np
import math
from mathutils import Matrix
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_parameters = aruco.DetectorParameters_create()

camera_baseline_translation = (0.5, 0, 0) # 0.5 meter along x axis
work_dir = r'C:\Users\zieft\Desktop\test1\renders\\'
# work_dir = '/tmp/blenderOutput/renders/'
a = 1

class CameraMatrixFromBlender:
    @staticmethod
    def get_calibration_matrix_K_from_blender(camd):
        """
        Returns the camera Matrix (intrinsics) of the virtual camera placed in Blender.
        :param camd: blender camera object data, e.g.: bpy.data.objects['cameraName'].data
        :return: Camera Matrix K
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
            s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
            s_v = resolution_y_in_px * scale / sensor_height_in_mm

        else:  # 'HORIZONTAL' and 'AUTO'
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

    @staticmethod
    def get_3x4_RT_matrix_from_blender(cam):
        """
        Get the transformation matrix which describe the translation and rotation of the camera, world to blender cam.
        :param cam: Blender Camera Object, e.g.: bpy.data.objects['cameraName']
        :return: Transformation matrix, Numpy Array
        """
        R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

        T_world2bcam = -1 * R_world2bcam @ location
        R_world2cv = R_bcam2cv @ R_world2bcam
        T_world2cv = R_bcam2cv @ T_world2bcam

        RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],), R_world2cv[1][:] + (T_world2cv[1],), R_world2cv[2][:] + (T_world2cv[2],)))

        return RT

    @staticmethod
    def get_3x4_P_matrix_from_blender(cam):
        """
        Return the P matrix from multiply K and RT.
        :param cam: Blender Camera Object. e.g.: bpy.data.objects['cameraName']
        :return: Transformation Matrix, NumPy array.
        """
        K = CameraMatrixFromBlender.get_calibration_matrix_K_from_blender(cam.data)
        RT = CameraMatrixFromBlender.get_3x4_RT_matrix_from_blender(cam)

        return K @ RT


class cameraObject:
    def __init__(self, cameraName):
        self.cameraName = cameraName
        self.location_world = np.array(bpy.data.objects[cameraName].location)
        self.R_world2cam = np.array(bpy.data.objects[cameraName].matrix_world)[:3, :3]
        self.T_world2cam = np.array(bpy.data.objects[cameraName].matrix_world)[:3, 3].reshape((1, 3))
        self.K = np.array(CameraMatrixFromBlender.get_calibration_matrix_K_from_blender(bpy.data.objects[cameraName].data), dtype=float)


class ArucoInfoDetection:
    @staticmethod
    def info_marker_uv(id_pos, aruco_info):
        """
        Get uv coordinates of the marker corners using index(not the actual id).
        :param id_pos: int, index of the marker, not the actuall id.
        :param aruco_info: list, a list stored all the information of detected marker.
        :return: tuple, A tuple of uv coordinates (list: [u, v]) of all 4 corners of an ArUco marker.
        """
        # for i in range(4):
        #     exec("uv_corner_{} = aruco_info['corners'][id_pos][0][{}].astype('int').tolist()".format(i, i))
        uv_corner_0 = aruco_info['corners'][id_pos][0][0].astype('int').tolist()
        uv_corner_1 = aruco_info['corners'][id_pos][0][1].astype('int').tolist()
        uv_corner_2 = aruco_info['corners'][id_pos][0][2].astype('int').tolist()
        uv_corner_3 = aruco_info['corners'][id_pos][0][3].astype('int').tolist()

        return uv_corner_0, uv_corner_1, uv_corner_2, uv_corner_3

    @staticmethod
    def detect_save_aruco_info_image(camera, img, work_dir=work_dir):
        """
        Detect aruco markers from a given image. The boundary of the markers will be outlined, and result will be saved
        into a new image.
        :param camera: str, camera Name. OR, dict, unpack from cameraList
        :param img: opened img through cv2.imread() or through ImageTransformProcess.readImage()
        :param work_dir: Work directory.
        :return: tuple, tuple of id and coordinates of corners in UV format.
        """
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

    @staticmethod
    def better_aruco_info(aruco_info):
        """
        Reconstruct the aruco_info to a more logical structure.
        # TODO: aruco_info and this function should be reconstructed to a more logical one
        :param aruco_info: dict, a dict contain ids and uv coordinates of corners, but unpaired.
        :return: dict, a dict, keys are IDs and values are uv coordinates of all corners
        """
        corners = aruco_info['corners']
        ids = aruco_info['ids']
        ids_list = ids.reshape(len(ids)).tolist()
        ids_list = [str(i) for i in ids_list]
        corners_list = []

        for i in range(len(corners)):
            one_corner = corners[i].reshape(4, 2).tolist()
            corners_list.append(one_corner)
        better = dict(zip(ids_list, corners_list))

        return better


class stereoCamera:
    def __init__(self, camera_left, camera_right):
        self.T_world2cam_l = np.array(bpy.data.objects[camera_left].matrix_world)[:3, 3].reshape((1, 3))
        self.R_world2cam_l = np.array(bpy.data.objects[camera_left].matrix_world)[:3, :3]
        self.cam_left_name = bpy.data.objects[camera_left].name

        self.cam_matrix_left = np.array(CameraMatrixFromBlender.get_calibration_matrix_K_from_blender(bpy.data.objects[camera_left].data), dtype=float)
        self.cam_matrix_right = np.array(CameraMatrixFromBlender.get_calibration_matrix_K_from_blender(bpy.data.objects[camera_right].data), dtype=float)
        self.distortion_left = np.zeros((1, 5), dtype=float)
        self.distortion_right = np.zeros((1, 5), dtype=float)

        self.R = np.eye(3, dtype=float)
        self.T = np.array([tuple(-ti for ti in camera_baseline_translation)], dtype=float).T

        self.focal_length = 50  # mm
        self.baseline = camera_baseline_translation[0] # meter


class StereoPointObject:
    def __init__(self, aruco_info_corner_left, aruco_info_corner_right, stereo_camera_obj):
        """
        Same physical point presented in a stereo camera set.
        :param aruco_info_corner_left: One uv of a marker in aruco_info detected by left camera, ex.: aruco_info['corners'][0][0][0]
        :param aruco_info_corner_right: same as above, but detected by the right camera.
        :param stereo_camera_obj: An instance of Class stereoCamera
        """
        self.uv_np_l = aruco_info_corner_left
        self.uv_np_r = aruco_info_corner_right
        self.uv_tuple_l = tuple(aruco_info_corner_left)
        self.uv_tuple_r = tuple(aruco_info_corner_right)
        self.u_l = aruco_info_corner_left[0]
        self.u_r = aruco_info_corner_right[0]
        self.v_l = aruco_info_corner_left[1]
        self.v_r = aruco_info_corner_right[1]

        self.disparity = self.u_l - self.u_r

        self.focal_length = stereo_camera_obj.focal_length
        self.cam_matrix_l = stereo_camera_obj.cam_matrix_left
        self.stereo_baseline = stereo_camera_obj.baseline # no need to change m to mm
        self.c_R_w = stereo_camera_obj.R_world2cam_l
        self.w_r_world2cam = stereo_camera_obj.T_world2cam_l

        self.coor_cam_np = self.__uv_to_coor_cam()
        self.coor_cam_tuple = tuple(self.__uv_to_coor_cam())
        self.coor_world_np = self.__coor_cam_2_world()
        self.coor_world_tuple = tuple(self.__coor_cam_2_world()[0])

    def __set_uv(self, aruco_info_corner_left, aruco_info_corner_right):
        """
        Set uv coordinate manually, debug only.
        :param aruco_info_corner: One uv of a marker in aruco_info, ex.: aruco_info['corners'][0][0][0]
        """
        self.uv_np_l = aruco_info_corner_left
        self.uv_tuple_l = tuple(aruco_info_corner_left)
        self.uv_np_r = aruco_info_corner_right
        self.uv_tuple_r = tuple(aruco_info_corner_right)

    def __uv_to_coor_cam(self):
        """
        Transform uv (2D) coordinate from image plane, to camera coordinate system (3D).
        :return: NumPy array, 3D coordinate of a point detected in 2D image plane.
        """
        f_x = self.cam_matrix_l[0][0]  # verified
        b = self.stereo_baseline
        d = self.disparity
        u_0 = self.cam_matrix_l[0][2]
        v_0 = self.cam_matrix_l[1][2]

        X_c = b * (self.u_l - u_0) / d
        Y_c = b * (self.v_l - v_0) / d
        Z_c = b * f_x / d

        return np.array([X_c, -Y_c, -Z_c]) # direction of y & z coordinates defined in blender and openCV are opposite

    def __coor_cam_2_world(self):
        """
        Transformation of a 3D point represented in camera system to 3D world system in Blender.
        :return: NumPy array, 3D coordinate of a point detected in 2D image plane.
        """
        coor_world_np = np.dot(self.c_R_w, self.coor_cam_np) + self.w_r_world2cam

        return coor_world_np

    @staticmethod
    def __point_3d_camera_2_world(point_coor_camera_np, camera_obj):
        """
        static method of __coor_cam_2_world()
        :param point_coor_camera_np: NumPy array, 3D coordinate in camera system
        :param camera_obj: Instance of cameraObject class.
        :return: tuple, tuple of the 3D coordinate in world system Blender.
        """
        point_coor_camera = point_coor_camera_np.reshape((1, 3))
        R_world2cam = camera_obj.R_world2cam
        T_world2cam = camera_obj.T_world2cam
        point_coor_world_np = np.dot(R_world2cam, point_coor_camera.T) + T_world2cam.T
        point_coor_world_tuple = (point_coor_world_np[0][0], point_coor_world_np[1][0], point_coor_world_np[2][0])

        return point_coor_world_tuple

    @staticmethod
    def corners_uv_2_3d_world(points_3d, aruco_info, id, cameraObj):
        """
        Transform UV to 3D coordinate in world system
        :param points_3d: NumPy array with shape ((resolution, 3), returned by cv2.reprojectimageto3d()
        :param aruco_info: dict
        :param id: int
        :param cameraObj: instance of class CameraObject
        :return:
        """
        lenth_of_ids = len(aruco_info['ids'])
        which_marker = aruco_info['ids'].reshape((1, lenth_of_ids)).tolist()[0].index(id)
        uv_corner_0 = tuple(aruco_info['corners'][which_marker][0][0].astype('int').tolist())
        uv_corner_1 = tuple(aruco_info['corners'][which_marker][0][1].astype('int').tolist())
        uv_corner_2 = tuple(aruco_info['corners'][which_marker][0][2].astype('int').tolist())
        uv_corner_3 = tuple(aruco_info['corners'][which_marker][0][3].astype('int').tolist())
        corner0_coor_camera = points_3d[uv_corner_0]
        corner1_coor_camera = points_3d[uv_corner_1]
        corner2_coor_camera = points_3d[uv_corner_2]
        corner3_coor_camera = points_3d[uv_corner_3]
        corners_coor_world_tuple_list = [
            StereoPointObject.__point_3d_camera_2_world(corner0_coor_camera, cameraObj),
            StereoPointObject.__point_3d_camera_2_world(corner1_coor_camera, cameraObj),
            StereoPointObject.__point_3d_camera_2_world(corner2_coor_camera, cameraObj),
            StereoPointObject.__point_3d_camera_2_world(corner3_coor_camera, cameraObj)
        ]

        return corners_coor_world_tuple_list

    @staticmethod
    def Point_3d_camera_2_world(point_coor_camera, camera_obj):
        """
        Transform the 3d coordinate in camera system to world system.
        :param point_coor_camera: NumPy array
        :param camera_obj: instance of CameraObject
        :return: tuple, coordinates of the point in world system
        """
        point_coor_camera = point_coor_camera.reshape((1, 3))
        R_world2cam = camera_obj.R_world2cam
        T_world2cam = camera_obj.T_world2cam
        point_coor_world_np = np.dot(R_world2cam, point_coor_camera.T) + T_world2cam.T
        point_coor_world_tuple = (point_coor_world_np[0][0], point_coor_world_np[1][0], point_coor_world_np[2][0])

        return point_coor_world_tuple

    @staticmethod
    def generate_dict_of_stereo_point_pairs_obj_allMarker(aruco_info_common, aruco_info_co_common, common_ids,
                                                          stereo_config):
        """
        Reconstruct all corners coordinates in Blender World System to a special shape in order to feed to the
        DetectedArUcoMarker_world.plane_from_all_corners() method.
        :param aruco_info_common: dict, returned by DetectedArUcoMarker_world.detected_markers_common()
        :param aruco_info_co_common: dict, returned by DetectedArUcoMarker_world.detected_markers_common()
        :param common_ids: set, returned by DetectedArUcoMarker_world.detected_markers_common()
        :param stereo_config: instance of Class stereoCamera.
        :return:
        """
        dict = {}
        list_of_all_corners = []

        for i in common_ids:
            list = []
            for j in range(4):
                stereo_corner_pair_obj = StereoPointObject(aruco_info_common[i][j], aruco_info_co_common[i][j],
                                                           stereo_config)
                exec('corner_{}_of_marker_{} = stereo_corner_pair_obj'.format(j, i))
                exec('list.append(corner_{}_of_marker_{})'.format(j, i))
                exec('list_of_all_corners.append(corner_{}_of_marker_{}.coor_world_np)'.format(j, i))
            dict[i] = list

        shape = (len(list_of_all_corners), 3)

        return dict, np.array(list_of_all_corners).reshape(shape)

    @staticmethod
    def addSurface(verts, edges, faces, surfaceName='surface'):
        """
        Adding a surface to the blender world according given point. The format is under Wavefront guideline.
        :param verts: list of coordinates of vertices. e.g [(1,2,3), (4,5,6)]
        :param edges: list of index(of vertices) pairs, e.g [(0,1)] means an edge built by vertex[0] and vertex[1]
        :param faces: list of index group,the number of indices in each group always > 2.
        :param surfaceName: str, custom name of the created surface in Blender, by default: 'surface'
        :return: str, surfaceName.
        """
        view_layer = bpy.context.view_layer
        mesh = bpy.data.meshes.new('referentPlane')
        mesh.from_pydata(verts, edges, faces)
        mesh.update()
        referencePlane = bpy.data.objects.new(surfaceName, mesh)
        view_layer.active_layer_collection.collection.objects.link(referencePlane)

        return surfaceName


class DetectedArUcoMarker_world:
    def __init__(self, list_of_stereo_point_pairs_obj_1marker, id):
        """

        :param list_of_stereo_point_pairs_obj_1marker:
        :param id:
        """
        self.id = id
        self.verts = [list_of_stereo_point_pairs_obj_1marker[i].coor_world_tuple for i in range(4)]
        self.verts_np = [list_of_stereo_point_pairs_obj_1marker[i].coor_world_np for i in range(4)]
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        self.faces = [(0, 1, 2, 3)]
        self.surface = None
        self.marker_size = self.__get_marker_size()
        self.av_edge_length = self.__get_av_length_of_edges()
        self.rescale_factor = self.__cal_rescale_factor()

    @staticmethod
    def detected_markers_common(better_aruco_info, better_aruco_info_co):
        """
        Find the common markers which are detected by both cameras.
        :param better_aruco_info: dict, returned by ArucoInfoDetect.better_aruco_info() of the left camera.
        :param better_aruco_info_co: dict, returned by ArucoInfoDetect.better_aruco_info() of the right camera.
        :return: UV coordinates of all corners in the common detected markers in dict format, and a set of common IDs.
        """
        common_ids = better_aruco_info.keys() & better_aruco_info_co.keys()
        print("detected markers in both pictures are: ", common_ids)
        aruco_info_common = {}
        aruco_info_co_common = {}

        for i in common_ids:
            aruco_info_common[i] = better_aruco_info[i]
            aruco_info_co_common[i] = better_aruco_info_co[i]

        return aruco_info_common, aruco_info_co_common, common_ids

    def __generate_surface(self):
        """
        Debug only,gererate a (same size) surface to cover up the given(detected) marker.
        :return: str, surface name of the surface object created in Blender.
        """
        view_layer = bpy.context.view_layer
        mesh = bpy.data.meshes.new('referentPlane')
        mesh.from_pydata(self.verts, self.edges, self.faces)
        mesh.update()
        referencePlane = bpy.data.objects.new('surface_marker_{}'.format(self.id), mesh)
        view_layer.active_layer_collection.collection.objects.link(referencePlane)

        return 'surface_marker_{}'.format(self.id)

    def __get_marker_size(self):
        """
        The markers used in the lab have 2 kind of size. Markers with ID 7, 8 and 9 are 20mm by 20mm,
        while 4, 5 and 6 are 10mm by 10mm.
        :return: The physical marker size (target value) of the given (detected) marker in meter.
        """
        if self.id in {'7', '8', '9'}:
            marker_size = 20 / 1000 # meter
        elif self.id in {'4', '5', '6'}:
            marker_size = 10 / 1000 # meter
        else:
            raise RuntimeError("No marker detected!")

        return marker_size

    def __get_av_length_of_edges(self):
        """
        Calculate the average length of all 4 edges of the given (detected) marker.
        :return: float, length of the edge in meter (actual value)
        """
        dist_list = []
        for i in range(-1, 3):
            dist = np.linalg.norm(self.verts_np[i] - self.verts_np[i + 1])
            dist_list.append(dist)

        return sum(dist_list) / len(dist_list)

    def __cal_rescale_factor(self):
        """
        Calculation of the rescale factor, which can be used in bpy.ops.transform.resize() to rescale the Model to the
        target (real) size.
        :return: float
        """
        return self.marker_size / self.av_edge_length

    @staticmethod
    def plane_from_all_corners(allCorners, show_plot=False):
        """
        Least square method to find a plane according to given points.
        :param allCorners: NumPy array, with the shape (number of corners, 3)
        :param show_plot: plot the diagram of the plane, debug only.
        :return: list of vertices, edges and faces. Can be feed to StereoPointObject.addSurface() method.
        """
        num_corners = len(allCorners)
        A = np.zeros((3, 3))
        x = allCorners[:, 0]
        y = allCorners[:, 1]
        z = allCorners[:, 2]

        for i in range(0, num_corners):
            A[0, 0] = A[0, 0] + x[i] ** 2
            A[0, 1] = A[0, 1] + x[i] * y[i]
            A[0, 2] = A[0, 2] + x[i]
            A[1, 0] = A[0, 1]
            A[1, 1] = A[1, 1] + y[i] ** 2
            A[1, 2] = A[1, 2] + y[i]
            A[2, 0] = A[0, 2]
            A[2, 1] = A[1, 2]
            A[2, 2] = num_corners

        b = np.zeros((3, 1))
        for i in range(0, num_corners):
            b[0, 0] = b[0, 0] + x[i] * z[i]
            b[1, 0] = b[1, 0] + y[i] * z[i]
            b[2, 0] = b[2, 0] + z[i]

        A_inv = np.linalg.inv(A)
        X = np.dot(A_inv, b)
        print('Plane：z = {} * x + {} * y + {}'.format(X[0, 0], X[1, 0], X[2, 0]))

        R = 0
        for i in range(0, num_corners):
            R = R + (X[0, 0] * x[i] + X[1, 0] * y[i] + X[2, 0] - z[i]) ** 2
        print('standard deviation：{}'.format(3, sqrt(R)))

        if show_plot == True:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            ax1.scatter(x, y, z, c='r', marker='o')
            x_p = np.linspace(min(x), max(x), num_corners)
            y_p = np.linspace(min(y), max(y), num_corners)
            x_p, y_p = np.meshgrid(x_p, y_p)
            z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
            ax1.plot_wireframe(x_p, y_p, z_p, rstride=5, cstride=5)
            plt.show()

        z1 = X[0, 0] * x.max() + X[1, 0] * y.max() + X[2, 0]
        z2 = X[0, 0] * x.max() + X[1, 0] * y.min() + X[2, 0]
        z3 = X[0, 0] * x.min() + X[1, 0] * y.max() + X[2, 0]

        verts = [(x.max(), y.max(), z1), (x.max(), y.min(), z2), (x.min(), y.max(), z3)]
        edges = [(0, 1), (1, 2), (2, 0)]
        faces = [(0, 1, 2)]

        return verts, edges, faces


class ImageTransformProcess:
    @staticmethod
    def readImageBIN(path, BIN=True, threshold=90):
        """
        cv2.imread() does not perform well inside blender python interpreter.
        This function is meant to solve the problem.
        :param path: str, path to the image file in the operation system.
        :param BIN: bool, True = apply binary filter
        :param threshold: int, threshold of the binary filter
        :return: NumPy array, image in Array format
        """
        img = plt.imread(path)[:, :, :3]
        img = (img * 255).astype('uint8')
        if BIN == True:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
            ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        return img

    @staticmethod
    def preprocess(img1, img2):
        """

        :param img1: NumPy array, imported by cv2.imread() or ImageTransformProcess.readImageBIN()
        :param img2: NumPy array, imported by cv2.imread() or ImageTransformProcess.readImageBIN()
        :return: NumPy array
        """
        if (img1.ndim == 3):
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if (img2.ndim == 3):
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)

        return img1, img2

    @staticmethod
    def undistortion(imgae, camera_matrix, dist_coeff):
        """

        :param imgae:
        :param camera_matrix:
        :param dist_coeff:
        :return:
        """
        undistortion_image = cv2.undistort(imgae, camera_matrix, dist_coeff)

        return undistortion_image

    @staticmethod
    def getRectifyTransform(height, width, config):
        """

        :param height:
        :param width:
        :param config:
        :return:
        """
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

    @staticmethod
    def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
        """

        :param image1:
        :param image2:
        :param map1x:
        :param map1y:
        :param map2x:
        :param map2y:
        :return:
        """
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

        return rectifyed_img1, rectifyed_img2

    @staticmethod
    def draw_line(image1, image2):
        """
        Draw parallel lines to help verify if the images are rectified.
        :param image1:
        :param image2:
        :return:
        """
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

    @staticmethod
    def stereoMatchSGBM(left_image, right_image, down_scale=False):
        """
        SGBM method to make a stereoMatch.
        :param left_image: NumPy array
        :param right_image:NumPy array
        :param down_scale: False recommended.
        :return: NumPy array, true disparity
        """
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

    @staticmethod
    def hw3ToN3(points):
        """
        reshape the array from ((resolution), 3) to N3
        :param points: NumPy array, returned by cv2.reprojectimageto3d()
        :return:Numpy array, with shape of N3
        """
        height, width = points.shape[0:2]
        points_1 = points[:, :, 0].reshape(height * width, 1)
        points_2 = points[:, :, 1].reshape(height * width, 1)
        points_3 = points[:, :, 2].reshape(height * width, 1)
        points_ = np.hstack((points_1, points_2, points_3))
        return points_


class BlenderCameraOperation:
    @staticmethod
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

    @staticmethod
    def addCamera(coordinates, cameraNumbers):
        """
        Add virtual camera in blender. Cameras added by this method are always track the world origin.
        :param coordinates: list, 3D coordinates in the World system where the camera focal center should be
        :param cameraNumbers: int, how many cameras should be added
        :return: list, list of camera info, with camera name and coordinate.
        """
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

    @staticmethod
    def addLightSources(coordinates, Numbers):
        """

        :param coordinates: list, 3D coordinates in the World system where the light source center should be
        :param Numbers: int, how many light sources should be added.
        """
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

    @staticmethod
    def render_through_camera(camera, resolution=(2430, 1620), resolution_percentage=100, samples=20):
        """
        Render a frame through a particular camera, and save the image
        :param camera: dict OR str, dict of camera info or str of camera name
        :param resolution: resolution of the rendered picture
        :param resolution_percentage: 100 recommended
        :param samples: the larger the better render quality, but also longer render time, recommend: 15-20
        """
        scene = bpy.context.scene
        bpy.context.scene.cycles.samples = samples
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.resolution_percentage = resolution_percentage
        scene.render.engine = 'CYCLES' # BLENDER_EEVEE engine requairs a display, can't running inside cluster
        # scene.view_layers[0].cycles.use_denoising = True  # super slow & bad quality
        scene.render.use_border = False

        if isinstance(camera, dict):
            scene.camera = bpy.data.objects[camera['name']]
            bpy.data.scenes["Scene"].render.filepath = work_dir+'{}.png'.format(camera['name'])

        elif isinstance(camera, str):
            scene.camera = bpy.data.objects[camera]
            bpy.data.scenes["Scene"].render.filepath = work_dir+'{}.png'.format(camera)

        bpy.ops.render.render(write_still=True)

    @staticmethod
    def isRotationMatrix(R):
        """
        To varify if a Matrix is a Rotation matrix
        :param R: NumPy array, a Matrix
        :return: bool
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)

        return n < 1e-6

    @staticmethod
    def rotationMatrixToEulerAngles(R):
        """
        Convert a Rotation matrix to Eular angle.
        :param R: NumPy array, a Rotation Matrix.
        :return: tuple, angle in rad rotate alone x, y, z coordinate according Euler Angles.
        """
        assert(BlenderCameraOperation.isRotationMatrix(R))

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

    @staticmethod
    def add_co_camera(best_angle_camera):
        """
        According the position of a given camera, add a co-camera to build the stereo camera set.
        :param best_angle_camera: str, camera name of the camera with most markers detected.
        :return: str. name of the generated co-camera
        """
        RT_Matrix_left = bpy.data.objects[best_angle_camera].matrix_world
        view_layer = bpy.context.view_layer
        co_cameraName = best_angle_camera + '_co'
        co_camera_data = bpy.data.cameras.new(name=co_cameraName)
        co_camera_object = bpy.data.objects.new(name=co_cameraName, object_data=co_camera_data)
        view_layer.active_layer_collection.collection.objects.link(co_camera_object)
        co_camera_object.matrix_world = RT_Matrix_left

        co_camera_object.select_set(True)
        view_layer.objects.active = co_camera_object
        bpy.ops.transform.translate(value=camera_baseline_translation, orient_type='LOCAL',
                                    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                    constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                    proportional_edit_falloff='SMOOTH', proportional_size=1,
                                    use_proportional_connected=False, use_proportional_projected=False)

        return co_cameraName

    @staticmethod
    def detect_co_image(best_camera_angle):
        """
        A quick detection of the markers through co-camera, to see if the number of detected markers matches the
        number detected from the left camera.
        :param best_camera_angle: str, name of the camera with most markers detected.
        :return:
        """
        bpy.ops.object.select_all(action='DESELECT')
        best_camera_angle_co = BlenderCameraOperation.add_co_camera(best_camera_angle)
        BlenderCameraOperation.render_through_camera(best_camera_angle_co)
        co_img = ImageTransformProcess.readImageBIN(work_dir + '{}.png'.format(best_camera_angle_co))
        co_corners, co_ids, _ = ArucoInfoDetection.detect_save_aruco_info_image(best_camera_angle_co, co_img)

        return co_corners, co_ids

    @staticmethod
    def debug_vertices(verts):
        """
        Debug only, display howmany vertices need to be deleted.
        :param verts:
        :return:
        """
        n_selected = len([v for v in verts if v.select])
        n_deselect = len([v for v in verts if not v.select])
        print('\nlen vertices = {}, {} selected, {} not selected'.format(
            len(verts), n_selected, n_deselect))

