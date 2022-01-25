import rospy
import rospkg
import json
import numpy as np
import math
import cv2
import time
import copy

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
fontthickness = 2
fontline = cv2.LINE_AA

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str


def get_pixel(x, y):
    px = int((x - maxX) / (minX - maxX) * bev_height)
    py = int((y - maxY) / (minY - maxY) * bev_width)

    return px, py

def draw_object(image, objects, type):
    for i in range(len(objects)):
        cx = objects[i][0]
        cy = objects[i][1]
        l = objects[i][4]
        w = objects[i][5]
        theta = objects[i][3]
        box = []
        box.append(get_pixel(cx + l/2 * math.cos(theta) - w/2 * math.sin(theta), cy + l/2 * math.sin(theta) + w/2 * math.cos(theta)))
        box.append(get_pixel(cx - l/2 * math.cos(theta) - w/2 * math.sin(theta), cy - l/2 * math.sin(theta) + w/2 * math.cos(theta)))
        box.append(get_pixel(cx - l/2 * math.cos(theta) + w/2 * math.sin(theta), cy - l/2 * math.sin(theta) - w/2 * math.cos(theta)))
        box.append(get_pixel(cx + l/2 * math.cos(theta) + w/2 * math.sin(theta), cy + l/2 * math.sin(theta) - w/2 * math.cos(theta)))
        box.append(get_pixel(cx, cy))
        if type == 0:
            if objects[i][7] < 0.01:
                continue
            cv2.line(image, (box[0][1], box[0][0]), (box[1][1], box[1][0]), (0, 0, 255), 2)
            cv2.line(image, (box[1][1], box[1][0]), (box[2][1], box[2][0]), (0, 0, 255), 2)
            cv2.line(image, (box[2][1], box[2][0]), (box[3][1], box[3][0]), (0, 0, 255), 2)
            cv2.line(image, (box[3][1], box[3][0]), (box[0][1], box[0][0]), (0, 255, 255), 2)
            cv2.putText(image, str(round(objects[i][7],2)), (box[4][1], box[4][0]+20), font, fontscale, (255, 255, 255), fontthickness, fontline)
        if type == 1 :
            cv2.line(image, (box[0][1], box[0][0]), (box[1][1], box[1][0]), (255, 0, 0), 2)
            cv2.line(image, (box[1][1], box[1][0]), (box[2][1], box[2][0]), (255, 0, 0), 2)
            cv2.line(image, (box[2][1], box[2][0]), (box[3][1], box[3][0]), (255, 0, 0), 2)
            cv2.line(image, (box[3][1], box[3][0]), (box[0][1], box[0][0]), (255, 255, 0), 2)
        if type == 2 :
            cv2.line(image, (box[0][1], box[0][0]), (box[1][1], box[1][0]), (0, 255, 0), 2)
            cv2.line(image, (box[1][1], box[1][0]), (box[2][1], box[2][0]), (0, 255, 0), 2)
            cv2.line(image, (box[2][1], box[2][0]), (box[3][1], box[3][0]), (0, 255, 0), 2)
            cv2.line(image, (box[3][1], box[3][0]), (box[0][1], box[0][0]), (255, 255, 0), 2)
        
data_name = "carla_220120"
n_data = 1000
offset = 0

LIDAR_Z = 1.7
minX = 0.0
maxX = 40.0
minY = -20.0
maxY = 20.0
minZ = 0.5 - LIDAR_Z
maxZ = 2.0 - LIDAR_Z

resolution = 1.0
bev_height = 40
bev_width = 40
cx = (minX + maxX) / 2.0
cy = (minY + maxY) / 2.0

maxlogDensity = math.log(20)

data_path = data_name + "/"
obj_path = data_path + "obj/"
lbl_path = data_path + "label_2/"
bin_path = data_path + "bin/"
out_path = data_path + "out/"
clb_path = data_path + "calib/"

for seq in range(offset, offset + n_data):
    # raw_file = raw_path + str(seq).zfill(6) + ".txt"
    obj_file = obj_path + str(seq).zfill(6) + ".txt"
    lbl_file = lbl_path + str(seq).zfill(6) + ".txt"
    bin_file = bin_path + str(seq).zfill(6) + ".bin"
    out_file = out_path + str(seq).zfill(6) + ".png"
    clb_file = clb_path + str(seq).zfill(6) + ".txt"
    
    calib = Calibration(clb_file)
    labels = get_objects_from_label(lbl_file)
    obj_list = []
    for i in range(len(labels)):
        obj = [labels[i].loc[0], labels[i].loc[1], labels[i].loc[2], labels[i].l, labels[i].w, labels[i].h, labels[i].ry]
        obj_list.append(obj)
    obj_list = np.array(obj_list)
    obj_list = boxes3d_kitti_camera_to_lidar(obj_list, calib)
    
    raw_objects = []
    # with open(raw_file, "r") as f:
    #     for line in f: 
    #         box = line.split()
    #         obj = []
    #         for i in range(0,7):
    #             obj.append(float(box[i]))
    #         raw_objects.append(obj)

    # objects = []
    # with open(obj_file, "r") as f:
    #     for line in f: 
    #         box = line.split()
    #         obj = []
    #         for i in range(0,8):
    #             obj.append(float(box[i]))
    #         objects.append(obj)

    # labels = []
    # with open(lbl_file, "r") as f:
    #     for line in f: 
    #         box = line.split()
    #         obj = []
    #         for i in range(0,7):
    #             obj.append(float(box[i]))
    #         labels.append(obj)
    
    # build bev map
    pcs = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    x = pcs[:,0]
    y = pcs[:,1]
    z = pcs[:,2]
    intensity = pcs[:,3]
    
    indices = []
    for i in range(len(pcs)):
        if x[i] > minX and x[i] < maxX and y[i] > minY and y[i] < maxY and z[i] > minZ and z[i] < maxZ:
            indices.append(i)
    pcs = pcs[indices,:]
    x = x[indices]
    y = y[indices]
    z = z[indices]
    intensity = intensity[indices]
    n_points = len(intensity)
    
    resolution = 0.1
    bev_height = 400
    bev_width = 400
    maxlogDensity = math.log(20)

    intensity_layer = np.zeros([bev_height, bev_width], dtype=np.float)
    density_layer = np.zeros([bev_height, bev_width], dtype=np.float)
    height_layer = np.zeros([bev_height, bev_width], dtype=np.float)
    for i in range(n_points):
        px, py = get_pixel(x[i], y[i])
        if px < 0 or px >= bev_height or py < 0 or py >= bev_width:
            continue
        intensity_layer[px][py] = max(intensity_layer[px][py], intensity[i])
        density_layer[px][py] += 1
        height_layer[px][py] = max(height_layer[px][py], (z[i]-minZ)/ (maxZ-minZ))
    for i in range(bev_height):
        for j in range(bev_width):
            density_layer[px][py] = min(1.0, math.log(1 + density_layer[px][py]) / maxlogDensity)
    intensity_layer = intensity_layer * 255.0
    density_layer = density_layer * 255.0
    height_layer = height_layer * 255.0
    intensity_layer = np.expand_dims(intensity_layer.astype('uint8'), axis = 0)
    density_layer = np.expand_dims(density_layer.astype('uint8'), axis = 0)
    height_layer = np.expand_dims(height_layer.astype('uint8'), axis = 0)
    local_map = np.transpose(np.vstack((intensity_layer, density_layer, height_layer)), (1,2,0))
    local_map = cv2.resize(local_map, (bev_height, bev_width))

    # draw_object(local_map, objects, 0)
    # draw_object(local_map, labels, 1)
    # draw_object(local_map, raw_objects, 2)
    draw_object(local_map, obj_list, 1)

    cv2.imwrite(out_file, local_map)

    if seq%10 == 0:
        print(seq)