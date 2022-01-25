#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import json
from collections import OrderedDict
import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 100

BB_COLOR = (248, 64, 24)

DATA_NAME = "carla_220120"
PROJECT_PATH = os.path.abspath(".")
DATA_PATH = PROJECT_PATH + "/" + DATA_NAME + "/"
if not os.path.exists(DATA_PATH):
    print("make data directory")
    os.mkdir(DATA_PATH)
data_type_list = ['image_2', 'label_2', 'ego_measurement', 'bin', 'calib']
for data_type in data_type_list:
    if not os.path.exists(DATA_PATH + data_type):
        print("make directory for %s" %(data_type))
        os.mkdir(DATA_PATH + data_type)

INIT_TIME = time.time()
N_DATA = 20000
EPI_SIZE = 50
TICK_OFFSET = 150



class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data



# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """
    @staticmethod
    def get_bounding_boxes(vehicles, cyclists, pedestrians, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bb_info = []
        for vehicle in vehicles:
            camera_bbox = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)
            left_top = np.min(camera_bbox, axis=0)
            right_bottom = np.max(camera_bbox, axis=0)

            if all(camera_bbox[:,2] > 0) and right_bottom[0,0]> 0 and right_bottom [0, 1] > 0 and left_top[0, 0] < 960 and left_top[0, 1] < 540:                
                
                truncated = 0

                if left_top[0,0] <= 0 or left_top[0,1] <= 0 or right_bottom[0,0] >= 960 or right_bottom[0,1] >= 540:
                    truncated = 1

                extent = vehicle.bounding_box.extent
                
                location_w = vehicle.get_location()
                
                rotation = vehicle.get_transform().rotation
                
                yaw = rotation.yaw - camera.get_transform().rotation.yaw

                T_ws = ClientSideBoundingBoxes.get_matrix(camera.get_transform())
                location_s = np.linalg.inv(T_ws) @ np.array([location_w.x, location_w.y, location_w.z, 1])

                score = 1.5
                
                # extent.z, extent.y, extent.x
                bb_info.append(['Car', truncated, 0, 0, left_top[0,0], left_top[0,1], right_bottom[0, 0],right_bottom[0,1], 2 * extent.z, 2 * extent.y, 2 * extent.x, location_s[0, 1], -location_s[0, 2], location_s[0, 0], -yaw / 180.0 * 3.1415926])
        
        for vehicle in cyclists:
            camera_bbox = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)
            left_top = np.min(camera_bbox, axis=0)
            right_bottom = np.max(camera_bbox, axis=0)

            if all(camera_bbox[:,2] > 0) and right_bottom[0,0]> 0 and right_bottom [0, 1] > 0 and left_top[0, 0] < 960 and left_top[0, 1] < 540:                
                
                truncated = 0

                if left_top[0,0] <= 0 or left_top[0,1] <= 0 or right_bottom[0,0] >= 960 or right_bottom[0,1] >= 540:
                    truncated = 1

                extent = vehicle.bounding_box.extent
                
                location_w = vehicle.get_location()
                
                rotation = vehicle.get_transform().rotation
                
                yaw = rotation.yaw - camera.get_transform().rotation.yaw

                T_ws = ClientSideBoundingBoxes.get_matrix(camera.get_transform())
                location_s = np.linalg.inv(T_ws) @ np.array([location_w.x, location_w.y, location_w.z, 1])

                score = 1.5
                
                bb_info.append(['Cyclist', truncated, 0, 0, left_top[0,0], left_top[0,1], right_bottom[0, 0],right_bottom[0,1], 2 * extent.z, 2 * extent.y, 2 * extent.x, location_s[0, 1], -location_s[0, 2], location_s[0, 0], -yaw / 180.0 * 3.1415926])
        

        for vehicle in pedestrians:
            camera_bbox = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)
            left_top = np.min(camera_bbox, axis=0)
            right_bottom = np.max(camera_bbox, axis=0)

            if all(camera_bbox[:,2] > 0) and right_bottom[0,0]> 0 and right_bottom [0, 1] > 0 and left_top[0, 0] < 960 and left_top[0, 1] < 540:                
                
                truncated = 0

                if left_top[0,0] <= 0 or left_top[0,1] <= 0 or right_bottom[0,0] >= 960 or right_bottom[0,1] >= 540:
                    truncated = 1

                extent = vehicle.bounding_box.extent
                
                location_w = vehicle.get_location()
                
                rotation = vehicle.get_transform().rotation
                
                yaw = rotation.yaw - camera.get_transform().rotation.yaw

                T_ws = ClientSideBoundingBoxes.get_matrix(camera.get_transform())
                location_s = np.linalg.inv(T_ws) @ np.array([location_w.x, location_w.y, location_w.z, 1])

                score = 1.5
                
                bb_info.append(['Pedestrian', truncated, 0, 0, left_top[0,0], left_top[0,1], right_bottom[0, 0],right_bottom[0,1], 2 * extent.z, 2 * extent.y, 2 * extent.x, location_s[0, 1], -location_s[0, 2], location_s[0, 0], -yaw / 180.0 * 3.1415926])
        
        
        return bb_info

    @staticmethod
    def get_bounding_boxes_raw(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box_raw(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]

        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

        return camera_bbox
    @staticmethod
    def get_bounding_box_raw(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_world(bb_cords, vehicle)[:3, :]
        point1 = [cords_x_y_z[0,0], cords_x_y_z[1,0],cords_x_y_z[2,0]]
        point2 = [cords_x_y_z[0,6], cords_x_y_z[1,6], cords_x_y_z[2,6]]
        return [point1, point2]

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera1 = None
        self.camera2 = None
        self.camera3 = None
        self.camera4 = None
        self.semantic1 = None
        self.semantic2 = None
        self.semantic3 = None
        self.semantic4 = None
        self.lidar = None
        self.frame = 0

        self.car = None

        self.display = None
        self.image = None
        self.capture = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        # camera_bp.set_attribute('sensor_tick', str(0.05))
        return camera_bp
    
    def lidar_blueprint(self):
        """
        Returns lidar blueprint
        """

        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        # lidar_bp.set_attribute('sensor_tick', str(0.05))
        # lidar_bp.set_attribute('rotation_frequency', str(20))
        lidar_bp.set_attribute('range', str(120))
        lidar_bp.set_attribute('channels', str(64))
        lidar_bp.set_attribute('points_per_second', str(3932160))
        # lidar_bp.set_attribute('upper_fov', str(22.5))
        # lidar_bp.set_attribute('lower_fov', str(22.5))
        # lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
        # lidar_bp.set_attribute('dropoff_general_rate', str(0.45))
        # lidar_bp.set_attribute('dropoff_intensity_limit', str(0.8))
        # lidar_bp.set_attribute('dropoff_zero_intensity', str(0.4))
        lidar_bp.set_attribute('rotation_frequency', str(30.0))
        return lidar_bp

    def semantic_blueprint(self):
        """
        Returns semantic segmentation camera blueprint.
        """
        semantic_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        semantic_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        semantic_bp.set_attribute('fov', str(VIEW_FOV))
        # semantic_bp.set_attribute('sensor_tick', str(0.05))

        return semantic_bp
    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera1_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=0))
        self.camera1 = self.world.spawn_actor(self.camera_blueprint(), camera1_transform, attach_to=self.car)

        camera2_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=90))
        self.camera2 = self.world.spawn_actor(self.camera_blueprint(), camera2_transform, attach_to=self.car)

        camera3_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=180))
        self.camera3 = self.world.spawn_actor(self.camera_blueprint(), camera3_transform, attach_to=self.car)

        camera4_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=270))
        self.camera4 = self.world.spawn_actor(self.camera_blueprint(), camera4_transform, attach_to=self.car)

        semantic1_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=0))
        self.semantic1 = self.world.spawn_actor(self.semantic_blueprint(), semantic1_transform, attach_to=self.car)

        semantic2_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=90))
        self.semantic2 = self.world.spawn_actor(self.semantic_blueprint(), semantic2_transform, attach_to=self.car)

        semantic3_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=180))
        self.semantic3 = self.world.spawn_actor(self.semantic_blueprint(), semantic3_transform, attach_to=self.car)

        semantic4_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.3), carla.Rotation(yaw=270))
        self.semantic4 = self.world.spawn_actor(self.semantic_blueprint(), semantic4_transform, attach_to=self.car)

        lidar_transform = carla.Transform(carla.Location(x=1.3, y=0.0, z=2.5), carla.Rotation(yaw=0))
        self.lidar = self.world.spawn_actor(self.lidar_blueprint(), lidar_transform, attach_to=self.car)

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        print(calibration)

        self.camera1.calibration = calibration
        self.camera2.calibration = calibration
        self.camera3.calibration = calibration
        self.camera4.calibration = calibration
        

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        car.set_autopilot(True)

        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self, n_data):
        """
        Main program loop.
        """
        target_n_data = n_data + EPI_SIZE
        clock = pygame.time.Clock()
        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            vehicle_bps = self.world.get_blueprint_library().filter("vehicle.*")
            blueprintsWalkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
            B = len(vehicle_bps)
            P = len(blueprintsWalkers)
            # print(B)
            # print(P)
            # for i in range(B):
            #     print(vehicle_bps[i].id)
            non_vehicle_list = [4,7,9,13,16,24]
            vehicle_type_list = [0 for i in range(B)]
            non_vehicle_id_list = [vehicle_bps[i].id for i in non_vehicle_list]

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')
            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
            

            offset = TICK_OFFSET
            with CarlaSyncMode(self.world, self.camera1, self.camera2, self.camera3, self.camera4, self.semantic1, self.semantic2, self.semantic3, self.semantic4, self.lidar, fps=30) as sync_mode:
                while True:
                    offset -= 1
                    clock.tick()
                    snapshot, rgb1, rgb2, rgb3, rgb4, sem1, sem2, sem3, sem4, pcl = sync_mode.tick(timeout=1.0)
                    
                    if offset < 0:
                        start = time.time()

                        ego_data = OrderedDict()
                        vel = self.car.get_velocity()
                        acc = self.car.get_acceleration()
                        loc = self.car.get_location()
                        rot = self.car.get_transform().rotation
                        ang_vel = self.car.get_angular_velocity()
                        ego_data["velocity"] = [vel.x, vel.y, vel.z]
                        ego_data["acceleration"] = [acc.x, acc.y, acc.z]
                        ego_data["location"] = [loc.x, loc.y, loc.z]
                        ego_data["rotation"] = [rot.roll, rot.pitch, rot.yaw]
                        ego_data["angular_velocity"] = [ang_vel.x, ang_vel.y, ang_vel.z]
                        with open (DATA_NAME + '/ego_measurement/%06d.json' %n_data, 'w', encoding='utf-8') as file:
                            json.dump(ego_data, file, ensure_ascii=False, indent="\t") 
                        file.close()
                        nearby_vehicles = []
                        nearby_cyclists = []
                        nearby_pedestrians = []
                        for p in pedestrians:
                            if p.get_location().distance(self.car.get_location()) < 50:
                                nearby_pedestrians.append(p)
                        for vehicle in vehicles:
                            if vehicle.get_location().distance(self.car.get_location()) < 50:
                                if vehicle.type_id in non_vehicle_id_list:
                                    nearby_cyclists.append(vehicle)
                                else :
                                    nearby_vehicles.append(vehicle)
                        print("vehicles: %d, cyclists: %d, pedestrians: %d" %(len(nearby_vehicles), len(nearby_cyclists), len(nearby_pedestrians)))

                        rgb1.save_to_disk(DATA_NAME + '/image_2/%06d.png' % n_data)
                        sem1.save_to_disk(DATA_NAME + '/semseg/%06d.png' % n_data, carla.ColorConverter.CityScapesPalette)

                        bb_for_drawing1 = ClientSideBoundingBoxes.get_bounding_boxes(nearby_vehicles, nearby_cyclists, nearby_pedestrians, self.camera1)
                        with open(DATA_NAME + "/label_2/%06d.txt" % n_data , 'w', encoding = 'utf-8') as file:
                            for vehicle_info in bb_for_drawing1:
                                for info in vehicle_info:
                                    file.write(str(info))
                                    file.write(" ")
                                file.write('\n')
                        file.close()
                        with open(DATA_NAME + "/calib/%06d.txt" % n_data, 'w', encoding = 'utf-8') as file:
                            file.write('P0: 480.0 0.0 480.0 0.0 0.0 480.0 270.0 0.0 0.0 0.0 1.0 0.0\n')
                            file.write('P1: 480.0 0.0 480.0 0.0 0.0 480.0 270.0 0.0 0.0 0.0 1.0 0.0\n')
                            file.write('P2: 480.0 0.0 480.0 0.0 0.0 480.0 270.0 0.0 0.0 0.0 1.0 0.0\n')
                            file.write('P3: 480.0 0.0 480.0 0.0 0.0 480.0 270.0 0.0 0.0 0.0 1.0 0.0\n')
                            file.write('R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01\n')
                            # file.write('Tr_velo_to_cam: 0.0 0.0 1.0 0.0 -1.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.2\n')
                            file.write('Tr_velo_to_cam: 0.0 -1.0 0.0 0.0 0.0 0.0 -1.0 -0.2 1.0 0.0 0.0 0.0\n')
                            file.write('Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01')
                        file.close()

                        n_data += 1

                        '''
                        with open (DATA_NAME + '/ego_measurement/%06d.json' %n_data, 'w', encoding='utf-8') as file:
                            json.dump(ego_data, file, ensure_ascii=False, indent="\t") 
                        file.close()

                        rgb2.save_to_disk(DATA_NAME + '/image_2/%06d.png' % n_data)
                        sem2.save_to_disk(DATA_NAME + '/semseg/%06d.png' % n_data, carla.ColorConverter.CityScapesPalette)

                        bb_for_drawing2 = ClientSideBoundingBoxes.get_bounding_boxes(nearby_vehicles, nearby_cyclists, nearby_pedestrians, self.camera2)

                        with open(DATA_NAME + "/label_2/%06d.txt" % n_data , 'w', encoding = 'utf-8') as file:
                            for vehicle_info in bb_for_drawing2:
                                for info in vehicle_info:
                                    file.write(str(info))
                                    file.write(" ")
                                file.write('\n')
                        file.close()

                        with open(DATA_NAME + "/calib/%06d.txt" % n_data, 'w', encoding = 'utf-8') as file:
                            file.write('P0: -480.0 0.0 480.0 0.0 -270.0 480.0 0.0 0.0 -1.0 0.0 0.0 0.0\n')
                            file.write('P1: -480.0 0.0 480.0 0.0 -270.0 480.0 0.0 0.0 -1.0 0.0 0.0 0.0\n')
                            file.write('P2: -480.0 0.0 480.0 0.0 -270.0 480.0 0.0 0.0 -1.0 0.0 0.0 0.0\n')
                            file.write('P3: -480.0 0.0 480.0 0.0 -270.0 480.0 0.0 0.0 -1.0 0.0 0.0 0.0\n')
                            file.write('R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01\n')
                            # file.write('Tr_velo_to_cam: -1.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 -1.0 0.0 0.2\n')
                            file.write('Tr_velo_to_cam: -1.0 0.0 0.0 0.0 0.0 0.0 -1.0 -0.2 0.0 -1.0 0.0 0.0\n')
                            file.write('Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01')
                        file.close()

                        n_data += 1

                        with open (DATA_NAME + '/ego_measurement/%06d.json' %n_data, 'w', encoding='utf-8') as file:
                            json.dump(ego_data, file, ensure_ascii=False, indent="\t") 
                        file.close()

                        rgb3.save_to_disk(DATA_NAME + '/image_2/%06d.png' % n_data)
                        sem3.save_to_disk(DATA_NAME + '/semseg/%06d.png' % n_data, carla.ColorConverter.CityScapesPalette)

                        bb_for_drawing3 = ClientSideBoundingBoxes.get_bounding_boxes(nearby_vehicles, nearby_cyclists, nearby_pedestrians, self.camera3)

                        with open(DATA_NAME + "/label_2/%06d.txt" % n_data , 'w', encoding = 'utf-8') as file:
                            for vehicle_info in bb_for_drawing3:
                                for info in vehicle_info:
                                    file.write(str(info))
                                    file.write(" ")
                                file.write('\n')
                        file.close()

                        with open(DATA_NAME + "/calib/%06d.txt" % n_data, 'w', encoding = 'utf-8') as file:
                            file.write('P0: -480.0 0.0 -480.0 0.0 0.0 480.0 -270.0 0.0 0.0 0.0 -1.0 0.0\n')
                            file.write('P1: -480.0 0.0 -480.0 0.0 0.0 480.0 -270.0 0.0 0.0 0.0 -1.0 0.0\n')
                            file.write('P2: -480.0 0.0 -480.0 0.0 0.0 480.0 -270.0 0.0 0.0 0.0 -1.0 0.0\n')
                            file.write('P3: -480.0 0.0 -480.0 0.0 0.0 480.0 -270.0 0.0 0.0 0.0 -1.0 0.0\n')
                            file.write('R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01\n')
                            # file.write('Tr_velo_to_cam: 0.0 0.0 -1.0 0.0 1.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.2\n')
                            file.write('Tr_velo_to_cam: 0.0 1.0 0.0 0.0 0.0 0.0 -1.0 -0.2 -1.0 0.0 0.0 0.0\n')
                            file.write('Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01')
                        file.close()

                        
                        n_data += 1

                        with open (DATA_NAME + '/ego_measurement/%06d.json' %n_data, 'w', encoding='utf-8') as file:
                            json.dump(ego_data, file, ensure_ascii=False, indent="\t") 
                        file.close()

                        rgb4.save_to_disk(DATA_NAME + '/image_2/%06d.png' % n_data)
                        sem4.save_to_disk(DATA_NAME + '/semseg/%06d.png' % n_data, carla.ColorConverter.CityScapesPalette)

                        bb_for_drawing4 = ClientSideBoundingBoxes.get_bounding_boxes(nearby_vehicles, nearby_cyclists, nearby_pedestrians, self.camera4)

                        with open(DATA_NAME + "/label_2/%06d.txt" % n_data , 'w', encoding = 'utf-8') as file:
                            for vehicle_info in bb_for_drawing4:
                                for info in vehicle_info:
                                    file.write(str(info))
                                    file.write(" ")
                                file.write('\n')
                        file.close()

                        with open(DATA_NAME + "/calib/%06d.txt" % n_data, 'w', encoding = 'utf-8') as file:
                            file.write('P0: 480.0 0.0 -480.0 0.0 270.0 480.0 0.0 0.0 1.0 0.0 0.0 0.0\n')
                            file.write('P1: 480.0 0.0 -480.0 0.0 270.0 480.0 0.0 0.0 1.0 0.0 0.0 0.0\n')
                            file.write('P2: 480.0 0.0 -480.0 0.0 270.0 480.0 0.0 0.0 1.0 0.0 0.0 0.0\n')
                            file.write('P3: 480.0 0.0 -480.0 0.0 270.0 480.0 0.0 0.0 1.0 0.0 0.0 0.0\n')
                            file.write('R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01\n')
                            # file.write('Tr_velo_to_cam: 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.2\n')
                            file.write('Tr_velo_to_cam: 1.0 0.0 0.0 0.0 0.0 0.0 -1.0 -0.2 0.0 1.0 0.0 0.0\n')
                            file.write('Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01')
                        file.close()

                        n_data += 1
                        '''
                        # pcl.save_to_disk(DATA_NAME + '/pointcloud/%06d.ply' % n_data)
                        pcl = np.frombuffer(pcl.raw_data, dtype=np.dtype('f4'))
                        pcl = pcl.reshape((int(pcl.shape[0] / 4), 4))
                        # pcl_alt = np.zeros(pcl.shape)
                        pcl_alt = []
                        n_pc = pcl.shape[0]
                        n_alt_pc = 0
                        for i in range(n_pc):
                            if pcl[i][0] < 0:
                                continue
                            pcl_alt.append([pcl[i][0], -pcl[i][1], pcl[i][2], pcl[i][3]])
                            # pcl_alt[n_alt_pc][0] = pcl[i][0]
                            # pcl_alt[n_alt_pc][1] = -pcl[i][1]
                            # pcl_alt[n_alt_pc][2] = -pcl[i][2]
                            n_alt_pc += 1
                        pcl_alt = np.array(pcl_alt)
                        # fake_intensity = np.ones((n_alt_pc, 1)) * 0.5
                        # pcl_alt = np.concatenate((pcl_alt, fake_intensity), axis=1)
                        # pcl_alt.astype('float32').tofile(DATA_NAME + '/bin/%06d.bin' % (n_data-4))
                        # pcl_alt.astype('float32').tofile(DATA_NAME + '/bin/%06d.bin' % (n_data-3))
                        # pcl_alt.astype('float32').tofile(DATA_NAME + '/bin/%06d.bin' % (n_data-2))
                        pcl_alt.astype('float32').tofile(DATA_NAME + '/bin/%06d.bin' % (n_data-1))

                        end = time.time()
                        print("[%.3f] elapsed time: %.3f, current # of data: %d" %(end - INIT_TIME, end - start, n_data))
                        if n_data >= target_n_data:
                            break

                    # print("********************************")
                    # print(self.car.get_transform())
                    # print(bounding_boxes1)
                    # print(len(bounding_boxes2))
                    # print(len(bounding_boxes3))
                    # print(len(bounding_boxes4))
                    # print("********************************")
                    
                    # ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bb_for_drawing)

                    pygame.display.flip()

                    # pygame.event.pump()
                    if self.control(self.car):
                        return
                    self.frame += 1
                    

        finally:
            self.set_synchronous_mode(False)
            self.camera1.destroy()
            self.camera2.destroy()
            self.camera3.destroy()
            self.camera4.destroy()
            self.semantic1.destroy()
            self.semantic2.destroy()
            self.semantic3.destroy()
            self.semantic4.destroy()
            self.lidar.destroy()
            self.car.destroy()
            pygame.quit()
            return n_data


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """
    n_data = 8400
    i_episode = 0
    while n_data < N_DATA:
        try:
            client = BasicSynchronousClient()
            n_data = client.game_loop(n_data)
            i_episode += 1
        except:
            print("ERROR. Now we restart scenario")
            time.sleep(5)
            continue
        finally:
            print("FINISTH EPISODE %d, total data: %d" %(i_episode, n_data))
            print("Now we sleep for a small moment")
            time.sleep(5)
    print('EXIT')


if __name__ == '__main__':
    main()
