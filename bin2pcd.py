import open3d as o3d
import numpy as np
import struct
import sys
import os

def bin_to_pcd(binFileName, pcdFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    print(np_pcd.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    o3d.io.write_point_cloud(pcdFileName, pcd)
    return

DATA_NAME = "carla_220120"
bin_path = DATA_NAME + "/bin/"
pcd_path = DATA_NAME + "/pcd/"
for i in range(0,70):
    seq = i
    bin_file = bin_path + str(seq).zfill(6) + ".bin"
    pcd_file = pcd_path + str(seq).zfill(6) + ".pcd"
    bin_to_pcd(bin_file, pcd_file)