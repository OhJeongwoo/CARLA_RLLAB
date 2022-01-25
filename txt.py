import numpy as np
import struct

n = 10000
DATA_NAME = "carla_220120"
# bin_path = DATA_NAME + "/bin/"
# def bin_valid(binFileName):
#     size_float = 4
#     list_pcd = []
#     with open(binFileName, "rb") as f:
#         byte = f.read(size_float * 4)
#         while byte:
#             x, y, z, intensity = struct.unpack("ffff", byte)
#             list_pcd.append([x, y, z])
#             byte = f.read(size_float * 4)
#     np_pcd = np.asarray(list_pcd)
#     if np_pcd.shape[1] != 3:
#         print("ERROR")
#     return

# for seq in range(n):
#     if seq % 100 == 0:
#         print(seq)
#     bin_file = bin_path + str(seq).zfill(6) + ".bin"
#     bin_valid(bin_file)


emp_list = []
for seq in range(n):
    lbl_file = DATA_NAME+'/label_2/' + str(seq).zfill(6) + ".txt"
    f = open(lbl_file, 'r')
    ln = 0
    while True:
        line = f.readline()
        if not line: break
        ln += 1
    if ln == 0:
        emp_list.append(seq)
    f.close()
print(len(emp_list))


f = open("test.txt", 'w')
for i in range(9000, 10000):
    if i not in emp_list:
        f.write(str(i).zfill(6) + "\n")
f.close()
f = open("train.txt", 'w')
for i in range(0, 2250):
    if 4*i not in emp_list:
        f.write(str(4*i).zfill(6) + "\n")
    if 4*i+1 not  in emp_list:
        f.write(str(4*i+1).zfill(6) + "\n")
    if 4*i+2 not in emp_list:
        f.write(str(4*i+2).zfill(6) + "\n")
f.close()
f = open("val.txt", 'w')
for i in range(0, 2250):
    if 4*i+3 not in emp_list:
        f.write(str(4*i+3).zfill(6) + "\n")
f.close()
