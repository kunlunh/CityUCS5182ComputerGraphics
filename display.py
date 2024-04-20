import open3d as o3d
import sys

if len(sys.argv) != 2:
    print("please use python script.py path_to_ply_file")
    exit(1)

ply_path = sys.argv[1]
point_cloud = o3d.io.read_point_cloud(ply_path)

o3d.visualization.draw_geometries([point_cloud])
