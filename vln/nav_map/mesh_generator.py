import numpy as np
from PIL import Image
import cv2

import triangle
from scipy.interpolate import griddata
from scipy.ndimage import label
import matplotlib.pyplot as plt
import trimesh

from sfm.tools.read_write_binary import read_points3D_binary


class GroundObstacleMesh:
    def __init__(self, sfm_dir, config):
        self.gravity_z = config['gravity_axis'] == 'z'
        self.region_min_thr = 10000
        self.ground_coord = 0 if self.gravity_z else config['camera_height']
        self.grid_size = config['grid_size']
        self.grid_map_path = f'{sfm_dir}/grid_map.png'
        self.grid_img = np.array(Image.open(self.grid_map_path).convert('L'))
        self.grid_h, self.grid_w = self.grid_img.shape
        print(f'grid_map w/h {self.grid_w}/{self.grid_h}')

        self.sfm_points = read_points3D_binary(f'{sfm_dir}/points3D.bin')
        print(f'GroundObstacleMesh process total {len(self.sfm_points)} points')

        self.mean_track_length = np.median([len(point.image_ids) for point in self.sfm_points.values()])
        self.mean_reproj_error = np.median([point.error for point in self.sfm_points.values()])
        print(f'mean reproj {self.mean_reproj_error}, mean track length {self.mean_track_length}')

        self.mask = self.grid_img < 50
        self.z_values = self._generate_z_values()
        self.interpolated_z = self._interpolate_z_values()
        self.regions = []
        self.mesh = None

        self.mesh_path = f'{sfm_dir}/ground_obstacle.obj'

    def _convert_coordinates(self, point, reverse=False):
        if self.gravity_z:
            return point
        else:
            x, y, z = point
            return x, z, y

    def _convert_to_grid_coordinates(self, x, y):
        grid_x = int(x / self.grid_size + self.grid_w / 2)
        grid_y = int(y / self.grid_size + self.grid_h / 2)
        return grid_x, grid_y

    def _convert_to_real_coordinates(self, grid_x, grid_y):
        x = (grid_x - self.grid_w / 2) * self.grid_size
        y = (grid_y - self.grid_h / 2) * self.grid_size
        return x, y

    def _generate_z_values(self):
        z_dict = {}
        for point in self.sfm_points.values():
            if point.error > self.mean_reproj_error + 0.5 or len(point.image_ids) < self.mean_track_length:
                continue
            x, y, z = self._convert_coordinates(point.xyz)
            # if z > 5.0 or z < self.ground_coord:
            #     continue
            if z > self.ground_coord - 1 or z < -2.0:
                continue
            grid_x, grid_y = self._convert_to_grid_coordinates(x, y)
            if (0 <= grid_x < self.grid_w) and (0 <= grid_y < self.grid_h) and self.mask[grid_y, grid_x]:
                z_dict[(grid_x, grid_y)] = z if (grid_x, grid_y) not in z_dict else min(z_dict[(grid_x, grid_y)], z)
        print(f'Generated Z values: {len(z_dict)} points')
        # self._visualize_z_values(z_dict)
        return z_dict

    def _interpolate_z_values(self):
        """
        Precompute interpolated Z values for the entire grid.
        """
        if not self.z_values:
            return {}
        grid_xs, grid_ys = zip(*self.z_values.keys())
        grid_zs = list(self.z_values.values())
        grid_x, grid_y = np.meshgrid(np.arange(self.grid_w), np.arange(self.grid_h))
        grid_z = griddata((grid_xs, grid_ys), grid_zs, (grid_x, grid_y), method='linear', fill_value=-2)
        return {(int(x), int(y)): z for x, y, z in zip(grid_x.flatten(), grid_y.flatten(), grid_z.flatten())}

    def _visualize_z_values(self, z_values):
        if not z_values:
            print("No Z values to visualize.")
            return
        grid_x, grid_y = zip(*z_values.keys())
        z_coords = list(z_values.values())
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(grid_x, grid_y, c=z_coords, cmap='viridis', s=1)
        plt.colorbar(scatter, label='Z value (height)')
        plt.title('Z Value Distribution')
        plt.xlabel('Grid X Coordinate')
        plt.ylabel('Grid Y Coordinate')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def _find_connected_regions(self):
        labeled_mask, num_features = label(self.mask)
        print(f'Total {num_features} obstacle regions found.')
        # cmap = ListedColormap(['black', 'blue', 'green', 'red', 'yellow', 'purple', 'orange', 'pink'])
        for region_id in range(1, num_features + 1):
            # 获取当前区域的像素坐标
            region = np.argwhere(labeled_mask == region_id)
            # 检查区域是否满足最小阈值
            if region.shape[0] > self.region_min_thr:
                # 创建一个新的二值掩膜，仅显示当前区域
                region_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
                region_mask[labeled_mask == region_id] = 1
                # 将区域掩膜添加到 regions 列表
                self.regions.append(region_mask)
                # 可视化当前区域（如果需要）
        print(f'After filtering, {len(self.regions)} obstacle regions remain.')

    def _has_edge_pixels(self, region):
        # 检查四个边缘的像素
        top_edge = region[0, :]  # 上边缘
        bottom_edge = region[-1, :]  # 下边缘
        left_edge = region[:, 0]  # 左边缘
        right_edge = region[:, -1]  # 右边缘
        # 判断是否有边缘像素
        if np.any(top_edge) or np.any(bottom_edge) or np.any(left_edge) or np.any(right_edge):
            return True  # 存在边缘像素
        return False  # 不存在边缘像素

    def _find_edges(self, region):
        boundary_points = set()
        for y in np.unique(region[:, 0]):
            x_coords = region[region[:, 0] == y][:, 1]
            boundary_points.add((min(x_coords), y))  # Leftmost point
            boundary_points.add((max(x_coords), y))  # Rightmost point

        for x in np.unique(region[:, 1]):
            y_coords = region[region[:, 1] == x][:, 0]
            boundary_points.add((x, min(y_coords)))  # Topmost point
            boundary_points.add((x, max(y_coords)))  # Bottommost point

        # Order boundary points in a clockwise manner
        boundary_points = np.array(list(boundary_points))
        center = np.mean(boundary_points, axis=0)
        angles = np.arctan2(boundary_points[:, 1] - center[1], boundary_points[:, 0] - center[0])
        ordered_boundary_points = boundary_points[np.argsort(angles)]

        print(f'Region {region.sum()} boundary points found: {ordered_boundary_points}')
        return ordered_boundary_points

    def _create_surface(self, region):
        # 确保输入是二值图像
        if region.dtype != np.uint8:
            region = (region > 0).astype(np.uint8)

        # 提取轮廓和层级关系
        contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("未检测到任何有效轮廓。")

        points = []  # 所有顶点坐标
        segments = []  # 边界约束
        for i, contour in enumerate(contours):
            contour = contour.squeeze()
            if len(contour) < 3:  # 排除过小的轮廓
                continue

            start_index = len(points)  # 当前轮廓顶点起始索引
            points.extend(contour.tolist())

            # 添加边界约束（闭合轮廓）
            for j in range(len(contour)):
                segments.append([start_index + j, start_index + (j + 1) % len(contour)])

        # 构建三角剖分输入
        unique_points, unique_indices = np.unique(points, axis=0, return_inverse=True)
        unique_segments = [[unique_indices[start], unique_indices[end]] for start, end in segments]
        tri_data = {
            'vertices': unique_points,
            'segments': np.array(unique_segments, dtype=np.int32),
        }

        # 执行 Triangle 三角剖分
        tri_result = triangle.triangulate(tri_data, 'p')
        if 'vertices' not in tri_result or 'triangles' not in tri_result:
            raise RuntimeError("三角剖分失败，未生成有效的顶点或三角形。")

        # 提取剖分结果
        vertices = tri_result['vertices']
        triangles = tri_result['triangles']

        # 转换为顶部和底部顶点, -y is up
        top_vertices = np.array([
            [real_x, -self.ground_coord, real_z]
            for real_x, real_z in (self._convert_to_real_coordinates(x, y) for x, y in vertices)
        ])
        bottom_vertices = np.array([
            [real_x, self.ground_coord, real_z]
            for real_x, real_z in (self._convert_to_real_coordinates(x, y) for x, y in vertices)
        ])

        # # 可视化结果（真实坐标）
        # plt.figure(figsize=(8, 8))
        # # 绘制三角剖分
        # plt.triplot(top_vertices[:, 0], top_vertices[:, 2], triangles, color='blue')
        # plt.scatter(top_vertices[:, 0], top_vertices[:, 2], color='red', s=1)
        # # 设置长宽比为 1:1
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.title('Triangulated Surface (Real Coordinates)')
        # plt.xlabel('X (real)')
        # plt.ylabel('Z (real)')
        # plt.show()

        # 底面索引偏移
        num_vertices = len(top_vertices)
        bottom_faces = [[i + num_vertices for i in face] for face in triangles]

        # 生成侧面索引
        side_faces = []
        for start, end in unique_segments:
            # 顶部两个点和底部两个点构成四边形，分解为两个三角形
            side_faces.append([start, end, end + num_vertices])
            side_faces.append([start, end + num_vertices, start + num_vertices])

        # 合并顶点和面
        all_vertices = np.vstack((top_vertices, bottom_vertices))
        all_faces = np.vstack((triangles, bottom_faces, side_faces))

        print(f"为区域创建了 {len(top_vertices)} 个顶点和 {len(all_faces)} 个面。")
        return all_vertices, all_faces

    def generate_mesh(self):
        self._find_connected_regions()
        all_meshes = []
        for region in self.regions:
            print(f'Generating mesh for region with {region.sum()} pixels...')
            all_vertices, faces = self._create_surface(region)
            if all_vertices is None:
                continue
            # Create independent mesh
            mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces)
            # Optimize, simplify and repair
            mesh.remove_unreferenced_vertices()
            mesh.fill_holes()
            # Add optimized mesh to the list
            all_meshes.append(mesh)
        # Combine all independent meshes
        combined_mesh = trimesh.util.concatenate(all_meshes)
        # Save combined mesh
        combined_mesh.export(self.mesh_path)
        print("Combined mesh saved successfully.")

    def _write_obj_file(self, vertices, faces):
        with open(self.mesh_path, 'w') as f:
            for vertex in vertices:
                f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            for face in faces:
                f.write(f'f {" ".join(str(v + 1) for v in face)}\n')  # OBJ format expects 1-based indexing


if __name__ == "__main__":
    job_root = '/home/lingkun/Documents/nhrecon-server/data/shanlanlou/ref'
    # job_root = '//mnt/1/sfm/station/station'
    config = {
        # 'keyword': 'floor or ground',
        # 'detect_thr': 0.5,
        'grid_size': 0.1,
        # 'expand_ratio': 1.2,
        'gravity_axis': 'y',
        # 'device': 'cuda',
        # 'mask_resolution': 400,
        # 'mask_ratio_thr': 0.5,
        'camera_height': 2.0,
    }
    mesh_generator = GroundObstacleMesh(job_root, config)
    mesh_generator.generate_mesh()