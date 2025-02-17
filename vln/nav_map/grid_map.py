# import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sfm.tools.read_write_binary import read_cameras_binary, read_images_binary, read_points3D_binary


class FloorAccumulator:
    def __init__(self, sfm_dir, images_dir, config):
        self.gravity_z = config['gravity_axis'] == 'z'
        grid_size = config['grid_size']
        # width = config['width']
        # height = config['height']
        device = config['device']
        expand_ratio = config['expand_ratio']

        self.sfm_cameras = read_cameras_binary(f'{sfm_dir}/cameras.bin')
        self.sfm_images = read_images_binary(f'{sfm_dir}/images.bin')
        self.sfm_points = read_points3D_binary(f'{sfm_dir}/points3D.bin')
        print(f'FloorAccumulator process total {len(self.sfm_images)} images')
        self.images_dir = images_dir
        max_x, max_y = 0, 0
        for sfm_image in self.sfm_images.values():
            t_cw = sfm_image.tvec
            q_cw = sfm_image.qvec
            q_cw = np.roll(q_cw, 3)
            rotation_wc = R.inv(R.from_quat(q_cw))
            sfm_pos = np.dot(-rotation_wc.as_matrix(), np.array(t_cw).T)

            max_x = max(abs(sfm_pos[0]), max_x)
            if self.gravity_z:
                max_y = max(abs(sfm_pos[1]), max_y)
            else:
                max_y = max(abs(sfm_pos[2]), max_y)
        width = 2 * max_x * expand_ratio
        height = 2 * max_y * expand_ratio

        self.device = device
        self.grid_size = grid_size  # 每个栅格的大小
        self.grid_width = int(width / grid_size + 1)  # 设置栅格地图的宽度
        self.grid_height = int(height / grid_size + 1)  # 设置栅格地图的高度
        print(f'init world box with size w/h {width}/{height}!')
        print(f'init grid_map with resolution w/h {self.grid_width}/{self.grid_height}!')

        # detector_id = "IDEA-Research/grounding-dino-tiny"
        # segmenter_id = "facebook/sam-vit-base"
        # self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        # self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device)
        # print(f"Initializing grounded segmenter with detector_id: {detector_id} and segmenter_id: {segmenter_id}")
        self.weight_root = Path(__file__).parent.parent.parent / 'weights'
        self.detector_processor = AutoProcessor.from_pretrained(str(self.weight_root / 'dino_tiny'))
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(str(self.weight_root / 'dino_tiny')).to(self.device)
        print('load detector done')

        # self.segmenter_processor = SamProcessor.from_pretrained(segmenter_id)
        # self.segmenter = SamModel.from_pretrained(segmenter_id).to(self.device)
        self.segmenter_processor = SamProcessor.from_pretrained(str(self.weight_root / 'sam_base'))
        self.segmenter = SamModel.from_pretrained(str(self.weight_root / 'sam_base')).to(self.device)
        print('load segmenter done')

        self.detect_thr = config['detect_thr']
        self.key_word = config['keyword']
        self.mask_resolution = config['mask_resolution']
        self.mask_ratio_thr = config['mask_ratio_thr']
        self.camera_height = config['camera_height']

        # self.floor_y = self.camera_height      # 相机坐标系重力轴为y，假设地面的y值为1,用于floor_mask的投影区域计算
        # self.floor_z = 0.0      # 相机坐标系重力轴为y，假设地面的y值为1,用于floor_mask的投影区域计算
        # 输出的地面二维占据栅格地图，用占据概率更新公式去刷新此grid
        self.grid_map_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.grid_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)  # 初始化栅格地图
        self.observation_count = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)  # 记录观测次数
        self.grid_map_path = f'{sfm_dir}/grid_map.png'  # 输出的占据栅格地图图片

    def to(self, device):
        self.device = device
        self.detector.to(device)
        self.segmenter.to(device)

    def segment(self, image: Image.Image, score_threshold: float = 0.5, box_threshold: float = 0.5,
                text_threshold: float = 0.5) -> (bool, np.ndarray):
        # original_width, original_height = image.size
        width, height = image.size
        # if width > height:
        new_width = self.mask_resolution
        new_height = int(height * (self.mask_resolution / width))
        # else:
        #     new_height = self.mask_resolution
        #     new_width = int(width * (self.mask_resolution / height))
        image = image.resize((new_width, new_height))
        
        # Detect objects
        with self.model_lock:
            # print("Detecting objects...")
            inputs = self.detector_processor(images=image, text=f"{self.key_word.lower()}.", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.detector(**inputs)

            results = self.detector_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]]
            )

        if not results or len(results[0]["boxes"]) == 0:
            # print("Detecting failed. No results found.")
            return False, Image.new("L", image.size, 0)
        
        first_box = results[0]["boxes"][0].tolist()
        input_box = [[first_box[0], first_box[1]], [first_box[2], first_box[3]]]  # SAM expects top-right and bottom-left points

        # Segment the object
        # print("Segmenting...")
        with torch.no_grad():
            inputs = self.segmenter_processor(image, input_boxes=[input_box], return_tensors="pt").to(self.device)
            outputs = self.segmenter(**inputs)

        masks = self.segmenter_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        scores = outputs.iou_scores[0][0]

        best_mask_idx = scores.argmax().item()
        best_score = scores[best_mask_idx].item()
        
        if best_score < score_threshold:
            # print("Segmentation failed. No results found.")
            return False, Image.new("L", image.size, 0)

        mask = masks[best_mask_idx].squeeze().numpy()
        mask = (mask * 255).astype(np.uint8)
        # print("Finished segmenting.")

        mask_ratio = np.sum(mask > 0) / mask.size
        if mask_ratio > self.mask_ratio_thr:
            # print(f'mask ratio {mask_ratio} too large, ignore')
            return False, Image.new("L", image.size, 0)

        mask = Image.fromarray(mask, mode='L')
        # print(f'found useful image, {self.key_word} mask ratio {mask_ratio}')
        # mask = mask.resize((original_width, original_height))
        return True, mask

    def project_mask_to_grid2(self, mask, rot, tvec, focal_length):
        """将掩膜投影到地平面 y = floor_y 并更新栅格地图"""
        local_grid_map = np.zeros_like(self.grid_map)
        local_observation_count = np.zeros_like(self.observation_count)

        t = np.array(tvec)  # 平移向量
        camera_center = -rot.T @ t  # 相机中心在世界坐标系中的位置

        # 相机内参：fx 和 cx, cy 为图像中心
        fx, fy = focal_length, focal_length
        cx, cy = mask.shape[1] / 2, mask.shape[0] / 2

        all_indices = np.ndindex(mask.shape)  # 所有像素

        for pixel_y, pixel_x in all_indices:
            # Step 1: 计算射线方向 (相机坐标系)
            ray_dir_camera = np.array([
                (pixel_x - cx) / fx,
                (pixel_y - cy) / fy,
                1.0
            ])
            ray_dir_camera /= np.linalg.norm(ray_dir_camera)
            # print(f'pixel ray in camera axes: {ray_dir_camera}')
            # Step 2: 射线方向变换到世界坐标系
            ray_dir_world = rot.T @ ray_dir_camera

            # Step 3: 计算射线与地平面 y = floor_y 的交点
            if self.gravity_z:
                estimate_floor_coord = camera_center[2] - self.camera_height  # robot axes z is up
                t_scale = (estimate_floor_coord - camera_center[2]) / ray_dir_world[2]
                if t_scale <= 0:
                    continue  # 跳过无法与地面交点的射线

                ground_point = camera_center + t_scale * ray_dir_world  # cam axes y is down
                grid_x = int((ground_point[0] / self.grid_size) + self.grid_width / 2)
                grid_y = int((ground_point[1] / self.grid_size) + self.grid_height / 2)
            else:
                estimate_floor_coord = camera_center[1] + self.camera_height
                t_scale = (estimate_floor_coord - camera_center[1]) / ray_dir_world[1]
                if t_scale <= 0:
                    continue  # 跳过无法与地面交点的射线

                ground_point = camera_center + t_scale * ray_dir_world
                grid_x = int(ground_point[0] / self.grid_size + self.grid_width / 2)
                grid_y = int(ground_point[2] / self.grid_size + self.grid_height / 2)

            # Step 4: 更新局部栅格地图
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                local_observation_count[grid_y, grid_x] += 1
                if self.observation_count[grid_y, grid_x] == 0:
                    update = 1.0
                else:
                    update = 1.0 / self.observation_count[grid_y, grid_x]

                if mask[pixel_y, pixel_x] > 0:  # 前景（地面）像素
                    local_grid_map[grid_y, grid_x] =  -update * self.grid_map[grid_y, grid_x] + update * 1.0
                else:  # 非地面像素
                    local_grid_map[grid_y, grid_x] = -update * self.grid_map[grid_y, grid_x] + update * 0.0

        # 合并局部结果到全局
        with self.grid_map_lock:
            self.grid_map += local_grid_map
            self.observation_count += local_observation_count

    def project_mask_to_grid(self, mask, rot, tvec, focal_length):
        """将掩膜投影到地平面 y = floor_y 并更新栅格地图"""
        t = np.array(tvec)  # 平移向量
        camera_center = -rot.T @ t  # 相机中心在世界坐标系中的位置

        # 相机内参：fx 和 cx, cy 为图像中心
        fx, fy = focal_length, focal_length
        cx, cy = mask.shape[1] / 2, mask.shape[0] / 2

        # mask_indices = np.argwhere(mask > 0)  # 前景像素
        all_indices = np.ndindex(mask.shape)  # 所有像素

        for pixel_y, pixel_x in all_indices:
            # Step 1: 计算射线方向 (相机坐标系)
            ray_dir_camera = np.array([
                (pixel_x - cx) / fx,
                (pixel_y - cy) / fy,
                1.0
            ])
            ray_dir_camera /= np.linalg.norm(ray_dir_camera)
            # print(f'pixel ray in camera axes: {ray_dir_camera}')
            # Step 2: 射线方向变换到世界坐标系
            ray_dir_world = rot.T @ ray_dir_camera

            # Step 3: 计算射线与地平面 y = floor_y 的交点
            if self.gravity_z:
                estimate_floor_coord = camera_center[2] - self.camera_height   # robot axes z is up
                t_scale = (estimate_floor_coord - camera_center[2]) / ray_dir_world[2]
                if t_scale <= 0:
                    continue  # 跳过无法与地面交点的射线

                ground_point = camera_center + t_scale * ray_dir_world  # cam axes y is down
                grid_x = int((ground_point[0] / self.grid_size) + self.grid_width / 2)
                grid_y = int((ground_point[1] / self.grid_size) + self.grid_height / 2)
            else:
                estimate_floor_coord = camera_center[1] + self.camera_height
                t_scale = (estimate_floor_coord - camera_center[1]) / ray_dir_world[1]
                if t_scale <= 0:
                    continue  # 跳过无法与地面交点的射线

                ground_point = camera_center + t_scale * ray_dir_world
                grid_x = int((ground_point[0] / self.grid_size) + self.grid_width / 2)
                grid_y = int((ground_point[2] / self.grid_size) + self.grid_height / 2)

            # Step 4: 更新栅格地图
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # with self.lock:
                    self.observation_count[grid_y, grid_x] += 1
                    update = 1.0 / self.observation_count[grid_y, grid_x]

                    if mask[pixel_y, pixel_x] > 0:  # 前景（地面）像素
                        self.grid_map[grid_y, grid_x] = (1 - update) * self.grid_map[grid_y, grid_x] + update * 1.0
                    else:  # 非地面像素
                        self.grid_map[grid_y, grid_x] = (1 - update) * self.grid_map[grid_y, grid_x] + update * 0.0

    def update_visualization(self, original_image, mask, image_idx, image_name):
        """实时更新可视化窗口，展示原始图像、掩膜和当前栅格地图"""
        # 原始图像
        self.ax1.clear()
        self.ax1.imshow(original_image)
        self.ax1.set_title(f"Image #{image_idx}: {image_name}")
        self.ax1.axis('off')  # 隐藏坐标轴

        # 分割掩膜
        self.ax2.clear()
        self.ax2.imshow(mask, cmap='gray')
        self.ax2.set_title("Segmented Mask")
        self.ax2.axis('off')

        # 栅格地图
        # occupancy_probability = np.clip(self.grid_map, 0, 1)  # 确保占用概率在 [0, 1] 范围内
        self.ax3.clear()
        im = self.ax3.imshow(self.grid_map, cmap='gray', origin='lower')
        self.ax3.set_title("Current Grid Map")
        self.ax3.set_xlabel("Grid X")
        self.ax3.set_ylabel("Grid Y")

        # 添加颜色条，只在第一次显示
        # if not hasattr(self, "colorbar"):
        #     self.colorbar = self.fig.colorbar(im, ax=self.ax3, label="Occupancy Probability")

        # 刷新图像
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # 短暂暂停以更新显示

    def save_grid_map(self):
        """保存栅格地图为图像"""
        occupancy_probability = self.grid_map  # 直接使用已经更新的占用概率
        occupancy_probability = np.clip(occupancy_probability, 0, 1)  # 限制在 [0, 1] 范围内
        occupancy_image = (occupancy_probability * 255).astype(np.uint8)  # 转换为图像格式
        occupancy_image = Image.fromarray(occupancy_image)
        occupancy_image.save(self.grid_map_path)  # 保存图像

    def run(self):
        # 设置非阻塞模式的图像窗口
        plt.ion()  # 打开交互模式
        # 创建一个 2x2 的网格布局
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])  # 控制布局
        self.ax1, self.ax2, self.ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

        image_list = list(self.sfm_images.values())
        import random
        random.shuffle(image_list)  # shuffle to visual rough map quickly
        for index, image in enumerate(image_list):
            # if not image.name.startswith('F2'):
            #     continue
            print(f'processing image {image.name}')
            pil_image = Image.open(f'{self.images_dir}/{image.name}')
            success, floor_mask = self.segment(pil_image, score_threshold=self.detect_thr,
                                               box_threshold=self.detect_thr,
                                               text_threshold=self.detect_thr)
            if success:
                rot_cw = image.qvec2rotmat()
                tvec_cw = image.tvec

                focal_length = (self.sfm_cameras[image.camera_id].params[0] /
                                self.sfm_cameras[image.camera_id].width * self.mask_resolution)
                # 将掩膜投影到y=floor_y的地平面栅格地图
                self.project_mask_to_grid(np.array(floor_mask), rot_cw, tvec_cw, focal_length)
            # 实时可视化栅格地图
            self.update_visualization(pil_image, floor_mask, index, image.name)
        plt.ioff()
        # 最后可视化或保存栅格地图
        self.save_grid_map()

    def run_concurrent(self):
        image_list = list(self.sfm_images.values())
        from tqdm import tqdm
        import random
        random.shuffle(image_list)  # shuffle to visual rough map quickly
        # 优化1: 使用多线程数据加载
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def process_image(image):
            pil_image = Image.open(f'{self.images_dir}/{image.name}')
            success, floor_mask = self.segment(pil_image, score_threshold=self.detect_thr,
                                               box_threshold=self.detect_thr,
                                               text_threshold=self.detect_thr)
            if success:
                rot_cw = image.qvec2rotmat()
                tvec_cw = image.tvec
                focal_length = (self.sfm_cameras[image.camera_id].params[0] /
                                self.sfm_cameras[image.camera_id].width * self.mask_resolution)
                # 将掩膜投影到y=floor_y的地平面栅格地图
                self.project_mask_to_grid2(np.array(floor_mask), rot_cw, tvec_cw, focal_length)
        processed_count = 0  # 计数器
        total_images = len(image_list)  # 总图片数量
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_image = {executor.submit(process_image, image): image for image in image_list}
            for future in tqdm(as_completed(future_to_image), total=total_images, desc="Grid Map Builder"):
                try:
                    future.result()  # 获取处理结果
                    processed_count += 1
                    # 每100张图片调用一次save_grid_map
                    if processed_count % 100 == 0:
                        self.save_grid_map()
                except Exception as exc:
                    print(f'An exception occurred: {exc}')
                    continue
        self.save_grid_map()

    def run_concurrent_visualize(self):
        # 设置非阻塞模式的图像窗口
        plt.ion()  # 打开交互模式
        # 创建一个 2x2 的网格布局
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])  # 控制布局
        self.ax1, self.ax2, self.ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

        image_list = list(self.sfm_images.values())
        import random
        random.shuffle(image_list)  # shuffle to visual rough map quickly
        # 优化1: 使用多线程数据加载
        from concurrent.futures import ThreadPoolExecutor   #, as_completed
        def process_image(image):
            pil_image = Image.open(f'{self.images_dir}/{image.name}')
            success, floor_mask = self.segment(pil_image, score_threshold=self.detect_thr,
                                               box_threshold=self.detect_thr,
                                               text_threshold=self.detect_thr)
            if success:
                rot_cw = image.qvec2rotmat()
                tvec_cw = image.tvec
                focal_length = (self.sfm_cameras[image.camera_id].params[0] /
                                self.sfm_cameras[image.camera_id].width * self.mask_resolution)
                # 将掩膜投影到y=floor_y的地平面栅格地图
                self.project_mask_to_grid2(np.array(floor_mask), rot_cw, tvec_cw, focal_length)
            return pil_image, floor_mask, image.name

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_image = {executor.submit(process_image, image): image for image in image_list}

            for index, future in enumerate(future_to_image):
                image = future_to_image[future]
                try:
                    pil_image, floor_mask, image_name = future.result()
                except Exception as exc:
                    print(f'Image {image.name} generated an exception: {exc}')
                    continue
                # 实时可视化栅格地图
                self.update_visualization(pil_image, floor_mask, index, image_name)
        plt.ioff()
        # 最后可视化或保存栅格地图
        self.save_grid_map()


if __name__ == "__main__":
    job_root = '/home/lingkun/Documents/nhrecon-server/data/shanlanlou'
    # job_root = '//mnt/1/sfm/station/station'
    config = {
        'keyword': 'floor or ground',
        'detect_thr': 0.5,
        'grid_size': 0.1,
        'expand_ratio': 1.2,
        'gravity_axis': 'y',
        'device': 'cuda',
        'mask_resolution': 400,
        'mask_ratio_thr': 0.5,
        'camera_height': 2.0,
    }
    grip_map_builder = FloorAccumulator(sfm_dir=f'{job_root}/ref',
                                        images_dir=f'{job_root}/images',
                                        config=config)
    # single-thread run with visualize
    grip_map_builder.run()
    # multi-thread run with visualize
    # grip_map_builder.run_concurrent_visualize()
    # multi-thread run
    # grip_map_builder.run_concurrent()

    grip_map_builder.to("cpu")