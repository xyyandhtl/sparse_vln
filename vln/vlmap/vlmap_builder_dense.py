import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set
import time

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
import h5py
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
from scipy.spatial.transform import Rotation as R

from sfm.tools.read_write_binary import read_images_binary, read_cameras_binary, read_points3D_binary
from ..utils.lseg_utils import get_lseg_feat
from ..utils.mapping_utils import (
    cvt_pose_vec2tf,
    load_depth_npy,
    depth2pc,
    transform_pc,
    project_point,
    get_sim_cam_mat,
    align_scale_offset
)
from ..lseg.modules.models.lseg_net import LSegEncNet

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def visualize_pc(pc: np.ndarray):
    """Take (N, 3) point cloud and visualize it using open3d.

    Args:
        pc (np.ndarray): (N, 3) point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


class VLMapBuilderSfMDense:
    def __init__(
        self,
        data_dir: Path,
        map_config: DictConfig,
        sfm_path: Path,
        rgb_path: Path,
        # depth_paths: List[Path],
        # base2cam_tf: np.ndarray,
        # base_transform: np.ndarray,
    ):
        self.data_dir = data_dir
        self.sfm_path = sfm_path
        self.rgb_path = rgb_path
        # self.depth_paths = depth_paths
        self.map_config = map_config
        # self.base2cam_tf = base2cam_tf
        # self.base_transform = base_transform
        # self.rot_type = map_config.pose_info.rot_type

        self.weight_root = Path(__file__).parent.parent.parent / 'weights'
        # self.depth_est_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_processor = AutoImageProcessor.from_pretrained(str(self.weight_root / 'depth_anything_v2'))
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(str(self.weight_root / 'depth_anything_v2'))
        self.tick = time.time()

    def print_time(self, info):
        print(f'[time] {info}: {time.time() - self.tick} s')
        self.tick = time.time()

    def create_camera_map_sfm(self):
        """
        build a map centering at the global original frame. The poses are camera pose in the global coordinate frame.
        """
        # access config info
        cs = self.map_config.cell_size
        # gs = self.map_config.grid_size
        depth_sample_rate = self.map_config.depth_sample_rate
        expand_ratio = self.map_config.expand_ratio
        # self.camera_pose_tfs = np.loadtxt(self.pose_path)
        self.sfm_cameras = read_cameras_binary(f'{self.sfm_path}/cameras.bin')
        self.sfm_images = read_images_binary(f'{self.sfm_path}/images.bin')
        self.sfm_points = read_points3D_binary(f'{self.sfm_path}/points3D.bin')
        pts_indices = np.array([self.sfm_points[key].id for key in self.sfm_points])
        pts_xyzs = np.array([self.sfm_points[key].xyz for key in self.sfm_points])
        self.points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
        self.points3d_ordered[pts_indices] = pts_xyzs
        print(f'sparse total images {len(self.sfm_images)}, total points {len(self.points3d_ordered)}')

        self.camera_pose_tfs = [cvt_pose_vec2tf(np.concatenate([image.tvec, np.roll(image.qvec, -1)])) for image in self.sfm_images.values()]
        print(f'get images poses done')

        self.map_save_dir = self.data_dir / "vlmap"
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir / "vlmaps.h5df"

        # init lseg model
        lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

        # load camera calib matrix in config
        # calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))

        max_x, max_y, sum_height = 0, 0, 0
        for sfm_image in self.sfm_images.values():
            t_cw = sfm_image.tvec
            q_cw = sfm_image.qvec
            q_cw = np.roll(q_cw, 3)
            rotation_wc = R.inv(R.from_quat(q_cw))
            sfm_pos = np.dot(-rotation_wc.as_matrix(), np.array(t_cw).T)

            max_x = max(abs(sfm_pos[0]), max_x)
            max_y = max(abs(sfm_pos[2]), max_y)
            sum_height += sfm_pos[1]
        ave_height = round(sum_height / len(self.sfm_images))
        self.pcd_min = np.array([-max_x * expand_ratio, ave_height - 2.0, -max_y * expand_ratio])
        self.pcd_max = np.array([max_x * expand_ratio, ave_height + 2.0, max_y * expand_ratio])

        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self._init_map(
            self.pcd_min, self.pcd_max, cs, self.map_save_path
        )
        print(f'applying feat box {self.pcd_min}/{self.pcd_max}, grid_map_size {occupied_ids.shape}')

        cv_map = np.zeros((self.grid_size[0], self.grid_size[2], 3), dtype=np.uint8)
        height_map = -100 * np.ones(self.grid_size[[0, 2]], dtype=np.float32)

        pbar = tqdm(
            zip(self.sfm_images.values(), self.camera_pose_tfs),
            total=len(self.sfm_images),
            desc="Get Global Map",
        )

        for frame_i, (sfm_image, camera_pose_tf) in enumerate(pbar):
            if frame_i % self.map_config.skip_frame != 0:
                continue
            if frame_i in mapped_iter_set:
                continue
            assert self.sfm_cameras[sfm_image.camera_id].model == 'PINHOLE', 'check sparse cam model not PINHOLE!'
            cam_intri = self.sfm_cameras[sfm_image.camera_id].params
            calib_mat = np.array([[cam_intri[0], 0, cam_intri[2]], [0, cam_intri[1], cam_intri[3]], [0, 0, 1]])
            image = Image.open(f'{self.rgb_path}/{sfm_image.name}')
            assert image.width == self.sfm_cameras[sfm_image.camera_id].width, 'check sparse matches the images!'
            # self.print_time('init')
            inputs = self.depth_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            # interpolate to original size
            # depth = torch.nn.functional.interpolate(
            #     predicted_depth.unsqueeze(1),
            #     size=image.size[::-1],
            #     mode="bicubic",
            #     align_corners=False,
            # )
            depth = predicted_depth.squeeze(0).numpy()
            # self.print_time('depth estimate')
            depth = align_scale_offset(depth, sfm_image, self.points3d_ordered, image.width, image.height)
            if depth is None:
                continue

            # self.print_time('depth align')
            bgr = cv2.imread(f'{self.rgb_path}/{sfm_image.name}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # # get pixel-aligned LSeg features
            pix_feats = get_lseg_feat(
                lseg_model, rgb, ["example"], lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std
            )
            pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])
            # self.print_time('pix_feats calculate')

            # backproject depth point cloud
            pc = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=100)

            # transform the point cloud to global frame (init base frame)
            transform_tf = camera_pose_tf  # @ self.habitat2cam_rot_tf
            pc_global = transform_pc(pc, transform_tf)  # (3, N)

            # # 转置 pc_global 以匹配 (N, 3) 形状
            # pc_global_T = pc_global.T
            #
            # # 创建一个 3D 图形
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # # 绘制点云
            # ax.scatter(pc_global_T[:, 0], pc_global_T[:, 1], pc_global_T[:, 2], c='b', marker='o')
            #
            # # 设置标签
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # ax.set_title('3D Point Cloud Visualization')
            #
            # # 显示图形
            # plt.show()

            # 候补method：assume cam height as fixed to alleviate large sfm drift influence
            # rotation_wc = R.inv(R.from_quat(np.roll(sfm_image.qvec, 3)))
            # floor_y = np.dot(-rotation_wc.as_matrix(), sfm_image.tvec.T)[1] + 2.0
            # print(f'estimate floor_y {floor_y}')

            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                row, height, col = np.round(((p - self.pcd_min) / cs)).astype(int)
                # height = -height    # +y is down
                # height = int((floor_y - p[1]) / cs)

                if (row >= height_map.shape[0] or col >= height_map.shape[1] or row < 0 or col < 0
                        or height >= occupied_ids.shape[1] or height < 0):
                    # print("out of range")
                    continue
                px, py, pz = project_point(calib_mat, p_local)
                rgb_v = rgb[py, px, :]
                px, py, pz = project_point(pix_feats_intr, p_local)

                if height > height_map[row, col]:
                    height_map[row, col] = height
                    cv_map[row, col, :] = rgb_v

                # when the max_id exceeds the reserved size,
                # double the grid_feat, grid_pos, weight, grid_rgb lengths
                if max_id >= grid_feat.shape[0]:
                    grid_feat, grid_pos, weight, grid_rgb = self._reserve_map_space(
                        grid_feat, grid_pos, weight, grid_rgb
                    )

                # apply the distance weighting according to
                # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
                radial_dist_sq = np.sum(np.square(p_local))
                sigma_sq = 0.6
                alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

                # update map features
                if px >= 0 and py >= 0 and px < pix_feats.shape[3] and py < pix_feats.shape[2]:
                    feat = pix_feats[0, :, py, px]
                    occupied_id = occupied_ids[row, height, col]
                    if occupied_id == -1:
                        occupied_ids[row, height, col] = max_id
                        grid_feat[max_id] = feat.flatten() * alpha
                        grid_rgb[max_id] = rgb_v
                        weight[max_id] += alpha
                        grid_pos[max_id] = [row, height, col]
                        max_id += 1
                    else:
                        if weight[occupied_id] + alpha == 0:
                            # print(f"weight/alpha {weight[occupied_id]}/{alpha}")
                            continue6
                        grid_feat[occupied_id] = (
                            grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                        ) / (weight[occupied_id] + alpha)
                        grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                            weight[occupied_id] + alpha
                        )
                        weight[occupied_id] += alpha
                    # print(f'updating r/h/c {row}/{height}/{col} grid with occupied_id {occupied_id}, alpha {alpha}, weight {weight[occupied_id]}')

            mapped_iter_set.add(frame_i)
            if frame_i % (self.map_config.skip_frame * 200) == self.map_config.skip_frame * 199:
                print(f"Temporarily saving {max_id} features at iter {frame_i}...")
                self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)
            # self.print_time('pointclouds vlm mapping')

        self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)

    def create_mobile_base_map(self):
        """
        TODO: To be implemented

        build the 3D map centering at the first base frame.
        """
        return NotImplementedError

    def _init_map(self, pcd_min: np.ndarray, pcd_max: np.ndarray, cs: float, map_path: Path) -> Tuple:
        """
        initialize a voxel grid of size grid_size = (pcd_max - pcd_min) / cs + 1
        """
        grid_size = np.ceil((pcd_max - pcd_min) / cs + 1).astype(int)  # col, height, row
        self.grid_size = grid_size
        occupied_ids = -1 * np.ones(grid_size[[0, 1, 2]], dtype=np.int32)
        grid_feat = np.zeros((grid_size[0] * grid_size[2], self.clip_feat_dim), dtype=np.float32)
        grid_pos = np.zeros((grid_size[0] * grid_size[2], 3), dtype=np.int32)
        weight = np.zeros((grid_size[0] * grid_size[2]), dtype=np.float32)
        grid_rgb = np.zeros((grid_size[0] * grid_size[2], 3), dtype=np.uint8)
        # init the map related variables
        mapped_iter_set = set()
        mapped_iter_list = list(mapped_iter_set)
        max_id = 0

        # check if there is already saved map
        if os.path.exists(map_path):
            (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs) = (
                self.load_3d_map(self.map_save_path)
            )
            mapped_iter_set = set(mapped_iter_list)
            max_id = grid_feat.shape[0]
            self.pcd_min = pcd_min
            self.pcd_max = pcd_max

        return grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id

    @staticmethod
    def load_3d_map(map_path: Union[Path, str]):
        with h5py.File(map_path, "r") as f:
            mapped_iter_list = f["mapped_iter_list"][:].tolist()
            grid_feat = f["grid_feat"][:]
            grid_pos = f["grid_pos"][:]
            weight = f["weight"][:]
            occupied_ids = f["occupied_ids"][:]
            grid_rgb = f["grid_rgb"][:]
            pcd_min = f["pcd_min"][:]
            pcd_max = f["pcd_max"][:]
            cs = f["cs"][()]
            return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs

    def _init_lseg(self):
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[2] / "weights" / "lseg"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _backproject_depth(
        self,
        depth: np.ndarray,
        calib_mat: np.ndarray,
        depth_sample_rate: int,
        min_depth: float = 0.1,
        max_depth: float = 10,
    ) -> np.ndarray:
        pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        return pc

    def _out_of_range(self, row: int, col: int, height: int, gs: int, vh: int) -> bool:
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0

    def _reserve_map_space(
        self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb

    def _reserve_grid(
        self, row: int, col: int, height: int, gs: int, vh: int, init_height_id: int, occupied_ids: np.ndarray
    ) -> Tuple[int, int]:
        if not self._out_of_range(row, col, height, gs, vh):
            print("not out of range")
            return occupied_ids, init_height_id, vh
        if height < 0:
            tmp = -np.ones((gs, gs, -height), dtype=occupied_ids.dtype)
            print("smaller than 0")
            print("before: ", occupied_ids.shape)
            occupied_ids = np.concatenate([occupied_ids, tmp], axis=2)
            print("after: ", occupied_ids.shape)
            init_height_id += -height
            vh += -height
        elif height >= vh:
            tmp = -np.ones((gs, gs, height - vh + 1), dtype=occupied_ids.dtype)
            print("larger and equal than vh")
            print("before: ", occupied_ids.shape)
            occupied_ids = np.concatenate([tmp, occupied_ids], axis=2)
            print("after: ", occupied_ids.shape)
            init_height_id += height
            vh += height
        return occupied_ids, init_height_id, vh

    def save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        with h5py.File(self.map_save_path, "w") as f:
            f.create_dataset("mapped_iter_list", data=np.array(list(mapped_iter_set), dtype=np.int32))
            f.create_dataset("grid_feat", data=grid_feat)
            f.create_dataset("grid_pos", data=grid_pos)
            f.create_dataset("weight", data=weight)
            f.create_dataset("occupied_ids", data=occupied_ids)
            f.create_dataset("grid_rgb", data=grid_rgb)
            f.create_dataset("pcd_min", data=self.pcd_min)
            f.create_dataset("pcd_max", data=self.pcd_max)
            f.create_dataset("cs", data=self.map_config.cell_size)
