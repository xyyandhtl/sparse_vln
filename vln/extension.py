import os

from .utils.extra_util import print_title
from .nav_map.grid_map import FloorAccumulator
from .nav_map.mesh_generator import GroundObstacleMesh
from .vlmap.vlmap import create_vlmap


class VLMApplications:
    def __init__(self, work_dir, vlm_config):
        print_title(f'Construct: {self.__class__.__name__}')
        self.work_dir = work_dir
        self.map_name = os.path.basename(self.work_dir)
        self.sfm_dir = f'{work_dir}/ref'
        self.images_dir = f'{work_dir}/images'

        # self.ground_keyword = vlm_config['ground_keyword']
        self.config = vlm_config
        self.grid_map_config = vlm_config['grid_map']

        self.callback_update_progress = None

    def register_callback(self, callback):
        self.callback_update_progress = callback

    def run(self):
        print_title(f'Run: {self.__class__.__name__}')
        print(f'grid_map config {self.grid_map_config}')
        grip_map_builder = FloorAccumulator(sfm_dir=self.sfm_dir, images_dir=self.images_dir, config=self.grid_map_config)
        grip_map_builder.run()
        # grip_map_builder.run_concurrent()
        auto_mesh_builder = GroundObstacleMesh(sfm_dir=self.sfm_dir, config=self.grid_map_config)
        auto_mesh_builder.generate_mesh()

        from hydra import initialize, compose
        with initialize(config_path='./vlmap', version_base=None):
            config = compose(config_name="vlmaps.yaml")
        # share the config to syns the map pixel coord
        config.cell_size = self.grid_map_config['grid_size']
        config.expand_ratio = self.grid_map_config['expand_ratio']
        config.camera_height = self.grid_map_config['camera_height']
        config.gravity_axis = self.grid_map_config['gravity_axis']
        print(f'vlmap_builder config {config}')
        create_vlmap(map_path=self.work_dir, config=config)


if __name__ == "__main__":
    import yaml
    with open('../config/vlm.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    vlm_app = VLMApplications('/home/lingkun/Documents/nhrecon-server/data/shanlanlou', cfg)
    vlm_app.run()

