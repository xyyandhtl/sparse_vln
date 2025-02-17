from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from .vlmap import VLMap
from ..utils.categories import (mp3dcat, cat_building, cat_building_ch, cat_emergency,
                                cat_station, cat_station_simple)
from ..utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent),
    config_name="vlmaps.yaml",
)
def index_map(config: DictConfig) -> None:
    categories = cat_station_simple
    vlmap = VLMap(config)
    vlmap.load_map('/home/lingkun/Documents/nhrecon-server/data/station')
    # visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)
    print("considering categories: ")
    # print(building3dcat[1:-1])
    print(categories)
    # cat = input("What is your interested category in this scene?")
    cat = "main gate"

    vlmap._init_clip()
    if config.init_categories:
        vlmap.init_categories(categories)
        mask = vlmap.index_map(cat, with_init_cat=True)
    else:
        mask = vlmap.index_map(cat, with_init_cat=False)

    if config.index_2d:
        grid_map_size = max(vlmap.occupied_ids.shape[0], vlmap.occupied_ids.shape[2])
        mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, grid_map_size)
        rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, grid_map_size)
        # visualize_masked_map_2d(rgb_2d, mask_2d)
        heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.cell_size, decay_rate=config.decay_rate)
        visualize_heatmap_2d(rgb_2d, heatmap, transparency=0.5)
    else:
        visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
        heatmap = get_heatmap_from_mask_3d(
            vlmap.grid_pos, mask, cell_size=config.cell_size, decay_rate=config.decay_rate
        )
        visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb, transparency=0.2)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent),
    config_name="vlmaps.yaml",
)
def vlm_nav(config: DictConfig) -> None:
    vlmap = VLMap(config)
    vlmap.load_map('/home/lingkun/Documents/nhrecon-server/data/shanlanlou')
    # visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)

    vlmap._init_clip()
    vlmap.init_categories(cat_building)
    # rgb_2d = np.zeros((vlmap.occupied_ids.shape[0], vlmap.occupied_ids.shape[2], 3), dtype=np.uint8)
    # height = -100 * np.ones((vlmap.occupied_ids.shape[0], vlmap.occupied_ids.shape[2]), dtype=np.int32)
    # for i, pos in enumerate(vlmap.grid_pos):
    #     # row, col, h = pos
    #     row, h, col = pos
    #     if h > height[row, col]:
    #         rgb_2d[row, col] = vlmap.grid_rgb[i]
    #         height[row, col] = h
    # cv2.imwrite('test.png', np.transpose(rgb_2d, (1, 0, 2)))

    print("nav considering categories: ")
    # print(building3dcat[1:-1])
    print(cat_building)
    # cat = input("What is your interested category in this scene?")
    cur_pos = [30, 20]
    target_name = "door"
    pos = vlmap.get_nearest_pos(curr_pos=cur_pos, name=target_name)



if __name__ == "__main__":
    index_map()
    # vlm_nav()
