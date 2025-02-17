# 基于稀疏重建的视觉-语言导航应用功能

### 一、路径规划SDK
- 语义引导的自动化生成栅格地图，[examples](docs/imgs)
```text
for every sfm_images：
    grounding_dino检测floor类语义
    sam_vit分割floor框
    if 未检测到 or 置信度不高：
        continue
    for 每个像素：
        相机位姿+分割mask像素射线投影至估计地面水平面
        if 有交点：
            更新栅格占据概率
            更新观测数
保存栅格地图
```
todo：大地图多grid_map局部联通跨层导航

### 二、VLM语义地图
- CLIP表征的隐式语义地图构建，[result.md](docs/heatmap/result.md)
```text
vln/vlmap/vlamap_builder_dense.py
基于depth-anything-V2的稠密像素Vlmap
读取sparse_model
估计2d bonding box
计算height grid_size
for every sfm_images：
    depth-anything-V2深度估计
    尺度估计
        用逆深度而不用深度，因depth-anything-V2内在原理，已验证过拟合效果较好
        深度图refer to gt sparse points3d，算法参见3dgs源码
        ransac估计scale/offset
    真实尺度深度图映射grid_map
        lseg预训练模型计算稠密像素跨模态语义表征
        映射至grid map with height
        occupied_id += 1
保存save_3d_map隐式语义地图
```
```text
vln/vlmap/vlamap_builder_sparse.py
基于sparse points3d的低内存占用Vlmap
读取sparse_model
估计2d bonding box
计算height grid_size
for every sfm_images：
    取出points3d
    按track_lenght和reproj_error过滤掉主观unreliable
    映射grid_map
        lseg预训练模型计算稠密像素跨模态语义表征
        映射至grid map with height
        occupied_id += 1
保存save_3d_map隐式语义地图
```
- 零样本开放词汇语义搜索
```text
vln/vlmap/index_map.py
读取load_3d_map隐式语义地图
if 临时预定义词汇分类
    预定义词汇表扩展描述的clip_embedding
    3d_grid_map余弦距离点乘
    2d_grid的height_average
    最大相似度的归类
else:
    输入零样本文本query
    扩展描述以及others的clip_embedding
    3d_grid_map余弦距离点乘
    2d_grid的height_average
    最大相似度二分类
query文本的2d_grid像素归类点集聚合和过滤
获得query文本的目标区域
```
todo: config统一，和路径规划串联，用隐式语义搜索无需预定义的目标，转换栅格坐标，用导航SDK路径规划

### 三、VLN视觉语言导航
- todo: to be implemented

### 四、运行
```shell
# 环境
pip install requirements.txt
```
```shell
# 模块化运行
python -m vln.extension
```
