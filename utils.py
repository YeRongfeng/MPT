''' Common functions used in this library.
'''
import skimage.io
import skimage.morphology as skim

import io

import numpy as np

from ompl import base as ob

USE_GPU = True  # 是否使用GPU加速
if USE_GPU:
    try:
        import cupy as cp
    except ImportError:
        USE_GPU = False
        cp = None  # 如果无法导入cupy，则禁用GPU加速
# Note: The above import of cupy is conditional based on USE_GPU.
# This allows the module to be imported even without cupy installed.

# OMPL imports moved to where they're actually needed
# This allows the module to be imported even without OMPL installed


def png_decoder(key, value):
    '''
    PNG decoder with gray images.
    :param key:
    :param value:
    '''
    if not key.endswith(".png"):
        return None
    assert isinstance(value, bytes)
    return skimage.io.imread(io.BytesIO(value), as_gray=True)


def cls_decoder(key, value):
    '''
    Converts class represented as bytes to integers.
    :param key:
    :param value:
    :returns the decoded value
    '''
    if not key.endswith(".cls"):
        return None
    assert isinstance(value, bytes)
    return int(value)


def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    NOTE: The Pixel co-ordinates are represented as follows:
    (0,0)------ X ----------->|
    |                         |  
    |                         |  
    |                         |  
    |                         |  
    Y                         |
    |                         |
    |                         |  
    v                         |  
    ---------------------------  
    """
    return (np.int(np.floor(pos[0]/res)), np.int(size[0]-1-np.floor(pos[1]/res)))


class ValidityChecker(ob.StateValidityChecker):
    '''A class to check if an obstacle is in collision or not.
    '''
    def __init__(self, si, CurMap, MapMask=None, res=0.05, robot_radius=0.1):
        '''
        Intialize the class object, with the current map and mask generated
        from the transformer model.
        :param si: an object of type ompl.base.SpaceInformation
        :param CurMap: A np.array with the current map.
        :param MapMask: Areas of the map to be masked.
        '''
        super().__init__(si)
        self.size = CurMap.shape
        # Dilate image for collision checking
        InvertMap = np.abs(1-CurMap)
        InvertMapDilate = skim.dilation(InvertMap, skim.disk(robot_radius/res))
        MapDilate = abs(1-InvertMapDilate)
        if MapMask is None:
            self.MaskMapDilate = MapDilate>0.5
        else:
            self.MaskMapDilate = np.logical_and(MapDilate, MapMask)
            
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid.
        '''
        pix_dim = geom2pix(state, size=self.size)
        return self.MaskMapDilate[pix_dim[1], pix_dim[0]]
    
# GPU加速的碰撞检测函数
def gpu_collision_check(points, map_data, robot_radius, resolution):
    """
    使用GPU进行批量碰撞检测
    :param points: (N, 2)数组，要检查的点坐标
    :param map_data: 2D地图数组 (0=空闲, 1=障碍)
    :param robot_radius: 机器人的半径
    :param resolution: 地图的分辨率
    :return: (N,)布尔数组，True表示安全，False表示碰撞
    """
    if not USE_GPU or not cp:
        # GPU不可用时回退到CPU检测
        return cpu_collision_check(points, map_data, robot_radius, resolution)
    
    try:
        # 将地图数据复制到GPU
        d_map = cp.asarray(map_data)
        h, w = map_data.shape
        
        # 将点转换到像素坐标
        points_px = (points / resolution).astype(int)
        points_px = cp.asarray(points_px)
        
        # 创建安全点数组
        safe_points = cp.ones(len(points), dtype=cp.bool_)
        
        # 检查点是否在边界内
        in_bounds = (points_px[:, 0] >= 0) & (points_px[:, 0] < w) & \
                    (points_px[:, 1] >= 0) & (points_px[:, 1] < h)
        
        # 对于边界内的点，检查地图值
        safe_points[in_bounds] = d_map[points_px[in_bounds, 1], points_px[in_bounds, 0]] == 0
        
        # 检查机器人半径范围内的点
        radius_px = int(robot_radius / resolution)
        for dx in range(-radius_px, radius_px+1):
            for dy in range(-radius_px, radius_px+1):
                if dx == 0 and dy == 0:
                    continue
                offset = points_px + cp.array([dx, dy])
                valid = (offset[:, 0] >= 0) & (offset[:, 0] < w) & \
                        (offset[:, 1] >= 0) & (offset[:, 1] < h)
                
                safe_points[valid] &= d_map[offset[valid, 1], offset[valid, 0]] == 0
        
        return cp.asnumpy(safe_points)
    
    except Exception as e:
        print(f"GPU collision check failed: {str(e)}, falling back to CPU")
        return cpu_collision_check(points, map_data, robot_radius, resolution)
    
# CPU碰撞检测函数
def cpu_collision_check(points, map_data, robot_radius, resolution):
    """
    使用CPU进行碰撞检测
    :param points: (N, 2)数组，要检查的点坐标
    :param map_data: 2D地图数组 (0=空闲, 1=障碍)
    :param robot_radius: 机器人的半径
    :param resolution: 地图的分辨率
    :return: (N,)布尔数组，True表示安全，False表示碰撞
    """
    h, w = map_data.shape
    valid_points = []
    radius_px = int(robot_radius / resolution)
    
    for point in points:
        x, y = point
        px = int(x / resolution)
        py = int(y / resolution)
        
        if px < 0 or px >= w or py < 0 or py >= h:
            valid_points.append(False)
            continue
            
        # 检查机器人半径范围内的所有点
        valid = True
        for dx in range(-radius_px, radius_px+1):
            for dy in range(-radius_px, radius_px+1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if map_data[ny, nx] != 0:  # 障碍物
                        valid = False
                        break
            if not valid:
                break
                
        valid_points.append(valid)
        
    return np.array(valid_points)

# class ValidityChecker(ob.StateValidityChecker):
#     '''A class to check if an obstacle is in collision or not.
#     '''
#     def __init__(self, si, CurMap, MapMask=None, res=0.05, robot_radius=0.1):
#         '''
#         Initialize the class object, with the current map and mask generated
#         from the transformer model.
#         :param si: an object of type ompl.base.SpaceInformation
#         :param CurMap: A np.array with the current map.
#         :param MapMask: Areas of the map to be masked.
#         '''
#         super().__init__(si)
#         self.si = si  # 保存SpaceInformation引用
#         self.size = CurMap.shape
#         self.resolution = res
#         self.robot_radius = robot_radius
        
#         # 预处理地图
#         self.process_map(CurMap, MapMask)
    
#     def getSpaceInformation(self):
#         """获取SpaceInformation对象"""
#         return self.si
    
#     def process_map(self, CurMap, MapMask=None):
#         """
#         预处理地图数据，为GPU加速做准备
#         """
#         # 膨胀图像用于碰撞检测
#         InvertMap = np.abs(1 - CurMap)
#         InvertMapDilate = skim.dilation(InvertMap, skim.disk(self.robot_radius / self.resolution))
#         MapDilate = abs(1 - InvertMapDilate)
        
#         if MapMask is None:
#             self.MaskMapDilate = MapDilate > 0.5
#         else:
#             self.MaskMapDilate = np.logical_and(MapDilate, MapMask)
        
#         # 转换为0/1值用于GPU加速
#         self.binary_map = self.MaskMapDilate.astype(np.uint8)
        
#         # 如果使用GPU，将地图复制到GPU
#         if USE_GPU and cp:
#             try:
#                 self.gpu_map = cp.asarray(self.binary_map)
#             except:
#                 self.gpu_map = None
#         else:
#             self.gpu_map = None
    
#     def isValid(self, state):
#         '''
#         Check if the given state is valid.
#         :param state: An ob.State object to be checked.
#         :returns bool: True if the state is valid.
#         '''
#         # 提取坐标
#         x = state[0]
#         y = state[1]
        
#         # 使用GPU加速检查
#         if self.gpu_map is not None:
#             return self.gpu_isValid(x, y)
#         else:
#             return self.cpu_isValid(x, y)
    
#     # def gpu_isValid(self, x, y):
#     #     """
#     #     GPU加速的单个点碰撞检测
#     #     """
#     #     try:
#     #         # 转换到像素坐标
#     #         px = int(x / self.resolution)
#     #         py = int(y / self.resolution)
#     #         h, w = self.binary_map.shape
            
#     #         # 检查边界
#     #         if px < 0 or px >= w or py < 0 or py >= h:
#     #             return False
                
#     #         # 检查机器人半径范围内的点
#     #         radius_px = int(self.robot_radius / self.resolution)
#     #         for dx in range(-radius_px, radius_px+1):
#     #             for dy in range(-radius_px, radius_px+1):
#     #                 nx, ny = px + dx, py + dy
#     #                 if 0 <= nx < w and 0 <= ny < h:
#     #                     if self.gpu_map[ny, nx] != 0:  # 障碍物
#     #                         return False
#     #         return True
            
#     #     except Exception as e:
#     #         print(f"GPU single point check failed: {str(e)}, falling back to CPU")
#     #         return self.cpu_isValid(x, y)
    
#     def gpu_isValid(self, x, y):
#         """
#         GPU加速的单个点碰撞检测 (优化版，假设地图已预膨胀)
#         """
#         try:
#             px = int(x / self.resolution)
#             py = int(y / self.resolution)
#             h, w = self.binary_map.shape

#             if px < 0 or px >= w or py < 0 or py >= h:
#                 return False

#             # 直接查询预膨胀的地图
#             return self.gpu_map[py, px] == 0 # 0 表示空闲，非0表示障碍物
#         except Exception as e:
#             print(f"GPU single point check failed: {str(e)}, falling back to CPU")
#             return self.cpu_isValid(x, y) # 注意这里还是会回到原始的cpu_isValid，如果它逻辑有问题，那这里也会出问题
    
#     def cpu_isValid(self, x, y):
#         """
#         CPU实现的单个点碰撞检测
#         """
#         pix_dim = geom2pix((x, y), self.resolution, self.size)
#         return self.MaskMapDilate[pix_dim[1], pix_dim[0]]
    
#     def batch_isValid(self, states):
#         """
#         批量检查多个状态的有效性
#         :param states: 状态列表 [(x1,y1), (x2,y2), ...]
#         :return: 有效性列表 [bool1, bool2, ...]
#         """
#         points = np.array([(s[0], s[1]) for s in states])
#         return gpu_collision_check(
#             points, 
#             self.binary_map,
#             self.robot_radius,
#             self.resolution
#         )