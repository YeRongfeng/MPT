''' Generate a map using matplotlib and save it.
'''
import sys
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import shutil

import numpy as np
import skimage.morphology as skim
from skimage import io
from skimage import color
import pickle
import os
from os import path as osp

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

from utils import geom2pix, ValidityChecker
from generateMaps import generate_random_maps
from generateMazeMaps import generate_random_maze

import argparse

# All measurements are mentioned in meters
# Define global parameters
length = 24 # Size of the map
robot_radius = 0.1
dist_resl = 0.05


def get_path(start, goal, ValidityCheckerObj=None):
    '''
    Get a RRT path from start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    :param ValidityCheckerObj: An object of class ompl.base.StateValidityChecker
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    mapSize = ValidityCheckerObj.size
    # Define the space
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(0, mapSize[1]*dist_resl) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl) # Set height bounds (y)
    space.setBounds(bounds)
    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityCheckerObj)

    success = False
    # Create a simple setup
    ss = og.SimpleSetup(si)

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal, 0.1)

    # # Use RRT*
    planner = og.RRTstar(si)

    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time = 4
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(2.0)
        time +=3
        if time>240:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        # Define path
        # Get the number of interpolation points
        num_points = int(4*ss.getSolutionPath().length()//(dist_resl*32))
        ss.getSolutionPath().interpolate(num_points)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
            ])
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]
        path_interpolated = []

    return np.array(path), np.array(path_interpolated), success


def start_experiment_rrt(start, samples, fileDir=None):
    '''
    Run the experiment for random start and goal points.
    :param start: The start index of the samples
    :param samples: The number of samples to collect
    :param fileDir: Directory with the map and paths
    '''
    assert osp.isdir(fileDir), f"{fileDir} is not a valid directory"

    envNum = int(fileDir[-6:])
    CurMap = io.imread(osp.join(fileDir, f'map_{envNum}.png'), as_gray=True)
    mapSize = CurMap.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(0, mapSize[1]*dist_resl) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl) # Set height bounds (y)
    space.setBounds(bounds)

    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    # Validity checking
    ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
    si.setStateValidityChecker(ValidityCheckerObj)

    for i in range(start, start+samples):
        path_param = {}
        # Define the start and goal location
        start = ob.State(space)
        start.random()
        while not ValidityCheckerObj.isValid(start()):
            start.random()
        goal = ob.State(space)
        goal.random()
        while not ValidityCheckerObj.isValid(goal()):   
            goal.random()

        path, path_interpolated, success = get_path(start, goal, ValidityCheckerObj)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open(osp.join(fileDir,f'path_{i}.p'), 'wb'))

def start_experiment_rrtrealWorld(start, samples, mapFile, fileDir=None):
    '''
    Run the experiment for random start and goal points on the real world environment
    :param start: The start index of the samples
    :param samples: The number of samples to collect
    :param fileDir: Directory with the map and paths
    '''
    assert osp.isdir(fileDir), f"{fileDir} is not a valid directory"

    CurMap = io.imread(mapFile, as_gray=True)
    mapSize = CurMap.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    # Set bounds away from  boundary to avoid sampling points outside the map
    bounds.setLow(2.0)
    bounds.setHigh(0, mapSize[1]*dist_resl-2) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl-2) # Set height bounds (y)
    space.setBounds(bounds)

    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    # Validity checking
    ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
    si.setStateValidityChecker(ValidityCheckerObj)

    for i in range(start, start+samples):
        path_param = {}
        # Define the start and goal location
        start = ob.State(space)
        start.random()
        while not ValidityCheckerObj.isValid(start()):
            start.random()
        goal = ob.State(space)
        goal.random()
        while not ValidityCheckerObj.isValid(goal()):   
            goal.random()

        path, path_interpolated, success = get_path(start, goal, ValidityCheckerObj)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open(osp.join(fileDir,f'path_{i}.p'), 'wb'))


def start_map_collection_rrt(start, samples, envType, numPaths, fileDir, mapFile, height, width):
    '''
    Collect a single path for the given number of samples.
    :param start: The start index of the samples.
    :param samples: The number of samples to collect.
    :param envType: The type of environment to set up.
    :param numPaths: The number of paths to collect for each environment.
    :param fileDir: The directory to save the paths
    :param mapFile: Provide the location of the map file.
    :param height: The height of the map in pixels
    :param width: The width of the map in pixels
    '''
    if envType =='realworld':
        assert mapFile is not None, "Need to set a map for planning"
        start_experiment_rrtrealWorld(0, numPaths, mapFile, fileDir)
    else:
        for i in range(start, start+samples):
                envFileDir = osp.join(fileDir, f'env{i:06d}')
                if not osp.isdir(envFileDir):
                    os.mkdir(envFileDir)
                fileName = osp.join(envFileDir, f'map_{i}.png')
                if envType=='forest':
                    generate_random_maps(width=width*0.05, height=height*0.05, seed=1000+i, fileName=fileName, num_circle=100, num_box=100)
                if envType=='maze':
                    generate_random_maze(width=width*0.05, height=height*0.05, wt=1, pw=1.875, seed=i, fileName=fileName)

                start_experiment_rrt(0, numPaths, fileDir=envFileDir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help='Start of the sample index', required=True, type=int)
    parser.add_argument('--numEnv', help='Number of Environments to collect', required=True, type=int)
    parser.add_argument('--envType', help='Type of environment', choices=['maze', 'forest', 'realworld'])
    parser.add_argument('--numPaths', help='Number of paths to collect', default=1, type=int)
    parser.add_argument('--fileDir', help='The Folder to save the files', required=True)
    parser.add_argument('--mapFile', help='Need to provide mapFile, if generating data for real world maps')
    parser.add_argument('--height', help='The height of the map in pixels', type=int, default=480)
    parser.add_argument('--width', help='The width of the map in pixels', type=int, default=480) 
    args = parser.parse_args()

    # start_map_collection_rrt(args.start, args.numEnv, args.envType, args.numPaths, args.fileDir, args.mapFile, args.height, args.width)
    
    # 添加进度条
    pbar = tqdm(total=args.numEnv, desc="Processing Environments")
    
    # 使用多进程代替多线程
    max_workers = min(os.cpu_count(), 12)  # 限制最大进程数
    
    # 封装任务函数
    def process_environment(env_index, env_type, base_file_dir, 
                            map_file, map_height, map_width, num_paths, dist_resl):
        """
        参数说明：
        env_index: 环境编号
        env_type: 环境类型（maze/forest/realworld）
        base_file_dir: 保存环境的基础文件夹路径
        map_file: 真实环境地图文件路径（仅 realworld 使用）
        map_height: 地图高度（像素）
        map_width: 地图宽度（像素）
        num_paths: 每个环境生成的路径数量
        dist_resl: 距离分辨率（米/像素）
        """
        # 创建环境目录
        envFileDir = osp.join(base_file_dir, f'env{env_index:06d}')
        os.makedirs(envFileDir, exist_ok=True)
        
        # 根据环境类型处理地图
        if env_type == 'realworld':
            # 确保有地图文件
            if not osp.isfile(map_file):
                raise FileNotFoundError(f"Realworld map file not found: {map_file}")
            
            # 创建地图副本到环境目录
            env_map_file = osp.join(envFileDir, f'map_{env_index}.png')
            shutil.copy(map_file, env_map_file)
            
            # 读取地图
            CurMap = io.imread(env_map_file, as_gray=True)
        else:
            # 生成新地图
            env_map_file = osp.join(envFileDir, f'map_{env_index}.png')
            
            if env_type == 'forest':
                # width, height: 地图实际尺寸（米），像素乘以0.05为实际米数
                # seed: 随机种子，保证每个环境唯一
                # fileName: 保存地图的文件名
                # num_circle: 随机生成的圆形障碍物数量
                # num_box: 随机生成的方形障碍物数量
                generate_random_maps(
                    width=map_width * 0.05,  # 地图宽度（米）
                    height=map_height * 0.05, # 地图高度（米）
                    seed=1000 + env_index,    # 随机种子
                    fileName=env_map_file,    # 地图文件名
                    num_circle=69,           # 圆形障碍物数量
                    num_box=29               # 方形障碍物数量
                )
            elif env_type == 'maze':
                # width, height: 地图实际尺寸（米）
                # wt: 墙体宽度（米）
                # pw: 通道宽度（米）
                # seed: 随机种子
                # fileName: 保存地图的文件名
                generate_random_maze(
                    width=map_width * 0.05,   # 地图宽度（米）
                    height=map_height * 0.05, # 地图高度（米）
                    wt=1,                     # 墙体宽度（米）
                    pw=1.875,                 # 通道宽度（米）
                    seed=env_index,           # 随机种子
                    fileName=env_map_file     # 地图文件名
                )
            
            # 读取生成的地图
            CurMap = io.imread(env_map_file, as_gray=True)
        
        mapSize = CurMap.shape
        
        # 初始化规划空间
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0.0)
        bounds.setHigh(0, mapSize[1] * dist_resl)  # 宽度边界 (x)
        bounds.setHigh(1, mapSize[0] * dist_resl)  # 高度边界 (y)
        space.setBounds(bounds)
        
        # 创建状态有效性检查器
        si = ob.SpaceInformation(space)
        ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
        si.setStateValidityChecker(ValidityCheckerObj)
        
        # 为当前环境收集多条路径
        path_results = []
        for path_idx in range(num_paths):
            path_param = {}
            
            # 随机起点
            start_state = ob.State(space)
            start_state.random()
            while not ValidityCheckerObj.isValid(start_state()):
                start_state.random()
                
            # 随机终点
            goal_state = ob.State(space)
            goal_state.random()
            while not ValidityCheckerObj.isValid(goal_state()):
                goal_state.random()
                
            # 路径规划
            try:
                path, path_interpolated, success = get_path(start_state, goal_state, ValidityCheckerObj)
                path_param = {
                    'path': path,
                    'path_interpolated': path_interpolated,
                    'success': success
                }
            except Exception as e:
                print(f"Error on env {env_index} path {path_idx}: {str(e)}")
                path_param = {
                    'path': [], 
                    'path_interpolated': [], 
                    'success': False,
                    'error': str(e)
                }
            
            # 保存单条路径结果
            path_file = osp.join(envFileDir, f'path_{path_idx}.p')
            pickle.dump(path_param, open(path_file, 'wb'))
            path_results.append(path_param)
            
        return f"Env {env_index} completed with {sum(r['success'] for r in path_results)}/{num_paths} successful paths"
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(args.start, args.start + args.numEnv):
            # 提交单个环境+多条路径的任务
            future = executor.submit(
                process_environment,
                env_index=i,
                env_type=args.envType,
                base_file_dir=args.fileDir,
                map_file=args.mapFile,
                map_height=args.height,
                map_width=args.width,
                num_paths=args.numPaths,
                dist_resl=dist_resl  # 传递全局参数
            )
            futures[future] = i
            
        # 处理任务结果
        completed_count = 0
        for future in as_completed(futures):
            try:
                res = future.result(timeout=300)  # 5分钟超时
                pbar.set_postfix_str(res)
                completed_count += 1
            except TimeoutError:
                env_idx = futures[future]
                print(f"\nTask for env {env_idx} timed out")
            except Exception as e:
                env_idx = futures[future]
                print(f"\nError processing env {env_idx}: {str(e)}")
            finally:
                pbar.update(1)
    
    pbar.close()
    print(f"Completed {completed_count}/{args.numEnv} environments")