''' 
Generate a forest environment, and collect paths using sst on the environment - Parallel Version
'''

import numpy as np
import sys
import skimage.morphology as skim
from skimage import io
import pickle
import os
from os import path as osp
from math import sin, cos, tan
from functools import partial
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm

try:
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All measurements are mentioned in meters
# Define global parameters
length = 24  # Size of the map
robot_radius = 0.2
dist_resl = 0.05
carLength = 0.3
EPSILON = 1e-3  # Boundary buffer

# Define the space
space = ob.SE2StateSpace()

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(0)
bounds.setHigh(length)
space.setBounds(bounds)

cspace = oc.RealVectorControlSpace(space, 2)
cbounds = ob.RealVectorBounds(2)
cbounds.setLow(0, 0.0)
cbounds.setHigh(0, .3)
cbounds.setLow(1, -.5)
cbounds.setHigh(1, .5)
cspace.setBounds(cbounds)

# Define the SpaceInformation object.
si = ob.SpaceInformation(space)

class ValidityChecker(ob.StateValidityChecker):
    '''A class to check if an obstacle is in collision or not.'''
    def __init__(self, si, CurMap, MapMask=None, res=0.05, robot_radius=robot_radius):
        '''
        Initialize the class object with the current map.
        :param si: an object of type ompl.base.SpaceInformation
        :param CurMap: A np.array with the current map.
        '''
        super().__init__(si)
        self.size = CurMap.shape
        # Dilate image for collision checking
        InvertMap = np.abs(1 - CurMap)
        InvertMapDilate = skim.dilation(InvertMap, skim.disk((robot_radius + 0.1) / res))
        MapDilate = abs(1 - InvertMapDilate)
        if MapMask is None:
            self.MaskMapDilate = MapDilate > 0.5
        else:
            self.MaskMapDilate = np.logical_and(MapDilate, MapMask)
            
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid.
        '''
        x, y = state.getX(), state.getY()
        # Ensure state is within bounds
        if x < 0 or x >= length or y < 0 or y >= length:
            return False
            
        pix_dim = geom2pix([x, y], size=self.size)
        if pix_dim[0] < 0 or pix_dim[0] >= self.size[0] or pix_dim[1] < 0 or pix_dim[1] >= self.size[1]:
            return False
        return self.MaskMapDilate[pix_dim[1], pix_dim[0]]

def geom2pix(pos, size):
    """
    Convert geometric coordinates to pixel coordinates.
    :param pos: position (x, y)
    :param size: image size (height, width)
    :return: pixel coordinates (x_pixel, y_pixel)
    """
    x, y = pos
    pix_x = int(x * size[1] / length)
    pix_y = int(y * size[0] / length)
    return pix_x, pix_y

def kinematicCarODE(q, u, qdot):
    '''Define the ODE of the car.'''
    theta = q[2]
    qdot[0] = u[0] * cos(theta)
    qdot[1] = u[0] * sin(theta)
    qdot[2] = u[0] * tan(u[1]) / carLength

def get_path(start, goal, ValidityCheckerObj, max_time=500):
    '''
    Get a path from start to goal using SST.
    :param start: ob.State object.
    :param goal: ob.State object.
    :param ValidityCheckerObj: An object of class ompl.base.StateValidityChecker
    :param max_time: float max seconds for planning 
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    def isStateValid(spaceInformation, state):
        return spaceInformation.satisfiesBounds(state) and ValidityCheckerObj.isValid(state)
    
    success = False
    # Create a simple setup
    ss = oc.SimpleSetup(cspace)

    validityChecker = ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    ss.setStateValidityChecker(validityChecker)

    # Set the start and goal states with boundary checks
    start_state = ob.State(space)
    start_state().setX(max(EPSILON, min(length - EPSILON, start().getX())))
    start_state().setY(max(EPSILON, min(length - EPSILON, start().getY())))
    start_state().setYaw(start().getYaw())
    
    goal_state = ob.State(space)
    goal_state().setX(max(EPSILON, min(length - EPSILON, goal().getX())))
    goal_state().setY(max(EPSILON, min(length - EPSILON, goal().getY())))
    goal_state().setYaw(goal().getYaw())
    
    # Verify start and goal are valid
    if not ValidityCheckerObj.isValid(start_state()):
        logger.warning(f"Invalid start state: {start_state().getX()}, {start_state().getY()}")
        return np.array([[start().getX(), start().getY(), start().getYaw()],
                         [goal().getX(), goal().getY(), goal().getYaw()]]), np.array([]), False
    
    if not ValidityCheckerObj.isValid(goal_state()):
        logger.warning(f"Invalid goal state: {goal_state().getX()}, {goal_state().getY()}")
        return np.array([[start().getX(), start().getY(), start().getYaw()],
                         [goal().getX(), goal().getY(), goal().getYaw()]]), np.array([]), False
    
    ss.setStartAndGoalStates(start_state, goal_state, 2.0)

    ode = oc.ODE(kinematicCarODE)
    odeSolver = oc.ODEBasicSolver(ss.getSpaceInformation(), ode)
    propagator = oc.ODESolver.getStatePropagator(odeSolver)
    ss.setStatePropagator(propagator)
    ss.getSpaceInformation().setPropagationStepSize(0.1)
    ss.getSpaceInformation().setMinMaxControlDuration(1, 20)

    # Use SST
    planner = oc.SST(ss.getSpaceInformation())
    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time_inc = 60
    time = time_inc
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(time_inc)
        time += time_inc
        if time > max_time:
            break
            
    if ss.haveExactSolutionPath():
        success = True
        logger.info("Found solution")
        path = [
            [ss.getSolutionPath().getState(i).getX(), 
             ss.getSolutionPath().getState(i).getY(), 
             ss.getSolutionPath().getState(i).getYaw()]
            for i in range(ss.getSolutionPath().getStateCount())
        ]
        
        # Define path
        ss.getSolutionPath().interpolate()
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i).getX(), 
             path_obj.getState(i).getY(), 
             path_obj.getState(i).getYaw()] 
            for i in range(path_obj.getStateCount())
        ])        
    else:
        path = [[start().getX(), start().getY(), start().getYaw()], 
                [goal().getX(), goal().getY(), goal().getYaw()]]
        path_interpolated = []
        logger.warning("No solution found")

    return np.array(path), np.array(path_interpolated), success

def generate_random_map(env_id, fileDir):
    '''Generate a random map for the environment'''
    from generateMaps import generate_random_maps
    env_path = osp.join(fileDir, f'env{env_id:06d}')
    os.makedirs(env_path, exist_ok=True)
    fileName = osp.join(env_path, f'map_{env_id}.png')
    generate_random_maps(width=length, seed=env_id+200, fileName=fileName)
    return fileName

def process_environment(env_id, fileDir, num_paths):
    '''Process a single environment: generate map and collect paths'''
    try:
        logger.info(f"Processing environment {env_id}")
        env_path = osp.join(fileDir, f'env{env_id:06d}')
        
        # Generate map
        map_file = generate_random_map(env_id, fileDir)
        CurMap = io.imread(map_file, as_gray=True)
        
        # Validity checking
        ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
        si.setStateValidityChecker(ValidityCheckerObj)
        
        # Collect paths
        for i in range(num_paths):
            path_param = {}
            success = False
            while not success:
                sg_ok = False
                while not sg_ok:
                    # Define the start and goal location
                    start_state = ob.State(space)
                    start_state.random()
                    while not ValidityCheckerObj.isValid(start_state()):
                        start_state.random()
                        
                    goal_state = ob.State(space)
                    goal_state.random()
                    while not ValidityCheckerObj.isValid(goal_state()):   
                        goal_state.random()
                        dist = np.sqrt((start_state().getX() - goal_state().getX()) ** 2 + 
                                      (start_state().getY() - goal_state().getY()) ** 2)
                        if dist > 4 and dist < 15:
                            sg_ok = True

                path, path_interpolated, success = get_path(start_state, goal_state, ValidityCheckerObj)
                
            if success:
                path_param['path'] = path
                path_param['path_interpolated'] = path_interpolated
                path_param['success'] = success

                path_file = osp.join(env_path, f'path_{i}.p')
                with open(path_file, 'wb') as f:
                    pickle.dump(path_param, f)
                logger.info(f"Generated path {i} for env {env_id}")
        
        return env_id, num_paths
    except Exception as e:
        logger.error(f"Error processing environment {env_id}: {str(e)}")
        return env_id, 0

def parallel_map_collection(start, num_envs, num_paths, fileDir, num_workers):
    '''Parallel collection of environments and paths'''
    os.makedirs(fileDir, exist_ok=True)
    env_ids = list(range(start, start + num_envs))
    results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_environment, env_id, fileDir, num_paths): env_id for env_id in env_ids}
        
        with tqdm(total=len(env_ids), desc="Processing environments") as pbar:
            for future in as_completed(futures):
                env_id = futures[future]
                try:
                    env_id_result, paths_generated = future.result()
                    results[env_id] = paths_generated
                    pbar.update(1)
                    pbar.set_postfix(env=env_id, paths=paths_generated)
                except Exception as e:
                    logger.error(f"Failed to process env {env_id}: {str(e)}")
                    results[env_id] = 0
    
    # Summary
    success_count = sum(1 for count in results.values() if count > 0)
    failure_count = num_envs - success_count
    total_paths = sum(results.values())
    
    logger.info(f"\nProcessing completed:")
    logger.info(f"Environments processed: {num_envs}")
    logger.info(f"Successfully processed environments: {success_count}")
    logger.info(f"Failed environments: {failure_count}")
    logger.info(f"Total paths generated: {total_paths}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help="The start index of the environment")
    parser.add_argument('--numEnv', type=int, required=True, help="The number of environments to collect data")
    parser.add_argument('--numPaths', type=int, required=True, help="Number of paths to collect per environment")
    parser.add_argument('--fileDir', type=str, required=True, help="Location to save collected data")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    logger.info(f"Starting parallel collection")
    logger.info(f"Generating {args.numEnv} environments starting from {args.start}")
    logger.info(f"Each environment will have {args.numPaths} paths")
    logger.info(f"Output directory: {args.fileDir}")
    logger.info(f"Using {args.workers} parallel workers")
    
    start_time = time.time()
    results = parallel_map_collection(
        start=args.start,
        num_envs=args.numEnv,
        num_paths=args.numPaths,
        fileDir=args.fileDir,
        num_workers=args.workers
    )
    end_time = time.time()
    
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")