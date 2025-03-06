from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import sys


UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.parent = None
        self.children = []
        
        pass

    def set_parent(self, parent):
        self.parent = parent
        pass

    def add_child(self, child):
        self.children.append(child)


def sample_conf():
    joint_ranges = [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi), (-np.pi, np.pi)]

    if np.random.rand() < 0.05:
        return goal_conf, True

    random_conf = tuple(np.random.uniform(low, high) for low, high in joint_ranges)
    
    return random_conf, False
   
def find_nearest(rand_node, node_list):
    return min(node_list, key=lambda node: np.linalg.norm(np.array(node.conf) - np.array(rand_node.conf)))

        
def steer_to(rand_node, nearest_node):
    step_size = 0.05
    direction = np.array(rand_node.conf) - np.array(nearest_node.conf)
    norm = np.linalg.norm(direction)

    if norm < step_size:
        return rand_node if not collision_fn(rand_node.conf) else None

    step = (direction / norm) * step_size
    new_conf = np.array(nearest_node.conf)

    while np.linalg.norm(new_conf - np.array(rand_node.conf)) > step_size:
        new_conf += step
        if collision_fn(tuple(new_conf)):
            return None  

    return RRT_Node(tuple(new_conf))

def backtrack_path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append(node.conf)
        node = node.parent
    return path[::-1] 

def RRT():
    max_iterations = 1000
    nodes = [RRT_Node(start_conf)]  

    for _ in range(max_iterations):
        rand_conf, is_goal = sample_conf()
        rand_node = RRT_Node(rand_conf)

        nearest_node = find_nearest(rand_node, nodes)
        new_node = steer_to(rand_node, nearest_node)

        if new_node:
            new_node.set_parent(nearest_node)
            nearest_node.add_child(new_node)
            nodes.append(new_node)

            if is_goal or np.linalg.norm(np.array(new_node.conf) - np.array(goal_conf)) < 0.1:
                print("Goal reached!")
                print(backtrack_path(new_node))
                return backtrack_path(new_node)

    print("No valid path found")
    return None



###############################################################################
#your implementation ends here

if __name__ == "__main__":
    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=238.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    
		# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    path_conf = RRT()

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.5)
