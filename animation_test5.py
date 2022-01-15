import math
import copy
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from PIL import Image
import habitat_sim
from habitat.utils.visualizations import maps
import matplotlib.patches as patches
import random

from numpy.core.shape_base import block

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios': [1, 4]})

input_file_name =  'check1.npz'

global counter
counter = []

ax1.set_title("rgb camera view")
ax1.axis("off")

ax2.set_title("topdown map")
ax2.axis("off")

test_scene = "/home/himadri/Desktop/drive-download-20220109T052008Z-001/Willow.glb"

sim_settings = {
    "width": 128, 
    "height": 128,
    "scene": test_scene, 
    "default_agent": 0,
    "sensor_height": 1.5, 
    "color_sensor": True,  
    "seed": 1, 
    "enable_physics": False,  
    "meters_per_pixel": 0.025,
    "scan_length": 7.5,
    "scan_res": 0.5
}

#hopefully this is the fixed quaternion funciton
def quaternion_to_euler(rot):
    w = rot.w
    x = rot.x
    y = rot.z
    z = -rot.y
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)

    roll = math.atan2(sinr_cosp,cosr_cosp)

    sinp = 2 * (w*y - z*x)
    pitch = 0

    if sinp>=1:
        pitch = math.copysign(math.pi/2,sinp)
    elif sinp<=-1:
        pitch = math.copysign(math.pi/2,sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y +z*z)
    yaw = math.atan2(siny_cosp,cosy_cosp)

    return [roll,pitch,yaw]

def display_sample(rgb_obs):
    rgb_img = Image.fromarray(rgb_obs,mode="RGBA")
    return rgb_img

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "look_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "look_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=90.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def create_agent():
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([-0.6,0.0,0.0])
    agent.set_state(agent_state)

    return sim,agent

def create_map(sim,agent_state):
    top_down_map = maps.get_topdown_map(sim.pathfinder,
                                        height=agent_state.position[1],
                                        meters_per_pixel=sim_settings["meters_per_pixel"])
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

    return top_down_map,grid_dimensions

def scan_section(sim,settings):
    border1,border2 = sim.pathfinder.get_bounds()
    print(sim.pathfinder.get_bounds())
    choose_x = random.uniform(border1[0],border2[0])
    choose_z = random.uniform(border1[2],border2[2])

    sl = settings["scan_length"]

    # try creating 4 squares - one which considers the choose points as
    # lower right point of square
    # upper left point of square
    # lower left point of square
    # upper right point of square

    valid_positions = []

    print((choose_x,choose_z))

    # upper left corner
    trial_opp_corner_1 = (choose_x + sl,choose_z + sl)

    if (border1[0]<trial_opp_corner_1[0] and trial_opp_corner_1[0]<border2[0]) and (border1[2]<trial_opp_corner_1[1] and trial_opp_corner_1[1]<border2[2]):
        valid_positions.append(trial_opp_corner_1)
    
    #upper right corner
    trial_opp_corner_2 = (choose_x - sl,choose_z + sl)

    if (border1[0]<trial_opp_corner_2[0] and trial_opp_corner_2[0]<border2[0]) and (border1[2]<trial_opp_corner_2[1] and trial_opp_corner_2[1]<border2[2]):
        valid_positions.append(trial_opp_corner_2)

    #lower right corner 
    trial_opp_corner_3 = (choose_x - sl,choose_z - sl)

    if (border1[0]<trial_opp_corner_3[0] and trial_opp_corner_3[0]<border2[0]) and (border1[2]<trial_opp_corner_3[1] and trial_opp_corner_3[1]<border2[2]):
        valid_positions.append(trial_opp_corner_3)

    #lower left corner
    trial_opp_corner_4 = (choose_x + sl, choose_z - sl)

    if (border1[0]<trial_opp_corner_4[0] and trial_opp_corner_4[0]<border2[0]) and (border1[2]<trial_opp_corner_4[1] and trial_opp_corner_4[1]<border2[2]):
        valid_positions.append(trial_opp_corner_4)

    if len(valid_positions)==0:
        return True,None,None
    opp_corner = random.choice(valid_positions)
    sq_corner = (choose_x,choose_z)

    return False,sq_corner,opp_corner

def scan_coordinates(corner1,corner2,settings):
    coordinates = []
    scan_l = settings["scan_length"]
    scan_r = settings["scan_res"]
    anchor = (min(corner1[0],corner2[0]),min(corner1[1],corner2[1]))

    #setting the z coordinate to 0.0
    z = 0.0

    lim = int(scan_l/scan_r)

    for i in range(lim):
        for j in range(lim):
            coordinates.append(np.array([anchor[0]+i*scan_r,z,anchor[1]+j*scan_r]))

    return coordinates

def draw_rect(sim,corner1,corner2,settings):
    correction_term = settings["scan_res"]/2
    mpp = settings["meters_per_pixel"]
    border1,_ = sim.pathfinder.get_bounds()
    anchor = ((min(corner1[0],corner2[0]) - border1[0] - correction_term)/mpp,(min(corner1[1],corner2[1]) - border1[2] - correction_term)/mpp)

    rect = patches.Rectangle(anchor,7.5/mpp,7.5/mpp,linewidth=1,edgecolor='r', facecolor='none')

    return rect

def initialize_camera_output(sim,action):
    observations = sim.step(action)
    rgb = observations["color_sensor"]


    
    global im1

    im1 = ax1.imshow(display_sample(rgb))

def initialize_map(sim,top_down_map,grid_dimensions,agent_state):
    top_down_map_copy = copy.deepcopy(top_down_map)
    maps.draw_agent(top_down_map_copy,
                    maps.to_grid(agent_state.position[2],agent_state.position[0],grid_dimensions,pathfinder=sim.pathfinder),
                    quaternion_to_euler(agent_state.rotation)[2] + math.pi,
                    agent_radius_px=8)
    
    global im2

    im2 = ax2.imshow(top_down_map_copy)

def draw_map(sim,top_down_map,grid_dimensions,agent_state):
    top_down_map_copy = copy.deepcopy(top_down_map)
    maps.draw_agent(top_down_map_copy,
                    maps.to_grid(agent_state.position[2],agent_state.position[0],grid_dimensions,pathfinder=sim.pathfinder),
                    quaternion_to_euler(agent_state.rotation)[2] + math.pi,
                    agent_radius_px=8)

    im2.set_array(top_down_map_copy)

    plt.pause(0.0001)

def draw_camera_output(sim,action):

    global counter 

    observations = sim.step(action)
    rgb = observations["color_sensor"]

    im1.set_array(display_sample(rgb))
    # # plt.savefig("./images/" + str(counter) + ".png",rgb)

    # result = Image.fromarray((rgb).astype(np.uint8))
    # result.save("./images/" + str(counter) + ".png")

    # counter = counter + 1

    counter.append(rgb)

def initialize_viz():
    sim,agent = create_agent()
    top_down_map,grid_dimensions = create_map(sim,agent.get_state())

    initialize_map(sim,top_down_map,grid_dimensions,agent.get_state())
    initialize_camera_output(sim,"look_forward")

    return sim,agent,top_down_map,grid_dimensions

#test code below 

sim,agent,top_down_map,grid_dimensions = initialize_viz()

def update():
    scene_frames = []
    check = True
    while check:
        check,c1,c2 = scan_section(sim,sim_settings)

    coordinates = scan_coordinates(c1,c2,sim_settings)

    rect_patch = draw_rect(sim,c1,c2,sim_settings)
    ax2.add_patch(rect_patch)

    for coord in coordinates:
        agent_state = agent.get_state()
        agent_state.position = coord

        if sim.pathfinder.is_navigable(coord):
            agent.set_state(agent_state)

            draw_camera_output(sim,"look_forward")
            draw_map(sim,top_down_map,grid_dimensions,agent.get_state())

            draw_camera_output(sim,"look_left")
            draw_map(sim,top_down_map,grid_dimensions,agent.get_state())

            draw_camera_output(sim,"look_left")
            draw_map(sim,top_down_map,grid_dimensions,agent.get_state())

            draw_camera_output(sim,"look_left")
            draw_map(sim,top_down_map,grid_dimensions,agent.get_state())

            draw_camera_output(sim,"look_left")
            draw_map(sim,top_down_map,grid_dimensions,agent.get_state())

    rect_patch.remove()

for i in range(2):
    update()

np.savez(input_file_name,*counter)

plt.show(block=False)







        

        









