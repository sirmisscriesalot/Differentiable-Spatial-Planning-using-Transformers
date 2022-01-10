import math
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from PIL import Image
import habitat_sim
from habitat.utils.visualizations import maps
import time
import copy

from numpy.lib.function_base import angle 

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios': [1, 4]})

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
}

def quaternion_to_euler(rot):
    w = rot.w
    x = rot.x
    y = rot.y
    z = rot.z
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
    top_down_map = maps.get_topdown_map(sim.pathfinder,height=agent_state.position[1],meters_per_pixel=sim_settings["meters_per_pixel"])
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

    return top_down_map,grid_dimensions

def initialize_map(sim,top_down_map,grid_dimensions,agent_state):
    top_down_map_copy = copy.deepcopy(top_down_map)
    maps.draw_agent(top_down_map_copy,maps.to_grid(agent_state.position[2],agent_state.position[0],grid_dimensions,pathfinder=sim.pathfinder),quaternion_to_euler(agent_state.rotation)[1],agent_radius_px=8)
    
    global im2

    im2 = ax2.imshow((top_down_map_copy),animated=True)

def initialize_camera_output(sim,action):
    observations = sim.step(action)
    rgb = observations["color_sensor"]
    
    global im1

    im1 = ax1.imshow(display_sample(rgb),animated=True)

def initialize_viz():
    sim,agent = create_agent()
    top_down_map,grid_dimensions = create_map(sim,agent.get_state())
    initialize_map(sim,top_down_map,grid_dimensions,agent.get_state())
    initialize_camera_output(sim,"look_forward")

    return sim,agent,top_down_map,grid_dimensions

sim,agent,top_down_map,grid_dimensions = initialize_viz()

#test code below

actuation_spec_amount = 1.5708 #pi/2
print(quaternion_to_euler(agent.get_state().rotation)[1])
angles = [quaternion_to_euler(agent.get_state().rotation)[1] + actuation_spec_amount*i for i in range(1,11)]

def test_update_agent(degree,sim,agent):
    
    observations = sim.step("look_left")
    agent_state = agent.get_state()
    rgb = observations["color_sensor"]

    agent_state.position = np.array([-0.6,degree/5,degree/5])
    agent.set_state(agent_state)

    im1.set_array(display_sample(rgb))
    top_down_map_copy = copy.deepcopy(top_down_map)
    maps.draw_agent(top_down_map_copy,maps.to_grid(agent_state.position[2],agent_state.position[0],grid_dimensions,pathfinder=sim.pathfinder),degree,agent_radius_px=8)

    im2.set_array(top_down_map_copy)

    return im1,im2

ani = animation.FuncAnimation(fig,test_update_agent,frames=angles,fargs=(sim,agent),interval=1000,blit=True)

plt.show()




