
import numpy as np
import os 
import quaternion
import numpy as np
import wandb

DATA_LOC = ''
def get_exp_params(task):
    dir_name = DATA_LOC + task
    if task == 'veggies':
        img_key = 'img2'
        language_instruction = 'lift the vegetable'
    
    elif task == 'rightcabinet':
        img_key = 'img1'
        language_instruction = 'open the cabinet'
    
    elif task == 'knife':
        img_key = 'img2'
        language_instruction = 'lift the knife'

    return dir_name, img_key, language_instruction


def process_reward(reward):
    reward_list = np.array([reward - reward[0]])[0]
    return np.array([10*rew if rew > 0 else 0 for rew in reward_list])

def quaternion_to_ypr(quat):
    q = quaternion.quaternion(quat[0], quat[1], quat[2], quat[3])
    return quaternion.as_euler_angles(q)

def get_data_for_task(task):
    dir_name, default_img_key, language_instruction = get_exp_params(task)
    _files = os.listdir(dir_name)

    for _file in _files:
        episode_path = dir_name + '/' + _file
        data = np.load(episode_path, allow_pickle=True)
    
        img_key = 'img' if 'img' in data else default_img_key 
        proc_data= {'image' : data[img_key][:-1]}

        ep_len = len(data[img_key]) - 1
        proc_data['state']  = data['state'][:-1]
        proc_data['reward'] = process_reward(data['reward'][:-1])
        proc_data['structured_action'] = data['action'][:-1]
       
        #converting quat to ypr 
        quat_list = data['state'][:,3:]
        ypr_list  = np.array([quaternion_to_ypr(quat) for quat in quat_list])

        # delta actions
        delta_pos = data['state'][1:, :3] - data['state'][:-1, :3]
        delta_ypr = ypr_list[1:] - ypr_list[:-1]
        delta_ypr = np.where(delta_ypr > 2*np.pi, delta_ypr - 2*np.pi, delta_ypr)
     
        #gripper open for first 3 actions
        gripper_action = np.concatenate([np.ones(3), np.zeros(ep_len - 3)]).reshape(-1,1)
        ep_termination = np.concatenate([np.zeros(ep_len - 1), [1]]).reshape(-1,1)
        action = np.concatenate([delta_pos, delta_ypr, gripper_action, ep_termination ], axis = -1)
        proc_data['action'] = action
       
        proc_data['language_instruction'] = language_instruction
        np.savez_compressed('data/train/' + _file, **proc_data)

for task in ['veggies', 'knife', 'rightcabinet']:
    get_data_for_task(task)