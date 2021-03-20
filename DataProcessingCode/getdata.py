'''
Code for iterating through the hdf5 and exporting joint angles in a CSV file
hdf5 files taken from Stanford RoboTurk Project here:
https://roboturk.stanford.edu/dataset_real.html
Code adapted from parse_aligned_hdf5.py code provided by the Stanford RoboTurk Project
Public GitHub for the RoboTurk Project accessed here:
https://github.com/RoboTurk-Platform/roboturk_real_dataset
'''

import h5py
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import argparse


def demo_hdf5(f):
    data = f['data']
    EEFposition = np.array([0,0,0])
    JointPosition = np.array([0,0,0,0,0,0,0,0,0])

    for key in data.keys():

        user = data[key]
        demo_ids = user.keys()

        #print('--group user: {}'.format(key))

        for demo_id in demo_ids:
            print(demo_ids)
            demo_id = 'demo_57' #Here you can change which data set to open
            print(demo_id)
            #print('----group demo: {}'.format(demo_id))

            
            all_demo_attrs = dict(user[demo_id].attrs)
            for k in all_demo_attrs:
                attribute = user[demo_id].attrs[k]
                #print('----demo attribute: {} with value: {}'.format(k, attribute))
            robot_obs_keys = user[demo_id]['robot_observation'].keys()
        
            #print('-------group robot observation')
            for k in robot_obs_keys:
                obs_shape = user[demo_id]['robot_observation'][k].shape
                #print('--------robot observation dataset {} with {} shape'.format(k, obs_shape))
                #Radians of joint arms are rad of 0,3,6,9,12,15,18,21,24
                #7 DOF robot
                #print(obs_shape)
            
            for i in range(obs_shape[0]): #0 to num data points 

                JointInputAll = user[demo_id]['robot_observation']['joint_states_arm'][i];
                JointInputVec = np.array([]) #Initialize it
                for j in range(len(JointInputAll)): #Make a vector with position of each joint arm
                    if j % 3 == 0:
                        JointInputVec = np.append(JointInputVec, JointInputAll[j])
                JointPosition = np.vstack((JointPosition, JointInputVec))

            #We want to return a vector with all of the data points
            JointPosition = np.delete(JointPosition, 0, 0) #Delete initialization of the first row of zeros
            return EEFposition, JointPosition

            user_keys = user[demo_id]['user_control'].keys()

            #print('-------group user control')
            for k in user_keys:
                user_shape = user[demo_id]['user_control'][k].shape
                #print('------user control dataset {} with {} shape'.format(k, user_shape))
            break
        break

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hdf5_input', required=True, help='HDF5 file to parse')

    results = parser.parse_args()

    f = h5py.File(results.hdf5_input, 'r')

    EEFposition, JointPosition = demo_hdf5(f)
    np.savetxt("JointPositionLaundry3.csv",JointPosition, delimiter=",")

if __name__ == "__main__":
    main()
