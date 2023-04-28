import numpy as np
from scipy import io
from utils import derivative, DynamicRecArray

def vic_data_parser(path='data/move/', dt=1/20):
    folders = ['no_gear','b_gear','s_gear','sb_gear']
    dD = DynamicRecArray(([('KXX', 'O'), ('KYY', 'O'), ('KXY', 'O'), ('KYX', 'O'), ('demo', 'O'), ('xR1', 'O')]))

    for folder in folders:
        demos = DynamicRecArray(([('DataP', 'O'), ('DataF', 'O')]))
        for n in range(1,5):
            demo_path = path + folder + '/' + str(n)
            print(demo_path)
            Fn = np.load(demo_path+'_ee_forces.npy', allow_pickle=True)
            Pn = np.load(demo_path+'_ee_poses.npy', allow_pickle=True)
            # making sure the numbers are converted to matlab double - to br removed later
            Pn = np.array(Pn, dtype=np.float64)
            print('Before slicing : forces: {}, poses: {}'.format(len(Fn), len(Pn)))
            ## remove the points where the robot is idle - in the begining
            tol = 0.5e-2
            prev_P = Pn[0]
            for idx, P in enumerate(Pn):
                if max(abs(P - prev_P)) >= tol:
                    Pn = Pn[idx:,:]
                    Fn = Fn[idx:,:]
                    print('Tolerance achieved')
                    break
            
            Vn = derivative(Pn, dt)   
            An = derivative(Vn, dt) 
            print('After slicing : forces: {}, poses: {}'.format(len(Fn), len(Pn)))
            posID = [0,2] # for a 2d system (matlab)
            DataP = np.concatenate((Pn[:,posID], Vn[:,posID], An[:,posID]), axis=1).T
            DataF = Fn[:,posID].T

            demos.append((DataP[:, 0:100], DataF[:, 0:100]))
        xR1 = DataP[posID,0, np.newaxis]
        dD.append((np.ones((1,100)),np.ones((1,100)),np.zeros((1,100)),np.zeros((1,100)), demos.data, np.array(xR1)))

    # adding to have atleast 10 demos i.e 10 springs
    # for i in range(6):
    #     dD.append((np.ones((1,100)),np.ones((1,100)),np.ones((1,100)),np.ones((1,100)), demos.data, np.array(xR1)))

    mdict = {'dD': dD.data, 'nbData': 100.0}
    return mdict

def collaborative_planar(path='data/gear/sb_gear/', dt=1/20):
    demos = DynamicRecArray(([('p', 'O'), ('DataP', 'O'), ('DataF', 'O')]))
    for n in range(1,11):
        demo_path = path + str(n)
        print(demo_path)
        Fn = np.load(demo_path+'_ee_forces.npy', allow_pickle=True)
        Pn = np.load(demo_path+'_ee_poses.npy', allow_pickle=True)
        # making sure the numbers are converted to matlab double - to br removed later
        Pn = np.array(Pn, dtype=np.float64)
        print('Before : forces: {}, poses: {}'.format(len(Fn), len(Pn)))
        ## remove the points where the robot is idle - in the begining
        tol = 0.5e-2
        prev_P = Pn[0]
        for idx, P in enumerate(Pn):
            if max(abs(P - prev_P)) >= tol:
                Pn = Pn[idx:,:]
                Fn = Fn[idx:,:]
                #print('Tolerance achieved')
                break
        print('After : forces: {}, poses: {}'.format(len(Fn), len(Pn)))
        Vn = derivative(Pn, dt)   
        An = derivative(Vn, dt) 

        DataP = np.concatenate((Pn, Vn, An), axis=1).T
        DataF = Fn.T

        p = DynamicRecArray(([('b', 'O'), ('A', 'O'), ('invA', 'O')]))
        # initial frame position ann orientation
        xR1 = DataP[:3,0, np.newaxis]
        xR1 = np.insert(xR1, 0, 0., axis=0) # for time variable
        oR1 = np.identity(4)
        p.append((xR1, oR1, oR1))
        # target frame
        xRn = DataP[:3,-1, np.newaxis]
        xRn = np.insert(xRn, 0, 0., axis=0) # for time variable
        oRn = np.identity(4)
        p.append((xRn, oRn, oRn))
        # random frame (for now mean on start and end frame)
        xRrand = (xR1 + xRn)/2
        oRrand = np.identity(4)
        p.append((xRrand, oRrand, oRrand))
        demos.append((p.data, DataP[:, -300:], DataF[:, -300:])) # goal state is priority.. many samples around it
        
    mdict = {'s': demos.data}
    return mdict
    

