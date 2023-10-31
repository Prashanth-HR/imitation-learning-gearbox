import matlab.engine

def pHRI_toyExample_PlanarManip01(eng : matlab.engine, position, dt):
    path = '/home/prashanth/Thesis/Resources/Force/Codes/CollaborativeTransportation2D'
    s = eng.genpath(path)
    eng.addpath(s, nargout= 0)
    #currPos_T = eng.transpose(currPos)
    return eng.get_robot_dynamics(position, dt)
    

def toyExampleMSDstiffnessLearning(eng : matlab.engine):
    path = '/home/prashanth/Thesis/Resources/Force/Codes/ras18_toy_example'
    s = eng.genpath(path)
    eng.addpath(s, nargout= 0)
    
    return eng.toyExampleMSDstiffnessLearning(nargout= 1)