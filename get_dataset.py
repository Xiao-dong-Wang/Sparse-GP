import autograd.numpy as np
import os
import string

noise_var = 0.03
def init_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    x = np.random.uniform(bounds[0,0], bounds[0,1], (num, dim))
    dataset = {}
    dataset['train_x'] = x
    dataset['train_y'] = funct(x)
    return dataset

def get_test(funct, num, bounds):
    dim = bounds.shape[0]
    dataset = {}
    dataset['test_x'] = np.linspace(0.9*bounds[0,0], 1.1*bounds[0,1], num)[:,None]
    dataset['test_y'] = funct(dataset['test_x'])
    return dataset

def test1(x):
    err = (np.random.randn(x.shape[0])*noise_var).reshape(-1,1)
    ret = np.sin(x + err)+err
    return ret.reshape(-1,1)

def test2(x):
    err = (np.random.randn(x.shape[0])*noise_var).reshape(-1,1)
    ret = (x+err)**2 * np.sin(5.0*np.pi*(x+err))+err
    return ret.reshape(-1,1)



def get_funct(funct):
    if funct == 'test1':
        return test1
    elif funct == 'test2':
        return test2
    else:
        return test1
    


