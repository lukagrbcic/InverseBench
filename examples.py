import sys 
import numpy as np


sys.path.insert(0, 'src')

from benchmarks import *


name = 'inconel_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()
def evaluation_function(x):
    
    value = f.evaluate(x)
    
    return value

emissivity = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()

def evaluation_function(x):
    
    value = f.evaluate(x)

    return value

cp = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


name = 'scalar_diffusion_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()

def evaluation_function(x):
    value = f.evaluate(x)
    return value

probes = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


name = 'friedman_multioutput_benchmark'
f = benchmark_functions(name)
lb, ub = f.get_bounds()

def evaluation_function(x):
    value = f.evaluate(x)
    return value

y = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


