import numpy as np
from InverseBench.benchmarks import load_model, benchmark_functions, load_test_data

name = 'inconel_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()
def evaluation_function(x):
    
    value = f.evaluate(x)
    
    return value

emissivity = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])

#to load test data use

test_data = load_test_data(name).load()



name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()

def evaluation_function(x):
    
    value = f.evaluate(x)

    return value

cp = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])





