# InverseBench
Inverse design benchmark suite that containts problems from science and engineering.

The benchmark functions are:

**Inconel photonic surface inverse design**

**Airfoil inverse design**

**Scalar diffusion inverse reconstruction**

**Friedman multioutput inverse problem**


**examples.py** contains examples on how to access all of the four benchmark functions.
----------------------------------------------


**Friedman multiouput benchmark example**


```python

name = 'friedman_multioutput_benchmark' #define the name of the benchmark function
f = benchmark_functions(name) #load the benchmark function
lb, ub = f.get_bounds() #get lower and upper boundaries

def evaluation_function(x): #evaluation function definition, for every x it returns a response y
    value = f.evaluate(x)
    return value

y = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)]) #evaluation example


```

**Airfoil benchmark example**


```python

name = 'airfoil_benchmark' #define the name of the benchmark function
model = load_model(name).load_model() #load the forward model
f = benchmark_functions(name, model) #load the benchmark function
lb, ub = f.get_bounds() #get lower and upper boundaries

def evaluation_function(x): #evaluation function definition, for every x it returns a response y
    value = f.evaluate(x)
    return value

cp = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


```


