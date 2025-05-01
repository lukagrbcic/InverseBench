# InverseBlackBox
Inverse design benchmark suite that contains black box inverse problems from science and engineering.

Requirements are in the requirements are in the requirements.txt file. To install them just use:

```bash
pip install requrements.txt
```

----------------------------------------------
The benchmark functions are:

**Inconel photonic surface inverse design**

**Airfoil inverse design**

**Scalar diffusion inverse reconstruction**

**Friedman multioutput inverse problem**

----------------------------------------------
The models can be downloaded at: https://drive.google.com/file/d/1LPfFfEnR7UucaPESJIX_bjnRG5T0RNbM/view?usp=sharing

It is neccessary to unpack the .zip in the InverseBlackBox main directory. The data are already available on the github repository.


**examples.py** contains examples on how to access all of the four benchmark functions.
----------------------------------------------


**Airfoil benchmark example**


```python

import numpy as np
from InverseBlackBox.benchmarks import load_model, benchmark_functions #import required modules for the benchmarks

name = 'airfoil_benchmark' #define the name of the benchmark function
model = load_model(name).load() #load the forward model
f = benchmark_functions(name, model) #load the benchmark function
lb, ub = f.get_bounds() #get lower and upper boundaries

def evaluation_function(x): #evaluation function definition, for every x it returns a response y
    value = f.evaluate(x)
    return value

cp = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


```

# License
The licensing of this project is in progress by Berkeley Lab.
