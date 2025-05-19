# InverseBench - A blackbox inverse design benchmark suite
Inverse design benchmark suite that contains black box inverse problems from science and engineering.

The fastest way to install the module is just to use pip in the main directory (will be updated):


```bash
pip install .
```


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

It is neccessary to unpack the .zip in the InverseBench main directory. The data are already available on the github repository.


**examples.py** contains examples on how to access all of the four benchmark functions.
----------------------------------------------


**Airfoil benchmark example**


```python

import numpy as np
from InverseBench.benchmarks import load_model, benchmark_functions #import required modules for the benchmarks

name = 'airfoil_benchmark' #define the name of the benchmark function
model = load_model(name).load() #load the forward model
f = benchmark_functions(name, model) #load the benchmark function
lb, ub = f.get_bounds() #get lower and upper boundaries

def evaluation_function(x): #evaluation function definition, for every x it returns a response y
    value = f.evaluate(x)
    return value

cp = evaluation_function([np.random.uniform(lb, ub), np.random.uniform(lb, ub)])


```

## Copyright Notice

InverseBench: Inverse design benchmark suite that contains inverse problems from science and engineering (InverseBench) Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.


