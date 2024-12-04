import numpy as np
import joblib
from importlib import resources

class load_model:
    
    def __init__(self, f_name):
        self.f_name = f_name
        
    def load(self):
        if self.f_name == 'inconel_benchmark':
            with resources.path('InverseBench.models.inconel_models', 'inconel_model.pkl') as ml_model_path:
                ml_model = joblib.load(str(ml_model_path))
            with resources.path('InverseBench.models.inconel_models', 'inconel_pca.pkl') as pca_path:
                pca = joblib.load(str(pca_path))
            model = (pca, ml_model)    

        elif self.f_name == 'airfoil_benchmark':
            with resources.path('InverseBench.models.airfoil_models', 'airfoil_model.pkl') as model_path:
                model = joblib.load(str(model_path))

        elif self.f_name == 'scalar_diffusion_benchmark':
            with resources.path('InverseBench.models.scalar_diffusion_models', 'scalar_diffusion_model.pkl') as model_path:
                model = joblib.load(str(model_path))

        elif self.f_name == 'friedman_multioutput_benchmark':
            model = None
        
        return model

class load_test_data:
    
    def __init__(self, f_name):
        self.f_name = f_name
        
    def load(self):
        if self.f_name == 'inconel_benchmark':
            
            with resources.path('InverseBench.test_data.inconel_data', 'input_test_data.npy') as input_data_path:
                input_test_data = np.load(str(input_data_path))
            with resources.path('InverseBench.test_data.inconel_data', 'output_test_data.npy') as output_data_path:                
                output_test_data = np.load(str(output_data_path))
                
            test_data = (input_test_data, output_test_data)

        elif self.f_name == 'airfoil_benchmark':
            
            with resources.path('InverseBench.test_data.airfoils_data', 'input_test_data.npy') as input_data_path:
                input_test_data = np.load(str(input_data_path))
            with resources.path('InverseBench.test_data.airfoils_data', 'output_test_data.npy') as output_data_path:                
                output_test_data = np.load(str(output_data_path))
                
            test_data = (input_test_data, output_test_data)
            
        elif self.f_name == 'scalar_diffusion_benchmark':
            
            with resources.path('InverseBench.test_data.scalar_diffusion_data', 'input_test_data.npy') as input_data_path:
                input_test_data = np.load(str(input_data_path))
            with resources.path('InverseBench.test_data.scalar_diffusion_data', 'output_test_data.npy') as output_data_path:                
                output_test_data = np.load(str(output_data_path))
                
            test_data = (input_test_data, output_test_data)
            
        elif self.f_name == 'friedman_multioutput_benchmark':
            
            with resources.path('InverseBench.test_data.friedman_data', 'input_test_data.npy') as input_data_path:
                input_test_data = np.load(str(input_data_path))
            with resources.path('InverseBench.test_data.friedman_data', 'output_test_data.npy') as output_data_path:                
                output_test_data = np.load(str(output_data_path))
                
            test_data = (input_test_data, output_test_data)
        
        return test_data


class benchmark_functions:
    
    def __init__(self, f_name, model=None):
        
        self.f_name = f_name
        self.model = model

    def get_bounds(self):
        
        if self.f_name == 'inconel_benchmark':
            
            lb = np.array([0.3, 10, 15])
            ub = np.array([1.2, 700, 28])
        
        if self.f_name == 'airfoil_benchmark':
            
            lb = np.array([4e6, 0, 0.02, 0.2, 0.06])
            ub = np.array([6e6, 8, 0.09, 0.7, 0.15])
        
        if self.f_name == 'scalar_diffusion_benchmark':
            
            lb = np.zeros(20)
            ub = np.ones(20)*30
              
        if self.f_name == 'friedman_multioutput_benchmark':
            
            lb = np.zeros(5)
            ub = np.ones(5)
            
        return lb, ub
    
            
    def inconel_benchmark(self, x):
        
        pca, ml_model = self.model
        
        f = pca.inverse_transform(ml_model.predict(x))
        
        return  f

    def airfoil_benchmark(self, x):
        
        ml_model = self.model
        
        f = ml_model.predict(x)
        
        return  f
    
    def scalar_diffusion_benchmark(self, x):
        
        ml_model = self.model
        
        f = ml_model.predict(x)
        
        return f
    
    def friedman_multioutput_benchmark(self, x):   
        x = np.array(x)
        f = np.zeros((len(x), 10))
        for i in range(10):
            
            a = np.random.uniform(5, 8)  
            b = np.random.uniform(25, 28)  
            c = np.random.uniform(0.4, 0.5)  
            d = np.random.uniform(2, 3)  

            f[:, i] = (a * np.sin(np.pi * x[:, 0] * x[:, 1]) +
                       b * (x[:, 2] - 0.5)**2 +
                       c * x[:, 3] +
                       d * x[:, 4])
        
        return f
    
    def evaluate(self, x):

        if self.f_name == 'inconel_benchmark':
            responses = self.inconel_benchmark(x)
        
        if self.f_name == 'airfoil_benchmark':
            responses = self.airfoil_benchmark(x)
        
        if self.f_name == 'scalar_diffusion_benchmark':
            responses = self.scalar_diffusion_benchmark(x)
            
        if self.f_name == 'friedman_multioutput_benchmark':
            responses = self.friedman_multioutput_benchmark(x)
            
        return responses



