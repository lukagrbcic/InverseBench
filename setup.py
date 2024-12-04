from setuptools import setup, find_packages

setup(
    name='InverseBench',  
    version='0.1.0',      
    author='Luka Grbcic',      
    author_email='lgrbcic@lbl.gov',      
    description='An inverse design problem benchmark suite',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/lukagrbcic/InverseBench/', 
    
    # Set packages with proper location
    packages=find_packages(where='src'),  # Search for packages in the src directory
    package_dir={'': 'src'},              # Tell setuptools that the root package is in src

    classifiers=[              
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
    
    python_requires='>=3.6',  # Specify the minimum Python version required

    include_package_data=True,  # Include files as specified in package_data
    package_data={             
        '': [                   # This applies to all packages
            'data/**/*',       # Include all files in the data directory and subdirectories
            'models/**/*',     # Include all files in the models directory and subdirectories
        ],
    },

    install_requires=[         
        'numpy>=1.24.3',
        'xgboost==2.0.3',
        'scikit-learn==1.2.2',
        'joblib==1.2.0',
    ],
)
