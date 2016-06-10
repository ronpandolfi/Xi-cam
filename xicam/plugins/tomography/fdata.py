import yaml

PARAM_TYPES = {'int': int, 'float': float}

# Load yaml with names of all available functions in pipeline
with open('yaml/tomography/functions.yml','r') as stream:
    funcs=yaml.load(stream)

# Load parameter data for available functions
parameter_files = ('tomopy_function_parameters.yml',
                   'aux_function_parameters.yml',
                   'dataexchange_function_parameters.yml')
                   #'astra_function_parameters.yml',
                   #'mbir_function_parameters.yml')
parameters = {}

for file in parameter_files:
    with open('yaml/tomography/'+file ,'r') as stream:
        parameters.update(yaml.load(stream))

# Load dictionary with pipeline names and function names
with open('yaml/tomography/function_names.yml','r') as stream:
    names=yaml.load(stream)

# Add reconstruction methods to function name dictionary, but include the package the method is in
for algorithm in funcs['Functions']['Reconstruction']['TomoPy']:
    names[algorithm] = ['recon', 'tomopy']

# for algorithm in funcs['Functions']['Reconstruction']['Astra']:
#     names[algorithm] = 'recon'

# Load dictionary with function parameters to be retrieved from metadatas
with open('yaml/tomography/bl832_function_defaults.yml','r') as stream:
    als832defaults = yaml.load(stream)

