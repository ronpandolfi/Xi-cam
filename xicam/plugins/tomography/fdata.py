import yaml

with open('yaml/tomography/functions.yml','r') as stream:
    funcs=yaml.load(stream)

parameters = {}

for file in ('tomopy_function_parameters.yml', 'aux_function_parameters.yml'): #, 'astra_function_parameters.yml', 'mbir_function_parameters.yml'):
    with open('yaml/tomography/'+file ,'r') as stream:
        parameters.update(yaml.load(stream))

with open('yaml/tomography/function_names.yml','r') as stream:
    names=yaml.load(stream)

for algorithm in funcs['Functions']['Reconstruction']['TomoPy']:
    names[algorithm] = ['recon', 'tomopy']

# for algorithm in funcs['Functions']['Reconstruction']['Astra']:
#     names[algorithm] = 'recon'

with open('yaml/tomography/bl832_function_defaults.yml','r') as stream:
    als832defaults = yaml.load(stream)