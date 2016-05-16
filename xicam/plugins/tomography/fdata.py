import yaml

with open('yaml/tomography/functions.yml','r') as stream:
    funcs=yaml.load(stream)

parameters = {}

for file in ('tomopyparams.yml',): #, 'astraparams.yml', 'mbirparams.yml'):
    with open('yaml/tomography/'+file ,'r') as stream:
        parameters.update(yaml.load(stream))

with open('yaml/tomography/funcnames.yml','r') as stream:
    names=yaml.load(stream)

for algorithm in funcs['Reconstruction']['TomoPy']:
    names[algorithm] = 'recon'

# for algorithm in funcs['Reconstruction']['Astra']:
#     names[algorithm] = 'recon'