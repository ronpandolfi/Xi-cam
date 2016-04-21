import yaml


with open('xicam/plugins/tomography/yaml/funcs.yml','r') as stream:
    funcs=yaml.load(stream)

parameters = {}

for file in ('tomopyparams.yml',): #, 'astraparams.yml', 'mbirparams.yml'):
    with open('xicam/plugins/tomography/yaml/'+file ,'r') as stream:
        parameters.update(yaml.load(stream))

with open('xicam/plugins/tomography/yaml/funcnames.yml','r') as stream:
    names=yaml.load(stream)

for algorithm in funcs['Reconstruction']['TomoPy']:
    names[algorithm] = 'recon'

for algorithm in funcs['Reconstruction']['Astra']:
    names[algorithm] = 'recon'