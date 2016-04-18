import yaml


with open('xicam/plugins/stack/funcs.yml','r') as stream:
    funcs=yaml.load(stream)

parameters = {}

for file in ('tomopyparams.yml',): #, 'astraparams.yml', 'mbirparams.yml'):
    with open('xicam/plugins/stack/'+file ,'r') as stream:
        parameters.update(yaml.load(stream))
