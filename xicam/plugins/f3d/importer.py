import yaml

# Load yaml with names of all available filters in pipeline
with open('yaml/f3d/filters.yml','r') as stream:
    filters=yaml.load(stream)