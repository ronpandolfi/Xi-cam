import pickle

from pyqtgraph.parametertree import Parameter


class experiment(Parameter):
    def __init__(self, path=None):


        if path is None:
            config = [{'name': 'Name', 'type': 'str', 'value': 'New Experiment'},
                      {'name': 'Pixel Size X', 'type': 'int', 'value': 72e-6, 'siPrefix': True, 'suffix': 'm',
                       'step': 1e-6},
                      {'name': 'Pixel Size Y', 'type': 'int', 'value': 72e-6, 'siPrefix': True, 'suffix': 'm',
                       'step': 1e-6},
                      {'name': 'Center X', 'type': 'int', 'value': 823, 'suffix': ' px'},
                      {'name': 'Center Y', 'type': 'int', 'value': 1210, 'suffix': ' px'},
                      {'name': 'Detector Distance', 'type': 'float', 'value': 1, 'siPrefix': True, 'suffix': 'm',
                       'step': 1e-3},
                      {'name': 'Energy', 'type': 'float', 'value': 9000, 'siPrefix': True, 'suffix': 'eV'},
                      {'name': 'Notes', 'type': 'text', 'value': ''}]
            super(experiment, self).__init__(name='Experiment Properties', type='group', children=config)

        else:
            with open(path, 'r') as f:
                self.config = pickle.load(f)


    def save(self):
        path = '.emptyexperiment.json'
        with open(self.getvalue('Name') + '.json', 'w') as f:
            pickle.dump(self.saveState(), f)

    def getvalue(self, name):
        return self.child(name).value()

    def setValue(self, name, value):
        self.child(name).setValue(value)


    def edit(self):
        pass



        # edit the data
        # config['key3'] = 'value3'

        #write it back to the file
        #with open('config.json', 'w') as f:
        #    json.dump(config, f)







