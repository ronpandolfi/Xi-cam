def load(path):
    with open(path) as f:
        content = f.read()

    return hig(d)


def parsetodict(content):
    """
    :type content: str
    :param content:
    :return:
    """
    if '=' in content:
        content.partition('=')


#takes string and writes in hig format
def dict2str(d, depth=0):
    content = ''
    for key in d:
        if type(d[key]) is dict:
            content += u'{0}{1} = {{\n'.format(u'\t' * depth, str(key))
            content += dict2str(d[key], depth + 1)
            content = content[:-1] + u'\n{0}}},\n'.format(u'\t' * depth)
        elif type(d[key]) is tuple:
            content += u'{0}{1} = {2},\n'.format(u'\t' * depth, key, '[ ' + ' '.join(map(str, d[key])) + ' ]')
        elif type(d[key]) is list:
            content += u'{0}{1} = {2},\n'.format(u'\t' * depth, key, '[ ' + ' '.join(map(str, d[key])) + ' ]')
        elif type(d[key]) is unicode:
            content += u'{0}{1} = "{2}",\n'.format(u'\t' * depth, key, str(d[key]))
        elif type(d[key]) is str:
            content += u'{0}{1} = "{2}",\n'.format(u'\t' * depth, key, str(d[key]))
        else:
            content += u'{0}{1} = {2},\n'.format(u'\t' * depth, key, str(d[key]))

    content = content[:-2] + u'\n'

    return content


class hig:
    def __init__(self, **d):
        self.__dict__.update(d)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, item):
        return None

    def __str__(self):
        return dict2str(self.__dict__)

    def write(self, path):
        with open(path,'w') as f:
            f.write(str(self))


if __name__ == '__main__':
    d = {'hipRMCInput': {'instrumentation': {'inputimage': 'data/mysphere.tif',
                                             'imagesize': [512, 512],
                                             'numtiles': 1,
                                             'loadingfactors': [0.111]},
                         'computation': {'runname': "test",
                                         'modelstartsize': [32, 32],
                                         'numstepsfactor': 1000,
                                         'scalefactor': 32}}}
    h = hig(**d)
    print h