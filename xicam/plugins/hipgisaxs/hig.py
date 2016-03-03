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


def dict2str(d, depth=0):
    content = ''
    keyseparator = ' '
    tabseparator = ''
    for value in d.values():
        if type(value) is dict:
            keyseparator = '\n'
            tabseparator = u'\t'

    for key in d:
        keytext = key
        if key.startswith('param'): keytext = 'param'

        if type(d[key]) is dict:
            dictlineseparator = ''
            for value in d[key].values():
                if type(value) is dict:
                    dictlineseparator = '\n'
            content += u'{0}{1} = {{{2}'.format(u'\t' * depth, str(key), dictlineseparator)
            content += dict2str(d[key], depth + 1)
            content = content[:-1] + u'{0}{1}}},\n'.format(tabseparator * depth, dictlineseparator)
        elif type(d[key]) is tuple:
            content += u'{0}{1} = {2},{3}'.format(tabseparator * depth, keytext,
                                                  '[ ' + ' '.join(map(str, d[key])) + ' ]', keyseparator)
        elif type(d[key]) is list:
            content += u'{0}{1} = {2},{3}'.format(tabseparator * depth, keytext,
                                                  '[ ' + ' '.join(map(str, d[key])) + ' ]', keyseparator)
        elif type(d[key]) is unicode:
            content += u'{0}{1} = "{2}",{3}'.format(tabseparator * depth, keytext, str(d[key]), keyseparator)
        elif type(d[key]) is str:
            content += u'{0}{1} = "{2}",{3}'.format(tabseparator * depth, keytext, str(d[key]), keyseparator)
        else:
            content += u'{0}{1} = {2},{3}'.format(tabseparator * depth, keytext, str(d[key]), keyseparator)

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
        with open(path, 'w') as f:
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