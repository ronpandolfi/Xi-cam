functionManifest = """
Test functions:
    - displayName:  Set A to 3
      functionName: setAto3
      moduleName:   testfunctions
    - displayName:  Set Value
      functionName: setValue
      moduleName:   testfunctions
      parameters:
          - name:   Variable Name
            type:   str
            value:  B
          - name:   Value
            type:   int
            limits: [1,100000]


"""


def setAto3(**workspace):
    workspace['A'] = 3
    return workspace


def setValue(varname, value, **workspace):
    print 'setting', varname, 'to', value
    workspace[varname] = value
    return workspace
