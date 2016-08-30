from pipeline.workflowfunctions import updateworkspace

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
    updates = {'A': 3}
    workspace = updateworkspace(workspace, updates)
    return workspace, updates


def setValue(varname, value, **workspace):
    print 'setting', varname, 'to', value
    updates = {varname: value}
    workspace = updateworkspace(workspace, updates)
    return workspace, updates
