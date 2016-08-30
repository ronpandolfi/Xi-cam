def updateworkspace(workspace, updates):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = workspace.copy()
    z.update(updates)
    return z
