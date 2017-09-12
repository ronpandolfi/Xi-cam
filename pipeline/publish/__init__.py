

class PublishablePayload(dict):
    def __init__(self, title, source_name, data_contact, data_contributor, links, **kwargs):
        """

        Parameters
        ----------
        title:              str
        source_name:        str
        data_contact:       human
        data_contributor:   human
        links:              dict
        """
        super(PublishablePayload, self).__init__()
        self.__dict__ = self

        optionalkeys = ['license',                 # str
                        'citation',                # list of str
                        'source_name',             # str
                        'data_contact',            # human
                        'data_contributor',        # human
                        'author',                  # human
                        'repository',              # str
                        'collection',              # str
                        'tags',                    # list of str
                        'description',             # str
                        'raw',                     # str
                        'links',                   # dict
                        'year',                    # int
                        'composition']             # str
        for prop in optionalkeys:
            self[prop] = kwargs.get(prop,None)


class CITPayload(PublishablePayload):
    pass

class MDFPayload(PublishablePayload):
    pass

class MCPayload(PublishablePayload):
    pass


class human(dict):
    def __init__(self, given_name, family_name, email='', institution=''):
        super(human, self).__init__()
        self.__dict__ = self

