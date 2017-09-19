import datetime

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

        self['title'] = title
        self['source_name'] = source_name
        self['data_contact'] = data_contact
        self['data_contributor'] = data_contributor
        self['links'] = links


class CITPayload(PublishablePayload):
    pass

class MDFPayload(PublishablePayload):
    def __init__(self):
        super(MDFPayload, self).__init__()

    @property
    def metapayload(self):
        dataset = {
            "mdf": {
                "title": self.title,
                "acl": [],                                              # TODO: what goes here?
                "source_name": self.source_name,
                "citation": self.citation,
                "links": self.links,
                "data_contact": dict(self.data_contact),
                "data_contributor": dict(self.data_contributor),
                "ingest_date": datetime.datetime.now().strftime('%b %d, %Y'),                           # TODO: shouldn't this be implied?
                "metadata_version": "1.1",
                "mdf_id": "1",
                "resource_type": "dataset"
            },
            "dc": {},
            "misc": {}
        }

        return dataset



class MCPayload(PublishablePayload):
    pass


class human(dict):
    def __init__(self, given_name, family_name, email='', institution=''):
        super(human, self).__init__()
        self.__dict__ = self

