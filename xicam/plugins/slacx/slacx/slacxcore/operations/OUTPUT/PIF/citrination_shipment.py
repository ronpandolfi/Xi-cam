import os

from pypif import pif
from citrination_client import CitrinationClient 

from ...slacxop import Operation
from ... import optools
from .... import slacxtools

class BuildCitrinationDataSet(Operation):
    """
    Take a list of pypif.obj.System objects and ship them to Citrination as a new data set.    
    Requires a file on the local filesystem containing a valid Citrination API key.
    Requires also the web address of the target Citrination instance.
    """

    def __init__(self):
        input_names = ['pif_stack','address','key_file']
        output_names = ['return_codes']
        super(BuildCitrinationDataSet,self).__init__(input_names,output_names)
        self.input_doc['pif_stack'] = 'A list of pypif.obj.System objects'
        self.input_doc['address'] = 'The http web address of a Citrination instance' 
        self.input_doc['key_file'] = 'Path to a file containing (only) a valid API key for the Citrination instance' 
        self.output_doc['return_codes'] = 'List of codes indicating success or failure for uploading each PIF.'
        self.categories = ['PACKAGING.PIF']
        self.input_src['pif_stack'] = optools.wf_input
        self.input_src['address'] = optools.user_input
        self.input_type['address'] = optools.str_type
        self.inputs['address'] = 'https://slac.citrination.com'
        self.input_src['key_file'] = optools.fs_input

    def run(self):
        a = self.inputs['address']
        kpath = self.inputs['key_file']
        pifs = self.inputs['pif_stack']
        f = open(kpath,'r')
        k = str(f.readline()).strip()
        f.close()
        # Start Citrination client
        cl = CitrinationClient(api_key = k, site = a)
        # Create the data set
        #response = cl.create_data_set()
        #dsid = response.json()['id']
        dsid = 'dummy'
        retcodes = []
        # TODO: Note that the entire data set can be one json,
        # of an array of pif records, and this will lead to a faster upload.
        for p in pifs:
            try:
                json_file = slacxtools.scratchdir+'/'+p.uid+'.json'
                pif.dump(p, open(json_file,'w'))
                print 'add DATA SET {} to tags'.format(dsid)
                p.tags.append('DATA SET {}'.format(dsid))
                print 'dump {} to data set {}'.format(json_file,dsid)
                #cl.upload_file(json_file,data_set_id = dsid)
                print 'NOT SHIPPING {} (this is a test)'.format(json_file)
                retcodes.append(dsid)
                # delete dataset json
                print 'deleting file {}'.format(json_file)
                os.remove(json_file) 
            except:
                retcodes.append(-1)
        self.outputs['return_codes'] = retcodes


