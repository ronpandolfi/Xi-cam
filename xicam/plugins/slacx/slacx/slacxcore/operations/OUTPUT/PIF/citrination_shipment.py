from pypif import pif
from citrination_client import CitrinationClient 

from ...slacxop import Operation
from ... import optools

class CitrinationShipment(Operation):
    """
    Take a list of pypif.obj.System objects and ship them to Citrination.    
    Requires a file on the local filesystem containing a valid Citrination API key.
    Requires also the web address of the target Citrination instance.
    """

    def __init__(self):
        input_names = ['pif_stack','address','key_file']
        output_names = ['return_codes']
        super(CitrinationShipment,self).__init__(input_names,output_names)
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
        retcodes = []
        for p in pifs:
            try:
                pif.dump(p, open('tmp.json','w'))
                response = cl.create_data_set()
                dsid = response.json()['id']
                cl.upload_file('tmp.json',data_set_id = dsid)
                retcodes.append(int(dsid))
            except:
                retcodes.append(-1)
        self.outputs['return_codes'] = retcodes


