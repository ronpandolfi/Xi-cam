import numpy as np
import pypif.obj as pifobj

from ...slacxop import Operation
from ... import optools

class PifNPSynthBatch(Operation):
    """
    Package a nanoparticle solution synthesis SAXS batch into a list of pypif.obj.ChemicalSystem objects.
    Tag these objects such that each one has context wrt its related objects.
    For example, a time series of saxs measurements that were all performed during one synthesis experiment
    should be packaged with a common tag so that data can be extracted from that series of objects,
    for example to build time-temperature or time-size plots.
    """

    def __init__(self):
        input_names = ['saxs_batch_output','system_name']
        output_names = ['pif_stack']
        super(PifNPSynthBatch,self).__init__(input_names,output_names)
        self.input_doc['saxs_batch_output'] = str('batch output (list of dicts)' 
        + ' from processing a series of saxs spectra from a nanoparticle synthesis experiment')
        self.input_doc['system_name'] = 'A globally unique string to be used as a tag for this batch' 
        self.output_doc['pif_stack'] = 'A stack pif objects, one for each dict in saxs_batch_output'
        self.categories = ['PACKAGING.PIF']
        self.input_src['saxs_batch_output'] = optools.wf_input
        self.input_src['system_name'] = optools.user_input
        self.input_type['system_name'] = optools.str_type
        
    def run(self):
        sbatch = self.inputs['saxs_batch_output']
        sysname = self.inputs['system_name']
        pif_stack = []
        time_stack = []
        # ChemicalSystem constructor (all args default None): 
        # __init__(self,uid,names,ids,source,quantity,chemical_formula,composition,
        # properties,preparation,sub_systems,references,contact,licenses,tags,kwargs)
        colloid_sys = pifobj.ChemicalSystem(sysname+'_pd_colloid',['Pd nanoparticle colloid'],None,None,None,'Pd') 
        acid_sys = pifobj.ChemicalSystem(sysname+'_oleic_acid',['oleic acid'],None,None,None,'C18H34O2') 
        amine_sys = pifobj.ChemicalSystem(sysname+'_oleylamine',['oleylamine'],None,None,None,'C18H35NH2') 
        TOP_sys = pifobj.ChemicalSystem(sysname+'_trioctylphosphine',['trioctylphosphine'],None,None,None,'P(C8H17)3')
        subsys = []
        subsys.append(colloid_sys)
        subsys.append(acid_sys)
        subsys.append(amine_sys)
        subsys.append(TOP_sys)
        t_all = np.array([d['TimeTempFromHeader_1.outputs.time'] for d in sbatch],dtype=float)
        t0 = np.min(t_all)
        for d in sbatch:
            temp_i = float(d['TimeTempFromHeader_1.outputs.temp'])
            t_i = float(d['TimeTempFromHeader_1.outputs.time'])-t0 
            chemsys = pifobj.ChemicalSystem()
            chemsys.names = ['Pd nanoparticles']
            chemsys.sub_systems = []
            for s in subsys:
                # TODO: Get quantity information built into workflow
                # so that it can be added to the sub_system 
                # TODO ALSO: When quantity info is available, use it to build chemsys.composition
                chemsys.sub_systems.append(s)
            q_I_saxs = d['WindowZip_1.outputs.x_y_window']
            saxs_props = self.saxs_to_pifprops(q_I_saxs,t_i,temp_i)
            chemsys.properties = saxs_props
            chemsys.tags = ['SSRL','SLAC','BEAMLINE 1-5','BEAM ENERGY XXXeV','DETECTOR XXX']
            time_stack.append(t_i)
            pif_stack.append(chemsys)
        # Sort and save the output
        stk_sorted = np.sort(np.array(zip(time_stack,pif_stack)),0)
        for i in range(np.shape(stk_sorted)[0]):
            stk_sorted[i,1].uid = sysname+'_{}'.format(i)
        self.outputs['pif_stack'] = list(stk_sorted[:,1])

    def saxs_to_pifprops(self,q_I,time_,temp_):
        props = []
        for i in range(np.shape(q_I)[0]):
            q = q_I[i,0]
            I = q_I[i,1]
            p = pifobj.Property()
            p.name = 'SAXS intensity'
            p.scalars = [ pifobj.Scalar(I) ]
            p.conditions = [] 
            p.conditions.append( pifobj.Value('scattering vector',[pifobj.Scalar(q)],None,None,'Angstrom^-1') )
            p.conditions.append( pifobj.Value('temperature',[pifobj.Scalar(temp_)],None,None,'degrees Celsius') )
            p.conditions.append( pifobj.Value('time',[pifobj.Scalar(time_)],None,None,'seconds') )
            p.units = 'counts'
            props.append(p)
        return props
        
    def make_piftemperature(self,t):
        v = pifobj.Value()
        v.name = 'temperature'
        tscl = pifobj.Scalar()
        tscl.value = str(t)
        v.scalars = [tscl]
        v.units = 'degrees Celsius'
        return v

#    def make_pifvector(self,v):
#        pifv = []
#        for n in v:
#            s = pifobj.Scalar()
#            s.value = str(n)
#            pifv.append(s)
#        return pifv
#
#    def pifscalar(self,scl,errlo,errhi):
#        s = pifobj.Scalar()
#        s.value = str(scl) 
#        s.minimum = str(scl) 
#        s.maximum = str(scl) 
#        s.inclusiveMinimum = True
#        s.inclusiveMaximum = True
#        s.uncertainty = '+{},-{}'.format(errlo,errhi)
#        s.approximate = True
#        return s


