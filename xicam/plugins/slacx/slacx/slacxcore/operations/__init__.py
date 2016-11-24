import pkgutil
import importlib

from slacxop import Operation
from slacxop import Batch

#__path__ = pkgutil.extend_path(__path__, __name__)

# Begin by iterating through all modules.
# pkgutil.iter_modules returns module_loader, module_name, ispkg forall modules in path
mods = pkgutil.iter_modules(__path__)
mods = [mod for mod in mods if mod[1] not in ['__init__','slacxop','slacxopman','optools']]

op_list = []
cat_list = []
for modloader, modname, ispkg in mods:
    mod = importlib.import_module('.'+modname,__name__)
    # iterate through the module's __dict__, find Operations 
    for name, item in mod.__dict__.items():
        try:
            # is it a class?
            if isinstance(item,type):
                # is it a non-abstract subclass of Operation?
                if issubclass(item,Operation) and name not in ['Operation','Batch','Realtime']:
                    op = getattr(mod,name)
                    cats = op().categories
                    op_list.append( (cats,op) )
                    for cat in cats:
                        if not cat in cat_list:
                            cat_list.append(cat)
        except ImportError as ex:
            print '[{}] had trouble dealing with {}: {}'.format(__name__,name,item)
            print 'Error text: {}'.format(ex.message)
            pass 
            #raise

 

