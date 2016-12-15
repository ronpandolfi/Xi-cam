import pkgutil
import importlib

from slacxop import Operation
from slacxop import Batch

# TODO: Load a config file indicating which Operations are enabled. 

def load_ops_from_path(path_,pkg,cat_root='MISC'):
    ops = []
    cats = []
    # pkgutil.iter_modules returns module_loader, module_name, ispkg forall modules in path
    mods = pkgutil.iter_modules(path_)
    mods = [mod for mod in mods if mod[1] not in ['__init__','slacxop','slacxopman','optools']]
    for modloader, modname, ispkg in mods:
        print 'exploring module {}'.format(modname)
        mod = importlib.import_module('.'+modname,pkg)
        # if it is a package and not DMZ, recurse into that package
        if ispkg and not modname in ['DMZ','TRASH']:
            pkg_path = [path_[0]+'/'+modname]
            if cat_root == 'MISC':
                pkg_cat_root = modname 
            else:
                pkg_cat_root = cat_root+'.'+modname
            pkg_ops, pkg_cats = load_ops_from_path(pkg_path,pkg+'.'+modname,pkg_cat_root)
            pkg_ops = [op for op in pkg_ops if not op in ops]
            pkg_cats = [cat for cat in pkg_cats if not cat in cats]
            ops = ops + pkg_ops
            cats = cats + pkg_cats
        else:
            print 'load ops from module {}'.format(modname)
            new_ops, new_cats = load_ops_from_module(mod,cat_root)
            new_ops = [op for op in new_ops if not op in ops]
            new_cats = [cat for cat in new_cats if not cat in cats]
            print 'found new cats {}'.format(new_cats)
            print 'found new ops {}'.format(new_ops)
            ops = ops + new_ops
            cats = cats + new_cats
    return ops, cats

def load_ops_from_module(mod,cat_root):
    # iterate through the module's __dict__, find Operations 
    ops = []
    cats = []
    for nm, itm in mod.__dict__.items():
        try:
            # is it a class?
            if isinstance(itm,type):
                # is it a non-abstract subclass of Operation?
                if issubclass(itm,Operation) and not nm in ['Operation','Realtime','Batch']:
                    op = getattr(mod,nm)
                    #op_cats = op().categories
                    #if cat_root is not None and not cat_root in op_cats:
                    #    op_cats = [cat_root] + op_cats
                    op_cats = [cat_root]
                    ops.append( (op_cats,op) )
                    for cat in op_cats:
                        if not cat in cats:
                            cats.append(cat)
        except ImportError as ex:
            print '[{}] had trouble dealing with {}: {}'.format(__name__,name,item)
            print 'Error text: {}'.format(ex.message)
            pass 
    return ops, cats

op_list, cat_list = load_ops_from_path(__path__,__name__)

print 'found cats: {}'.format(cat_list)
print 'found ops: {}'.format(op_list)

