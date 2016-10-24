from PySide import QtCore
import dask.threaded

from ..treemodel import TreeModel
from ..treeitem import TreeItem
from ..operations import optools
from .slacxwf import Workflow

# TODO: See note on remove_op()

class WfManager(TreeModel):
    """
    Class for managing a Workflow(Operation) built from slacx Operations.
    """

    def __init__(self,**kwargs):
        self._n_loaded = 0 
        #TODO: build a saved tree from kwargs
        #if 'wf_loader' in kwargs:
        #    with f as open(wf_loader,'r'): 
        #        self.load_from_file(f)
        self._wf_dict = {}       # this will be a dict for a dask.threaded graph 
        super(WfManager,self).__init__()

    def add_op(self,new_op,tag):
        """Add an Operation to the tree as a new top-level TreeItem."""
        # Count top-level rows by passing parent=QModelIndex()
        ins_row = self.rowCount(QtCore.QModelIndex())
        # Make a new TreeItem, column 0, invalid parent 
        new_treeitem = TreeItem(ins_row,0,QtCore.QModelIndex())
        new_treeitem.data.append(new_op)
        new_treeitem.set_tag( tag )
        new_treeitem.long_tag = new_op.__doc__
        self.beginInsertRows(
        QtCore.QModelIndex(),ins_row,ins_row)
        # Insertion occurs between notification methods
        self.root_items.insert(ins_row,new_treeitem)
        self.endInsertRows()
        # Render Operation inputs and outputs as children
        indx = self.index(ins_row,0,QtCore.QModelIndex())
        self.io_subtree(new_op,indx)
        self._n_loaded += 1

    def update_op(self,indx,new_op):
        """Replace Operation at indx with new_op"""
        # Get the treeitem for indx
        item = self.get_item(indx)
        # Put the data in the treeitem
        item.data[0] = new_op
        item.long_tag = new_op.__doc__
        # Wipe out the children
        #for child in item.children:
        #    del child
        # Update the op subtree
        self.build_io_subtrees(new_op,indx)
        # TODO: update gui arg frames

    def io_subtree(self,op,parent):
        """Add inputs and outputs subtrees as children of an Operation TreeItem"""
        # Get a reference to the parent item
        p_item = parent.internalPointer()
        # TreeItems as placeholders for inputs, outputs lists
        inputs_treeitem = TreeItem(0,0,parent)
        inputs_treeitem.set_tag('Inputs')
        outputs_treeitem = TreeItem(1,0,parent)
        outputs_treeitem.set_tag('Outputs')
        # Insert the new TreeItems
        self.beginInsertRows(parent,0,1)
        p_item.children.insert(0,inputs_treeitem)
        p_item.children.insert(1,outputs_treeitem)
        self.endInsertRows()
        # Populate the new TreeItems with op.inputs and op.outputs
        self.build_io_subtrees(op,parent)

    def build_io_subtrees(self,op,parent):
        # Get a reference to the parent item
        p_item = parent.internalPointer()
        # Get references to the inputs and outputs subtrees
        inputs_treeitem = p_item.children[0]
        outputs_treeitem = p_item.children[1]
        # Get the QModelIndexes of the subtrees 
        inputs_indx = self.index(0,0,parent)
        outputs_indx = self.index(1,0,parent)
        # Eliminate their children
        nc_i = inputs_treeitem.n_children()
        nc_o = outputs_treeitem.n_children()
        self.removeRows(0,nc_i,inputs_indx)
        self.removeRows(0,nc_o,outputs_indx)
        # Populate new inputs and outputs
        n_inputs = len(op.inputs)
        input_items = op.inputs.items()
        n_outputs = len(op.outputs)
        output_items = op.outputs.items()
        self.beginInsertRows(inputs_indx,0,n_inputs-1)
        for i in range(n_inputs):
            name,val = input_items[i]
            inp_treeitem = TreeItem(i,0,inputs_indx)
            inp_treeitem.set_tag(name)
            # generate long tag from optools.parameter_doc(name,val,doc)
            inp_treeitem.long_tag = optools.parameter_doc(name,val,op.input_doc[name])
            inp_treeitem.data.append(val)
            inputs_treeitem.children.insert(i,inp_treeitem)
        self.endInsertRows()
        self.beginInsertRows(outputs_indx,0,n_outputs-1)
        for i in range(n_outputs):
            name,val = output_items[i]
            out_treeitem = TreeItem(i,0,outputs_indx)
            out_treeitem.set_tag(name)
            out_treeitem.long_tag = optools.parameter_doc(name,val,op.output_doc[name])
            out_treeitem.data.append(val)
            outputs_treeitem.children.insert(i,out_treeitem)
        self.endInsertRows()

    def remove_op(self,rm_indx):
        """Remove an Operation from the workflow tree"""
        rm_row = rm_indx.row()
        self.beginRemoveRows(
        QtCore.QModelIndex(),rm_row,rm_row)
        # Removal occurs between notification methods
        item_removed = self.root_items.pop(rm_row)
        self.endRemoveRows()
        # TODO: update any Operations that depended on the removed one

    # Overloaded headerData for WfManager 
    def headerData(self,section,orientation,data_role):
        if (data_role == QtCore.Qt.DisplayRole and section == 0):
            return "{} operation(s) loaded".format(self.rowCount(QtCore.QModelIndex()))
        elif (data_role == QtCore.Qt.DisplayRole and section == 1):
            return "info"
        else:
            return None

    def check_wf(self):
        """
        Check the dependencies of the workflow.
        Ensure that all loaded operations have inputs that make sense.
        Return a status code and message for each of the Operations.
        """
        pass

    def run_wf_serial(self):
        """
        Run the workflow by looping over Operations in self.root_items, 
        finding which ones are ready, and running them. 
        Repeat until no further Operations are ready.
        """
        ops_done = []
        to_run = self.ops_ready(ops_done)
        while len(to_run) > 0:
            print 'ops to run: {}'.format(to_run)
            for j in to_run:
                item = self.root_items[j]
                # Get QModelIndex of this item for later use in updating tree view
                indx = self.index(j,0,QtCore.QModelIndex())
                op = item.data[0]
                for name,val in op.inputs.items():
                    op.inputs[name] = self.locate_input(val)
                op.run()
                ops_done.append(j)
                self.update_op(indx,op)
            to_run = self.ops_ready(ops_done)

    def run_wf_graph(self):
        """
        Run the workflow by building a dask-compatible dict,
        then calling dask.threaded.get(dict, key)
        for each of the keys corresponding to operation outputs.
        TODO: optimize the execution of this by making the smallest
        possible number of calls to get().
        """
        # build the graph, get the list of outputs
        outputs_list = self.load_wf_dict()
        print 'workflow graph as dict:'
        print self._wf_dict
        

    def ops_ready(self,ops_done):
        """
        Give a list of indices in self.root_items 
        that contain Operations whose inputs are ready
        """
        indxs = []
        for j in range(len(self.root_items)):
            if not j in ops_done:
                item = self.root_items[j]
                op = item.data[0]
                inps = [self.locate_input(val) for name,val in op.inputs.items()]
                if not any([inp is None for inp in inps]):
                    indxs.append(j)
        return indxs

    def load_wf_dict(self):
        """
        Build a dask-compatible dictionary from the Operations in this tree
        """
        self._wf_dict = {}
        for j in range(len(self.root_items)):
            item = self.root_items[j]
            # Unpack the Operation
            op = item.data[0]
            keyindx = 0
            input_keys = [] 
            input_vals = ()
            for name,val in op.inputs.items():
                # Add a locate_input line for each input 
                dask_key = 'op'+str(j)+'inp'+str(keyindx)
                self._wf_dict[dask_key] = (self.locate_input, val)
                keyindx += 1
                input_keys.append(name)
                input_vals = input_vals + (dask_key)
            # Add a load_inputs line for op j
            dask_key = 'op'+str(j)+'_load'
            self._wf_dict[key] = (self.load_inputs, op, input_keys, input_vals) 
            # Add a run_op line for op j
            dask_key = 'op'+str(j)+'_run'
            self._wf_dict[key] = (self.run_op, op) 
            # Keep track of the keys corresponding to operation outputs.
            keyindx = 0
            output_keys = []
            for name,val in op.outputs.items():
                # Add a get_output line for each output
                dask_key = 'op'+str(j)+'out'+str()
                self._wf_dict[dask_key] = (self.get_output, val)
                keyindx += 1
                output_keys.append(name)

    @staticmethod
    def load_inputs(op,keys,vals):
        """
        By the time this is called, vals should be bound to actual input values by dask.
        Each dask key should have been assigned to a (self.locate_input, val)
        """
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i] 
            op.inputs[key] = val
        return op 

    @staticmethod
    def run_op(op):
        return op.run()

    def locate_input(self,inplocator):
        """
        Return the data pointed to by a given InputLocator object.
        If this is called on anything other than an InputLocator,
        it does nothing and returns the input argument.
        """
        if type(inplocator).__name__ == 'InputLocator':
            src = inplocator.src
            val = inplocator.val
            if src in optools.valid_sources:
                if (src == optools.fs_input
                   or src == optools.text_input): 
                    # val will be handled during operation loading- return it directly
                    return val 
                elif src == optools.op_input: 
                    # follow val as uri in workflow tree
                    path = val.split('.')
                    parent_indx = QtCore.QModelIndex()
                    for itemtag in path:
                        # get QModelIndex of item from itemtag
                        row = self.list_tags(parent_indx).index(itemtag)
                        qindx = self.index(row,0,parent_indx)
                        # get TreeItem from QModelIndex
                        item = self.get_item(qindx)
                        # set new parent in case the path continues...
                        parent_indx = qindx
                    # item.data[0] should now be the desired piece of data
                    return item.data[0]
            else: 
                msg = 'found input source {}, should be one of {}'.format(
                src, valid_sources)
                raise ValueError(msg)
        else:
            # if this method gets called on an input that is not an InputLocator,
            # do nothing.
            return inplocator

