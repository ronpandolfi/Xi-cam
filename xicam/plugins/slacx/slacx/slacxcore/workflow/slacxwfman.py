import os
import time
import traceback
from collections import OrderedDict
import copy

from PySide import QtCore
from functools import partial
import dask.threaded
import yaml

from ..treemodel import TreeModel
from ..treeitem import TreeItem
from ..operations import optools
from ..operations.slacxop import Operation, Batch, Realtime
from ..operations.optools import InputLocator, OutputContainer
from .. import slacxtools

# TODO: Write a slot that can be called when a set of Operations are finished, 
# where the slot will make all the right dataChanged() calls. 

class WfManager(TreeModel):
    """
    Class for managing a Workflow built from slacx Operations.
    """

    wfdone = QtCore.Signal()

    def __init__(self,**kwargs):
        super(WfManager,self).__init__()
        if 'logmethod' in kwargs:
            self.logmethod = kwargs['logmethod']
        else:
            self.logmethod = None
        if 'wfl' in kwargs:
            self.load_from_file( kwargs['wfl'] )
        if 'app' in kwargs:
            self.appref = kwargs['app']
        else:
            self.appref = None
        self._wf_dict = {}       
        # Flags to assist in thread control
        self._exec_ready = True 
        self._keep_going = True
        #self._n_threads = QtCore.QThread.idealThreadCount()
        self._n_threads = 1
        self._wf_threads = dict.fromkeys(range(self._n_threads)) 
        if self.logmethod:
            self.logmethod('Slacx workflow manager started, working with {} threads'.format(self._n_threads))

    def load_inputs(self,op):
        """
        Loads data for an Operation from that Operation's input_locators.
        It is expected that op.input_locator[name] will refer to an InputLocator.
        """
        for name,il in op.input_locator.items():
            if isinstance(il,InputLocator):
                src = il.src
                if not src == optools.batch_input:
                    il.data = self.locate_input(il,op)
                    op.inputs[name] = il.data
                else:
                    # Batch executor should have already set the batch inputs. 
                    il.data = op.inputs[name]
            else:
                msg = '[{}] Found broken Operation.input_locator for {}: {}'.format(
                __name__, name, il)
                raise ValueError(msg)
        #import pdb; pdb.set_trace()

    def locate_input(self,inplocator,op):
        """
        Return the data pointed to by a given InputLocator object.
        Takes the Operation that owns this inplocator as a second arg,
        so that if it is a Batch its input routes can be handled properly.
        """
        #if isinstance(inplocator,InputLocator):
        src = inplocator.src
        tp = inplocator.tp
        val = inplocator.val
        if src == optools.no_input: 
            return None
        elif src == optools.user_input: 
            return optools.cast_type_val(tp,val)
        elif src == optools.wf_input:
            if tp == optools.list_type:
                # val should be a list- get each item from wfman
                return [optools.parse_wf_input(self,v,op) for v in val]
            else:
                # get one item from wfman
                return optools.parse_wf_input(self,val,op)
        elif src == optools.fs_input:
            # Trust that Operations using fs input 
            # are taking care of parsing the file names in whatever form
            return val 
        elif src == optools.batch_input:
            return val 
        else: 
            msg = 'found input source {}, should be one of {}'.format(
            src, optools.valid_sources)
            raise ValueError(msg)

    def load_from_file(self,opman,wfl):
        """
        Load things in to the Workflow from an OpManager and a YAML .wfl file 
        """
        # TODO: Migrate to own module
        for row in range(len(self.root_items)):
            idx = self.index(row,0,QtCore.QModelIndex())
            self.remove_op(idx)
        f = open(wfl, "r")
        dct = yaml.load(f)
        f.close()
        for uri, opdict in dct.items():
            opname = opdict['type']
            op = opman.get_op_byname(opname)()
            # TODO: Eventually remove this and deprecate the hard-coded 'Inputs' key
            if 'Inputs' in opdict.keys():
                ilspec = opdict['Inputs']
            else:
                ilspec = opdict[optools.inputs_tag]
            for name, srctypeval in ilspec.items():
                src = srctypeval['src']
                tp = srctypeval['type']
                val = srctypeval['val'] 
                # TODO: Eventually remove this and deprecate the hard-coded 'Inputs' key
                if src == optools.wf_input and isinstance(val,str):
                    val = val.replace('Inputs',optools.inputs_tag)
                il = optools.InputLocator(src,tp,val)
                op.input_locator[name] = il
            self.add_op(uri,op)
        #print dct
        
    def save_to_file(self,filename):
        """
        Save the current image of the Workflow as a YAML 
        """
        # TODO: Migrate to own module
        if not os.path.splitext(filename)[1] == '.wfl':
            filename = filename+'.wfl'
        #filename = slacxtools.rootdir+'/'+'test.wfl'
        wf_dict = OrderedDict() 
        #wf_dict = {} 
        for row in range(len(self.root_items)):
            item = self.root_items[row]
            idx = self.index(row,0,QtCore.QModelIndex())
            uri = self.build_uri(idx)
            wf_dict[str(uri)] = self.op_dict(item)
        if self.logmethod:
            self.logmethod( 'dumping current workflow image to {}'.format(filename) )
        f = open(filename, "w")
        #yaml.dump(wf_dict, f, encoding='utf-8')
        yaml.dump(wf_dict, f)
        f.close()
    def op_dict(self,op_item):
        dct = OrderedDict() 
        op = op_item.data
        dct['type'] = type(op).__name__ 
        dct[optools.inputs_tag] = self.inputs_dict(op)
        return dct
    def inputs_dict(self,op):
        dct = OrderedDict() 
        for name in op.inputs.keys():
            il = op.input_locator[name]
            dct[name] = {'src':il.src,'type':il.tp,'val':il.val}
        return dct
    #def outputs_dict(self,op):
    #    #dct = {}
    #    dct = OrderedDict() 
    #    for name in op.outputs.keys():
    #        dct[name] = str(op.outputs[name])
    #    return dct

    def add_op(self,uri,new_op):
        """Add an Operation to the tree as a new top-level TreeItem."""
        # Count top-level rows by passing parent=QModelIndex()
        ins_row = self.rowCount(QtCore.QModelIndex())
        # Make a new TreeItem, column 0, invalid parent 
        new_treeitem = TreeItem(ins_row,0,QtCore.QModelIndex())
        new_treeitem.data = new_op
        new_treeitem.set_tag( uri )
        #new_treeitem.set_long_tag( new_op.__doc__ )
        self.beginInsertRows(
        QtCore.QModelIndex(),ins_row,ins_row)
        # Insertion occurs between notification methods
        self.root_items.insert(ins_row,new_treeitem)
        self.endInsertRows()
        # Render Operation inputs and outputs as children
        idx = self.index(ins_row,0,QtCore.QModelIndex())
        self.build_next(new_op,idx) 
        #self.io_subtree(new_op,indx)
        #self._n_loaded += 1

    def remove_op(self,rm_indx):
        """Remove an Operation from the workflow tree"""
        rm_row = rm_indx.row()
        self.beginRemoveRows(QtCore.QModelIndex(),rm_row,rm_row)
        # Removal occurs between notification methods
        item_removed = self.root_items.pop(rm_row)
        self.endRemoveRows()
        # Update any Operations that depended on the removed one
        self.update_io_deps(item_removed.tag())

    def update_op(self,uri,new_op):
        """
        Update Operation in treeitem indicated by uri.
        First, clean up any dependencies that are broken by the change.
        Then, inform any attached QTreeView(s) that the Operation has been changed.
        It is expected that new_op is a reference to the Operation stored at uri. 
        """
        #print '--- update op at '+uri
        #print 'new op outputs: {}'.format(new_op.outputs)
        item, idx = self.get_from_uri(uri)
        # Take care of any IO dependencies: 
        self.update_io_deps(uri,new_op)
        # Call out the dataChanged
        self.tree_dataChanged(idx)

    def tree_dataChanged(self,idx):
        itm = idx.internalPointer()
        self.dataChanged.emit(idx,idx)
        # Build any new children that resulted from the dataChanged at idx.
        self.update_children(idx)
        for c_row in range(itm.n_children()):
            c_idx = self.index(c_row,0,idx)
            self.tree_dataChanged(c_idx)
    
    def update_children(self,idx):
        """
        Check the children of the item at idx and compare them against the item's data.
        If new children are found in the data, render them as new children.
        """
        #print 'UPDATE CHILDREN for {}'.format(idx.internalPointer().data)
        #print 'n_children: {}'.format(idx.internalPointer().n_children())
        itm = idx.internalPointer()
        x = itm.data
        # If it is an OutputContainer, unpack it
        if isinstance(x,OutputContainer):
            x = x.data
        if isinstance(x,Operation):
            # Operations should never gain children- they only have inputs and outputs
            #print 'Operation - never changes its children'
            pass
        elif isinstance(x,dict):
            self.update_dict(x,idx)
            #print 'NEW n_children: {}'.format(idx.internalPointer().n_children())
        elif isinstance(x,list):
            self.update_list(x,idx)
            #print 'NEW n_children: {}'.format(idx.internalPointer().n_children())
        else:
            #print 'no new children'
            pass

    def update_dict(self,d,idx):
        itm = idx.internalPointer()
        child_keys = [itm.children[j].tag() for j in range(itm.n_children())]
        nc = itm.n_children()
        for k in d.keys():
            if not k in child_keys:
                self.beginInsertRows(idx,nc,nc)
                d_item = TreeItem(nc,0,idx)
                d_item.set_tag(k)
                d_item.data = d[k] 
                itm.children.insert(nc,d_item)
                self.build_next(d[k],self.index(nc,0,idx))
                self.endInsertRows()
                nc += 1
        # TODO: Will I ever need to remove children too?

    def update_list(self,l,idx):
        itm = idx.internalPointer()
        nit_old = itm.n_children()
        nit_new = len(l)
        for i in range(nit_old,nit_new):
            self.beginInsertRows(idx,i,i)
            l_item = TreeItem(i,0,idx)
            l_item.set_tag(str(i))
            l_item.data = l[i] 
            itm.children.insert(i,l_item)
            self.build_next(l[i],self.index(i,0,idx))
            self.endInsertRows()
        # TODO: Will I ever need to remove children too?

    def update_io_deps(self,uri,new_op=None):
        """
        Explicitly remove any broken dependencies in the workflow
        created by placing new_op (default new_op=None) at uri.
        """
        new_op_uri = uri.split('.')[0]
        # Loop through the existing ops...
        for row in range(len(self.root_items)):
            itm = self.root_items[row]
            op = itm.data
            idx = self.index(row,0,QtCore.QModelIndex())
            # For each input locator of this Operation...
            for name,il in op.input_locator.items():
                # If the source is workflow input (optools.wf_input)...
                if il.src == optools.wf_input:
                    if not il.tp == optools.list_type:
                        vals = [il.val]
                    else:
                        vals = il.val
                    for val in vals:
                        # if the uri(s) stored in il.val start(s) with new_op_uri...
                        uri_parts = val.split('.')
                        if uri_parts[0] == new_op_uri:
                            # then check if new_op will provide this input.
                            input_ok_flag = False
                            if new_op:
                                if len(uri_parts) < 3:
                                    # the input is either the Operation or its inputs or outputs dicts
                                    input_ok_flag = True
                                elif uri_parts[1] == optools.inputs_tag and uri_parts[2] in new_op.inputs.keys():
                                    input_ok_flag = True
                                elif uri_parts[1] == optools.outputs_tag and uri_parts[2] in new_op.outputs.keys():
                                    input_ok_flag = True
                            # If not, this input locator must be reset.
                            if not input_ok_flag:
                                op.input_locator[name] = optools.InputLocator()
                                il_uri = itm.tag()+'.'+optools.inputs_tag+'.'+name
                                il_itm,il_idx = self.get_from_uri(il_uri)
                                self.dataChanged.emit(il_idx,il_idx)

    def list_from_widget(self,widg):
        print '[{}]: need to implement list_from_widget'.format(__name__)
        return None

    def io_subtree(self,op,idx):
        """Add inputs and outputs subtrees as children of an Operation TreeItem"""
        # Get a reference to the parent item
        p_item = idx.internalPointer()
        # TreeItems inputs, outputs dicts
        i_item = TreeItem(optools.inputs_idx,0,idx)
        o_item = TreeItem(optools.outputs_idx,0,idx)
        i_item.set_tag(optools.inputs_tag)
        o_item.set_tag(optools.outputs_tag)
        i_item.data = op.input_locator
        o_item.data = op.outputs
        # Insert the new TreeItems
        self.beginInsertRows(idx,optools.inputs_idx,optools.outputs_idx)
        p_item.children.insert(optools.inputs_idx,i_item)
        p_item.children.insert(optools.outputs_idx,o_item)
        self.endInsertRows()
        # Build io trees from io dicts:
        i_idx = self.index(optools.inputs_idx,0,idx)
        o_idx = self.index(optools.outputs_idx,0,idx)
        self.build_next(op.input_locator,i_idx)
        self.build_next(op.output_container,o_idx)

    def build_from_dict(self,d,idx):
        """
        Add TreeItems from a dict, tagged by their dict keys.
        idx is the QModelIndex of the dict item.
        """
        nit = len(d)
        self.beginInsertRows(idx,0,nit-1)
        itm = idx.internalPointer()
        i=0
        for k,v in d.items():
            d_item = TreeItem(i,0,idx)
            d_item.set_tag(k)
            d_item.data = v
            itm.children.insert(i,d_item)
            self.build_next(v,self.index(i,0,idx))
            i += 1
        self.endInsertRows()

    def build_from_list(self,l,idx):
        """
        Add TreeItems from a list, tagged by their list index.
        idx is the QModelIndex of the list item.
        """
        nit = len(l)
        self.beginInsertRows(idx,0,nit-1)
        itm = idx.internalPointer()
        for i in range(nit):
            l_item = TreeItem(i,0,idx)
            l_item.set_tag(str(i))
            l_item.data = l[i] 
            itm.children.insert(i,l_item)
            self.build_next(l[i],self.index(i,0,idx))
        self.endInsertRows()
       
    def build_next(self,x,x_idx): 
        """
        Render TreeItems from an object, either an Operation, a dict, or a list.
        x_idx is the index of the input object x.
        """
        if isinstance(x,Operation):
            self.io_subtree(x,x_idx)
        elif isinstance(x,dict):
            self.build_from_dict(x,x_idx)
        elif isinstance(x,list):
            self.build_from_list(x,x_idx)
        else:
            pass

    # Overloaded data() for WfManager
    def data(self,item_indx,data_role):
        if (not item_indx.isValid()):
            return None
        item = item_indx.internalPointer()
        if item_indx.column() == 1:
            if item.data is not None:
                if ( isinstance(item.data,Operation)
                    or isinstance(item.data,list)
                    or isinstance(item.data,dict) ):
                    return type(item.data).__name__ 
                else:
                    return ' '
            else:
                return ' '
        else:
            return super(WfManager,self).data(item_indx,data_role)

    # Overloaded headerData() for WfManager 
    def headerData(self,section,orientation,data_role):
        if (data_role == QtCore.Qt.DisplayRole and section == 0):
            return "{} operation(s) loaded".format(self.rowCount(QtCore.QModelIndex()))
        elif (data_role == QtCore.Qt.DisplayRole and section == 1):
            return "type"
        else:
            return None

    # Overload columnCount()
    def columnCount(self,parent):
        """Let WfManager have two columns, one for item tag, one for item type"""
        return 2

    def check_wf(self):
        """
        Check the dependencies of the workflow.
        Ensure that all loaded operations have inputs that make sense.
        Return a status code and message for each of the Operations.
        """
        pass

    def find_batch_item(self):
        batch_items = [] 
        for item in self.root_items:
            if isinstance(item.data,Batch):
               batch_items.append(item)
        if len(batch_items) > 1:
            msg = 'Found {} Batches in workflow. Only one Batch Operation is currently supported.'.format(len(batch_items))
            raise ValueError(msg)
        elif batch_items:
            return batch_items[0] 
        else:
            return []

    def find_rt_item(self):
        rt_items = [] 
        for item in self.root_items:
            if isinstance(item.data,Realtime):
               rt_items.append(item)
        if len(rt_items) > 1:
            msg = 'Found {} Realtimes in workflow. Only one Realtime Operation is currently supported.'.format(len(rt_items))
            raise ValueError(msg)
        elif rt_items:
            return rt_items[0] 
        else:
            return []

    def run_deps(self,item):
       deps = self.upstream_stack(item)
       if deps:
           if self.logmethod:
               self.logmethod('Running dependencies for {}: {}'.format(item, [dep.tag() for dep in deps]))
           self.run_wf_serial(deps)
           #for dep in deps:
           #    self.run_and_update(item)

    @QtCore.Slot()
    def stop_wf(self):
        self._keep_going = False

    def run_wf(self):
        if self.find_rt_item():
            self.run_wf_realtime()
        elif self.find_batch_item():
            self.run_wf_batch()
        else:
            self.run_wf_serial()
        self.wfdone.emit()

    def next_available_thread(self):
        for idx,th in self._wf_threads.items():
            if not th:
                self._exec_ready = True
                return idx
            else:
                if th.isFinished():
                    th.finished.emit()
                    self._exec_ready = True
                    return idx
        #for idx,th in self._wf_threads.items():
        #    print 'thread {} running: {}'.format(idx,th.isRunning())
        self._exec_ready = False
        return None

    def run_wf_serial(self,stk=None):
        """
        Run the workflow by building a dependency stack 
        and running the operations in order.
        The dependency stack is a list of lists, 
        where each list contains the items that are ready to be executed,
        assuming the items in the lists above it have been executed already.
        """
        if self.logmethod:
            self.logmethod('SERIAL EXECUTION STARTING')
        if not stk:
            stk = self.serial_execution_stack()
        msg = '\n----\nexecution stack: '
        for to_run in stk:
            msg = msg + '\n{}'.format( [itm.tag() for itm in to_run] ) 
        msg += '\n----'
        if self.logmethod:
            self.logmethod(msg)
        for to_run in stk:
            msg = 'running {} '.format( [itm.tag() for itm in to_run] )
            if self.logmethod:
                self.logmethod(msg)
            th_idx = self.next_available_thread()
            while not self._exec_ready:
                # Use an event loop to non-busy wait    
                l = QtCore.QEventLoop()
                t = QtCore.QTimer()
                t.setSingleShot(True)
                t.timeout.connect(l.quit)
                t.start(1000)
                l.exec_()
                th_idx = self.next_available_thread()
            # We should be _exec_ready - load the inputs
            for itm in to_run:
                #print 'load inputs for '+itm.tag()
                #print 'before: {}'.format(itm.data.inputs)
                op = itm.data
                self.load_inputs(op)
                #print 'after: {}'.format(itm.data.inputs)
            # Make a new Worker, give None parent so that it can be thread-mobile
            wf_wkr = slacxtools.WfWorker(to_run,None)
            wf_thread = QtCore.QThread(self)
            wf_wkr.moveToThread(wf_thread)
            self._wf_threads[th_idx] = wf_thread
            wf_thread.started.connect( partial(self.start_thread,th_idx) )
            wf_thread.started.connect(wf_wkr.work)
            wf_thread.finished.connect( partial(self.finish_thread,th_idx) )
            wf_thread.start()
            # Calling wf_thread.wait() hands over control to wf_thread.
            # i.e. this makes the current thread wait on wf_thread.
            #self.appref.processEvents()
            wf_thread.wait()
            # When the thread is finished, update the ops it ran.
            for itm in to_run:
                op = itm.data
                self.update_op(itm.tag(),op)
        if self.logmethod:
            self.logmethod('SERIAL EXECUTION FINISHED')
            #self.appref.processEvents()

    def start_thread(self,th_idx):
        if self.logmethod:
            self.logmethod('beginning execution in thread {}...'.format(th_idx))
        self.appref.processEvents()

    def finish_thread(self,th_idx):
        if self.logmethod:
            self.logmethod('finished execution in thread {}.'.format(th_idx))
        self._wf_threads[th_idx] = None
        self.appref.processEvents()

    def run_wf_realtime(self):
        """
        Executes the workflow under the control of one Realtime(Operation) 
        """
        rt_item = self.find_rt_item() 
        #for rt_item in rt_items:
        if self.logmethod:
            self.logmethod( 'REALTIME EXECUTION STARTING' )
        if self.logmethod:
            self.logmethod( 'Running dependencies... ' )
        self.run_deps(rt_item)
        if self.logmethod:
            self.logmethod( 'Preparing Realtime controller... ' )
        rt = rt_item.data
        self.load_inputs(rt)
        rt.run_and_update()
        self.update_op(rt_item.tag(),rt)
        self.appref.processEvents()
        if rt.downstream_ops():
            to_run = [[optools.parse_wf_input(self,dsname,rt) for dsname in rt.downstream_ops()]]
        else:
            to_run = self.downstream_stack(rt_item)
        nx = 0
        while self._keep_going:
            # After rt.run(), it is expected that rt.input_iter()
            # will generate lists of input values whose respective routes are rt.input_routes().
            # unless there are no new inputs to run, in which case it will iterate None. 
            vals = rt.input_iter().next()
            inp_dict = dict( zip(rt.input_routes(), vals) )
            if inp_dict and not None in vals:
                wait_flag = False
                nx += 1
                for uri,val in inp_dict.items():
                    self.set_op_input_at_uri(uri,val)
                if self.logmethod:
                    self.logmethod( 'REALTIME EXECUTION {}'.format(nx))
                self.run_wf_serial(to_run)
                opdict = {}
                for op_list in to_run:
                    opdict.update(self.ops_as_dict(op_list))
                rt.output_list().append(opdict)
                self.update_op(rt_item.tag(),rt)
            else:
                if self.logmethod and not wait_flag:
                    self.logmethod( 'Waiting...' )
                wait_flag = True
            # start a local event loop to pause without busywaiting
            l = QtCore.QEventLoop()
            t = QtCore.QTimer()
            t.setSingleShot(True)
            t.timeout.connect(l.quit)
            t.start(rt.delay())
            l.exec_()

    def run_wf_batch(self):
        """
        Executes the workflow under the control of one Batch(Operation)
        """
        b_item = self.find_batch_item() 
        if self.logmethod:
            self.logmethod( 'BATCH EXECUTION STARTING' )
        if self.logmethod:
            self.logmethod( 'Running dependencies... ' )
        self.run_deps(b_item)
        if self.logmethod:
            self.logmethod( 'Preparing Batch controller... ' )
        b = b_item.data
        self.load_inputs(b)
        b.run_and_update()
        self.update_op(b_item.tag(),b)
        self.appref.processEvents()
        if b.downstream_ops():
            to_run = [[optools.parse_wf_input(self,dsname,b) for dsname in b.downstream_ops()]]
        else:
            to_run = self.downstream_stack(b_item)
        # After b.run(), it is expected that b.input_list()
        # will refer to a list of dicts, where each dict has the form [workflow tree uri:input value]. 
        for i in range(len(b.input_list())):
            if self._keep_going:
                input_dict = b.input_list()[i]
                for uri,val in input_dict.items():
                    self.set_op_input_at_uri(uri,val)
                # inputs are set, run in serial 
                if self.logmethod:
                    self.logmethod( 'BATCH EXECUTION {} / {}'.format(i+1,len(b.input_list())) )
                self.run_wf_serial(to_run)
                for op_list in to_run:
                    b.output_list()[i].update(self.ops_as_dict(op_list))
                self.update_op(b_item.tag(),b)
        if self.logmethod:
            self.logmethod( 'BATCH EXECUTION FINISHED' )

    def set_op_input_at_uri(self,uri,val):
        """Set an op input, indicated by uri, to provided value."""
        path = uri.split('.')
        if not len(path) == 3:
            msg = 'uri '+uri+' should have format Operation.'+optools.inputs_tag+'.inputname'
            raise ValueError(msg)
        op_itm, idx = self.get_from_uri(path[0])
        op = op_itm.data
        if path[1] == optools.inputs_tag and path[2] in op.inputs.keys():
            op.inputs[path[2]] = val
        else:
            msg = 'uri {} does not specify inputs, or specifies an invalid inputname'.format(uri)
            raise ValueError(msg)

    def ops_as_dict(self,op_items=None):
        od = OrderedDict()
        if not op_items:
            op_items = self.root_items
        for itm in op_items:
            od[itm.tag()] = copy.deepcopy(itm.data)
        return od

    def serial_execution_stack(self):
        """
        Get a stack (list of lists) of Operations,
        such that each list contains a set of Operations whose dependencies are satisfied
        assuming all operations above them have been executed successfully.
        """
        ordered_items = []
        item_stack = []
        next_items = []
        # Build the first layer of the stack... screen for batch and workflow inputs
        for itm in self.root_items:
            print 'stack item: {}'.format(itm.tag())
            op_rdy = True
            op = itm.data
            inp_srcs = [il.src for name,il in op.input_locator.items()] 
            if optools.batch_input in inp_srcs: 
                # This Op is not ready until its Batch controller has run.
                op_rdy = False
                print 'HAS BATCH INPUTS.'
            elif optools.wf_input in inp_srcs:
                # If the Operation is not a Batch or Realtime, it must not be ready
                if not isinstance(op,Batch) and not isinstance(op,Realtime):
                    op_rdy = False
                    print 'HAS WF INPUTS AND IS NOT BATCH OR REALTIME.'
                else:
                    # If it is Batch or Realtime, check if this is one of the input_routes() 
                    if not il.val in op.input_routes():
                        print 'HAS WF INPUTS THAT ARE NOT INPUT ROUTES.'
                        op_rdy = False
            if op_rdy:
                print 'OP IS READY.'
                next_items.append(itm)
        #next_items = self.items_ready(ordered_items)
        while next_items:
            ordered_items = ordered_items + next_items
            item_stack.append( next_items )
            next_items = self.items_ready(ordered_items)
        return item_stack 

    def upstream_stack(self,root_item):
        """
        Get the portion of serial_execution_stack() that is upstream of a given item
        """
        stk = self.serial_execution_stack()
        substk = []
        for lst in stk:
            if root_item in lst:
                return
            else:
                substk.append(lst)
        return substk
        #ordered_items = [root_item]
        #done = False
        #while not done:
        #    done = True
        #    for item in ordered_items:
        #        op = item.data
        #        for name in op.inputs.keys():
        #            # Check if this input is supposed to come from a field in another Operation
        #            # TODO: ensure that when this is called, all Input Locators have already been loaded.
        #            src = op.input_locator[name].src
        #            tp = op.input_locator[name].tp
        #            if src == optools.wf_input:
        #                if tp == optools.list_type:
        #                    uris = op.input_locator[name].val
        #                else:
        #                    uris = [op.input_locator[name].val]
        #                for uri in uris:
        #                    # TODO: Check if this is an Input field that is not in Batch.input_routes()
        #                    # Check whether or not this is an Output field 
        #                    uri_items = uri.split('.')
        #                    if len(uri_items) > 1:
        #                        op_tag = uri_items[0]
        #                        io_tag = uri_items[1]
        #                        op_item, indx = self.get_from_uri(op_tag)
        #                        if io_tag == 'outputs' and not op_item in ordered_items:
        #                            ordered_items.insert(0,op_item)
        #                            done = False
        ## Remove root_item from the end
        #ordered_items.pop(-1)
        #return ordered_items

    def downstream_stack(self,root_item):
        """
        Get the portion of serial_execution_stack() that is level with or downstream of a given item
        """
        stk = self.serial_execution_stack()
        print 'full stack:'
        for itms in stk:
            print [itm.tag() for itm in itms]
        substk = []
        for lst in stk:
            if root_item in lst:
                lst.pop(lst.index(root_item))
                if lst:
                    substk = [lst]
                else:
                    substk = []
            else:
                substk.append(lst)
        print 'substack:'
        for itms in substk:
            print [itm.tag() for itm in itms]
        return substk
        #ds_items = []
        #op = op_item.data
        ## Get uris of inputs that will be set by the op 
        #inroutes = op.input_routes()
        #for uri in inroutes:
        #    in_op_tag = uri.split('.')[0]
        #    in_op_item, indx = self.get_from_uri(in_op_tag)
        #    # Exclude Batches and Realtimes from downstream ops.
        #    # TODO: Consider how / whether embedded Batches / Realtimes should work
        #    if ( not isinstance(in_op_item.data,Batch)
        #    and not isinstance(in_op_item.data,Realtime) ):
        #        ds_items.append(in_op_item)
        #next_items = self.items_ready(ds_items)
        #while next_items:
        #    next_items = self.items_ready(ds_items)
        #    pop_list = []
        #    for i in range(len(next_items)):
        #        in_item = next_items[i]
        #        if isinstance(in_item.data,Batch) or isinstance(in_item.data,Realtime):
        #            pop_list.append(i)
        #        else:
        #            ds_items.append(in_item)
        #    for idx in pop_list:
        #        next_items.pop(idx)
        #return ds_items

    def items_ready(self,items_done):
        """
        Give a list of Operation items whose inputs are satisfied, given items_done
        """
        rdy = []
        print 'checking for ops ready given:'
        print [itm.tag() for itm in items_done]
        for itm in self.root_items:
            op = itm.data
            op_rdy = True
            for name,il in op.input_locator.items():
                src = il.src
                if src == optools.wf_input:
                    tp = op.input_locator[name].tp
                    if tp == optools.list_type:
                        uris = op.input_locator[name].val
                    else:
                        uris = [op.input_locator[name].val]
                    for inp_uri in uris:
                        uri_fields = inp_uri.split('.')
                        # Get the op item and see if it is in items_done
                        op_itm,idx = self.get_from_uri(uri_fields[0])
                        if not op_itm in items_done:
                            op_rdy = False
                        # Get the op.inout.name three-level uri
                        #uri_tl = uri_fields[0]+'.'+uri_fields[1]+'.'+uri_fields[2] 
                        #if not self.is_good_uri(uri_tl):
                        #    op_rdy = False
                elif src == optools.batch_input:
                    inp_uri = itm.tag()+'.'+optools.inputs_tag+'.'+name
                    # Look for a Batch in items_done
                    b_itm = self.find_batch_item()
                    rt_itm = self.find_rt_item()
                    if not b_itm and not rt_itm:
                        op_rdy = False
                    elif b_itm:
                        if not inp_uri in b_itm.data.input_routes():
                            op_rdy = False
                    elif rt_itm:
                        if not inp_uri in rt_itm.data.input_routes():
                            op_rdy = False
            if not itm in items_done and op_rdy:
                rdy.append(itm)
        return rdy 

    def get_from_uri(self, uri):
        """Get from this tree the item at the given uri."""
        path = uri.split('.')
        p_idx = QtCore.QModelIndex()
        try:
            for itemuri in path:
                # get QModelIndex of item 
                row = self.list_tags(p_idx).index(itemuri)
                idx = self.index(row,0,p_idx)
                # get TreeItem from QModelIndex
                item = self.get_item(idx)
                # set new parent in case the path continues...
                p_idx = idx
            return item, idx
        except Exception as ex:
            msg = '-----\nbad uri: {}\n-----'.format(uri)
            print msg
            raise ex

    def next_uri(self,prefix):
        indx = 0
        goodtag = False
        while not goodtag:
            testtag = prefix+'_{}'.format(indx)
            if not testtag in self.list_tags(QtCore.QModelIndex()):
                goodtag = True
            else:
                indx += 1
        return testtag
                    
    def is_good_uri(self,uri):
        path = uri.split('.')
        p_idx = QtCore.QModelIndex()
        for itemuri in path:
            try:
                row = self.list_tags(p_idx).index(itemuri)
            except ValueError as ex:
                return False
            idx = self.index(row,0,p_idx)
            # get TreeItem from QModelIndex
            item = self.get_item(idx)
            # set new parent in case the path continues...
            p_idx = idx
        return True

    #def run_wf_graph(self):
    #    """
    #    Run the workflow by building a dask-compatible dict,
    #    then calling dask.threaded.get(dict, key)
    #    for each of the keys corresponding to operation outputs.
    #    TODO: optimize the execution of this by making the smallest
    #    possible number of calls to get().
    #    """
    #    # build the graph, get the list of outputs
    #    outputs_list = self.load_wf_dict()
    #    print 'workflow graph as dict:'
    #    print self._wf_dict

    #def load_wf_dict(self):
    #    """
    #    Build a dask-compatible dictionary from the Operations in this tree
    #    """
    #    self._wf_dict = {}
    #    for j in range(len(self.root_items)):
    #        item = self.root_items[j]
    #        # Unpack the Operation
    #        op = item.data
    #        keyindx = 0
    #        input_keys = [] 
    #        input_vals = ()
    #        for name,val in op.inputs.items():
    #            # Add a locate_input line for each input 
    #            dask_key = 'op'+str(j)+'inp'+str(keyindx)
    #            self._wf_dict[dask_key] = (self.locate_input, val, op)
    #            keyindx += 1
    #            input_keys.append(name)
    #            input_vals = input_vals + (dask_key)
    #        # Add a set_inputs line for op j
    #        dask_key = 'op'+str(j)+'_load'
    #        self._wf_dict[key] = (self.set_inputs, op, input_keys, input_vals) 
    #        # Add a run_op line for op j
    #        dask_key = 'op'+str(j)+'_run'
    #        self._wf_dict[key] = (self.run_op, op) 
    #        # Keep track of the keys corresponding to operation outputs.
    #        keyindx = 0
    #        output_keys = []
    #        for name,val in op.outputs.items():
    #            # Add a get_output line for each output
    #            dask_key = 'op'+str(j)+'out'+str()
    #            self._wf_dict[dask_key] = (self.get_output, val)
    #            keyindx += 1
    #            output_keys.append(name)

    #@staticmethod
    #def set_inputs(op,keys,vals):
    #    """
    #    By the time this is called, vals should be bound to actual input values by dask.
    #    Each dask key should have been assigned to a (self.locate_input, val)
    #    """
    #    for i in range(len(keys)):
    #        key = keys[i]
    #        val = vals[i] 
    #        op.inputs[key] = val
    #    return op 

    #@staticmethod
    #def run_op(op):
    #    return op.run()

