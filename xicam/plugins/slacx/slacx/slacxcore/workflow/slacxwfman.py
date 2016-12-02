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
from ..operations.optools import InputLocator#, OutputContainer
from .. import slacxtools


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
        self._running = False
        #self._n_threads = QtCore.QThread.idealThreadCount()
        self._n_threads = 1
        self._wf_threads = dict.fromkeys(range(self._n_threads)) 
        if self.logmethod:
            self.logmethod('Slacx workflow manager started, working with {} threads'.format(self._n_threads))

    def load_from_file(self,opman,wfl):
        """
        Load things in to the Workflow from an OpManager and a YAML .wfl file 
        """
        # TODO: Migrate to own module
        while self.root_items:
            idx = self.index(self.rowCount(QtCore.QModelIndex())-1,0,QtCore.QModelIndex())
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
    def ops_as_dict(self,op_items=None):
        od = OrderedDict()
        if not op_items:
            op_items = self.root_items
        for itm in op_items:
            od[itm.tag()] = copy.deepcopy(itm.data)
        return od
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

    def locate_input(self,il,op):
        """
        Return the data pointed to by a given InputLocator object.
        Takes the Operation that owns this inplocator as a second arg,
        so that if it is a Batch its input routes can be handled properly.
        """
        #if isinstance(inplocator,InputLocator):
        src = il.src
        tp = il.tp
        val = il.val
        if src == optools.no_input: 
            return None
        elif src == optools.user_input: 
            return optools.cast_type_val(tp,val)
        elif src == optools.wf_input:
            return optools.parse_wf_input(self,il,op)
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


    def add_op(self,uri,new_op):
        """Add an Operation to the tree as a new top-level TreeItem."""
        # Count top-level rows by passing parent=QModelIndex()
        ins_row = self.rowCount(QtCore.QModelIndex())
        itm = TreeItem(ins_row,0,QtCore.QModelIndex())
        itm.set_tag( uri )
        self.beginInsertRows(QtCore.QModelIndex(),ins_row,ins_row)
        self.root_items.insert(ins_row,itm)
        self.endInsertRows()
        idx = self.index(ins_row,0,QtCore.QModelIndex()) 
        self.tree_update(idx,new_op)

    def remove_op(self,rm_idx):
        """Remove an Operation from the workflow tree"""
        rm_row = rm_idx.row()
        self.beginRemoveRows(QtCore.QModelIndex(),rm_row,rm_row)
        # Removal occurs between notification methods
        item_removed = self.root_items.pop(rm_row)
        self.endRemoveRows()
        # Inform views 
        self.tree_dataChanged(rm_idx)
        # Update any Operations that depended on the removed one
        self.update_io_deps()

    def update_op(self,uri,new_op):
        """
        Update Operation in treeitem indicated by uri.
        It is expected that new_op is a reference to the Operation stored at uri. 
        """
        itm, idx = self.get_from_uri(uri)
        self.tree_update(idx,new_op)

    def tree_update(self,idx,x_new):
        """
        Call this function to store x_new in the TreeItem at idx 
        and then build/update/prune the subtree rooted at that item.
        """
        itm = idx.internalPointer()
        x = itm.data
        itm.data = x_new
        # Build dict of the intended children 
        x_dict = optools.get_child_dict(x_new)
        # Remove obsolete children
        c_kill = [] 
        for j in range(itm.n_children()):
            if not self.index(j,0,idx).internalPointer().tag() in x_dict.keys():
                c_kill.append( j )
        c_kill.sort()
        for j in c_kill[::-1]:
            self.beginRemoveRows(idx,j,j)
            itm.children.pop(j)
            self.endRemoveRows()
        # Add items for any new children 
        c_keys = [itm.children[j].tag() for j in range(itm.n_children())]
        for k in x_dict.keys():
            if not k in c_keys:
                nc = itm.n_children()
                c_itm = TreeItem(nc,0,idx)
                c_itm.set_tag(k)
                self.beginInsertRows(idx,nc,nc)
                itm.children.insert(nc,c_itm)
                self.endInsertRows()
        # Recurse to update children
        for j in range(itm.n_children()):
            c_idx = self.index(j,0,idx)
            c_tag = c_idx.internalPointer().tag()
            self.tree_update(c_idx,x_dict[c_tag])
        # If x is (was) an Operation, update workflow IO dependencies.
        if isinstance(x,Operation):
            self.update_io_deps()
        # Finish by informing views that dataChanged().
        self.tree_dataChanged(idx) 

    def update_io_deps(self):
        """
        Remove any broken dependencies in the workflow.
        NB: Only effective after all changes to tree data are finished. 
        """
        #itm = idx.internalPointer()
        #update_uri = itm.tag()
        for row in range(len(self.root_items)):
            itm = self.root_items[row]
            op = itm.data
            #op_idx = self.index(row,0,QtCore.QModelIndex())
            for name,il in op.input_locator.items():
                self.update_input_locator(il)

    def update_input_locator(self,il):
        """
        Clear an input_locator if the uris it points to are obsolete.
        This helps clean up after an Operation is altered or removed.
        """
        # If the source is workflow input (optools.wf_input)...
        if il.src == optools.wf_input:
            vals = optools.val_list(il)
            for v in vals:
                # Check if new_op will provide this input.
                input_ok_flag = self.is_good_uri(v) 
                # If not, this should be reset. 
                if not input_ok_flag:
                    if self.logmethod:
                        self.logmethod('--- NB: clearing InputLocator for {} ---'.format(v))
                    il = optools.InputLocator()

    def tree_dataChanged(self,idx):
        self.dataChanged.emit(idx,idx)
        itm = idx.internalPointer()
        for c_row in range(itm.n_children()):
            c_idx = self.index(c_row,0,idx)
            self.tree_dataChanged(c_idx)

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

    def is_running(self):
        return self._running

    def stop_wf(self):
        self._running = False

    def run_wf(self):
        self._running = True
        if self.find_rt_item():
            self.run_wf_realtime()
        elif self.find_batch_item():
            self.run_wf_batch()
        else:
            self.run_wf_serial()
        if self.is_running():
            self.wfdone.emit()

    def next_available_thread(self):
        for idx,th in self._wf_threads.items():
            if not th:
                return idx
        return None

    def wait_for_thread(self,th_idx):
        """Wait for the thread at th_idx to be finished"""
        done = False
        interval = 10
        wait_iter = 0
        while not done:
            done = True
            if self._wf_threads[th_idx]:
                if not self._wf_threads[th_idx].isFinished():
                    done = False
                if not done:
                    self.loopwait(interval)
                    self.appref.processEvents()
                    wait_iter += 1
        if self.logmethod and wait_iter > 0:
            self.logmethod('... waited {}ms for thread {}'.format(wait_iter*interval,th_idx))
            self.appref.processEvents()

    def wait_for_threads(self):
        """Wait for all workflow execution threads to finish"""
        done = False
        interval = 10
        wait_iter = 0
        while not done:
            done = True
            for idx,th in self._wf_threads.items():
                if not th.isFinished():
                    done = False
            if not done:
                self.loopwait(interval)
                self.appref.processEvents()
                wait_iter += 1
        if self.logmethod and wait_iter > 0:
            self.logmethod('... waited {}ms for threads to finish'.format(wait_iter*interval))
            self.appref.processEvents()

    def loopwait(self,interval):
        l = QtCore.QEventLoop()
        t = QtCore.QTimer()
        t.setSingleShot(True)
        t.timeout.connect(l.quit)
        t.start(interval)
        l.exec_()
        # This processEvents() is meant to process any Signals
        # that were emitted during waiting.
        self.appref.processEvents()

    def run_wf_serial(self,stk=None,thd=0):
        """
        Run the workflow by building a dependency stack 
        and running the operations in order.
        The dependency stack is a list of lists, 
        where each list contains the items that are ready to be executed,
        assuming the items in the lists above it have been executed already.
        """
        if self.logmethod:
            self.logmethod('SERIAL EXECUTION STARTING in thread {}'.format(thd))
            self.appref.processEvents()
        if not stk:
            stk = self.serial_execution_stack()
        msg = '\n----\nexecution stack: '
        for to_run in stk:
            msg = msg + '\n{}'.format( [itm.tag() for itm in to_run] ) 
        msg += '\n----'
        if self.logmethod:
            self.logmethod(msg)
            self.appref.processEvents()
        for to_run in stk:
            self.wait_for_thread(thd)
            for itm in to_run:
                op = itm.data
                self.load_inputs(op)
            # Make a new Worker, give None parent so that it can be thread-mobile
            wf_wkr = slacxtools.WfWorker(to_run,None)
            wf_thread = QtCore.QThread(self)
            wf_wkr.moveToThread(wf_thread)
            self._wf_threads[thd] = wf_thread
            wf_thread.started.connect(wf_wkr.work)
            wf_thread.finished.connect( partial(self.finish_thread,thd) )
            msg = 'running {} in thread {}'.format([itm.tag() for itm in to_run],thd)
            if self.logmethod:
                self.logmethod(msg)
                self.appref.processEvents()
            wf_thread.start()
            # Let the thread finish
            self.wait_for_thread(thd)
            # When the thread is finished, update the ops it ran.
            for itm in to_run:
                op = itm.data
                self.update_op(itm.tag(),op)
        if self.logmethod:
            self.logmethod('SERIAL EXECUTION FINISHED in thread {}'.format(thd))
            self.appref.processEvents()

    def finish_thread(self,th_idx):
        if self.logmethod:
            self.logmethod('finished execution in thread {}.'.format(th_idx))
            self.appref.processEvents()
        self._wf_threads[th_idx] = None

    def run_wf_realtime(self):
        """
        Executes the workflow under the control of one Realtime(Operation) 
        """
        rt_item = self.find_rt_item() 
        #for rt_item in rt_items:
        if self.logmethod:
            self.logmethod( 'REALTIME EXECUTION STARTING' )
            self.logmethod( 'Running dependencies... ' )
            self.appref.processEvents()
        self.run_deps(rt_item)
        if self.logmethod:
            self.logmethod( 'Preparing Realtime controller... ' )
            self.appref.processEvents()
        rt = rt_item.data
        self.load_inputs(rt)
        #rt.run_and_update()
        rt.run()
        self.update_op(rt_item.tag(),rt)
        self.appref.processEvents()
        to_run = self.downstream_stack(rt_item)
        nx = 0
        while self._running:
            # After rt.run(), it is expected that rt.input_iter()
            # will generate lists of input values whose respective routes are rt.input_routes().
            # unless there are no new inputs to run, in which case it will iterate None. 
            vals = rt.input_iter().next()
            inp_dict = dict( zip(rt.input_routes(), vals) )
            if inp_dict and not None in vals:
                waiting_flag = False
                nx += 1
                for uri,val in inp_dict.items():
                    self.set_op_input_at_uri(uri,val)
                thd = self.next_available_thread()
                if self.logmethod:
                    self.logmethod( 'REALTIME EXECUTION {} in thread {}'.format(nx,thd))
                    self.appref.processEvents()
                self.run_wf_serial(to_run,thd)
                opdict = {}
                for op_list in to_run:
                    opdict.update(self.ops_as_dict(op_list))
                rt.output_list().append(opdict)
                self.update_op(rt_item.tag(),rt)
            else:
                if self.logmethod and not waiting_flag:
                    self.logmethod( 'Waiting for new inputs...' )
                    self.appref.processEvents()
                waiting_flag = True
                self.loopwait(rt.delay())
        self.logmethod( 'REALTIME EXECUTION TERMINATED' )
        self.appref.processEvents()
        return

    def run_wf_batch(self):
        """
        Executes the workflow under the control of one Batch(Operation)
        """
        b_item = self.find_batch_item() 
        if self.logmethod:
            self.logmethod( 'BATCH EXECUTION STARTING' )
            self.logmethod( 'Running dependencies... ' )
            self.appref.processEvents()
        self.run_deps(b_item)
        if self.logmethod:
            self.logmethod( 'Preparing Batch controller... ' )
            self.appref.processEvents()
        b = b_item.data
        self.load_inputs(b)
        #b.run_and_update()
        b.run()
        self.update_op(b_item.tag(),b)
        self.appref.processEvents()
        to_run = self.downstream_stack(b_item)
        # After b.run(), it is expected that b.input_list() will refer to a list of dicts,
        # where each dict has the form [workflow tree uri:input value]. 
        for i in range(len(b.input_list())):
            if self._running:
                input_dict = b.input_list()[i]
                for uri,val in input_dict.items():
                    self.set_op_input_at_uri(uri,val)
                # inputs are set, run in serial 
                thd = self.next_available_thread()
                if self.logmethod:
                    self.logmethod( 'BATCH EXECUTION {} / {} in thread {}'.format(i+1,len(b.input_list()),thd) )
                    self.appref.processEvents()
                self.run_wf_serial(to_run,thd)
                for op_list in to_run:
                    b.output_list()[i].update(self.ops_as_dict(op_list))
                self.update_op(b_item.tag(),b)
            elif self.logmethod:
                self.logmethod( 'BATCH EXECUTION TERMINATED' )
                self.appref.processEvents()
                return
        if self.logmethod:
            self.logmethod( 'BATCH EXECUTION FINISHED' )
            self.appref.processEvents()

    def set_op_input_at_uri(self,uri,val):
        """Set an op input, indicated by uri, to provided value."""
        p = uri.split('.')
        # Allow some structure here: expect no meta-inputs. 
        op_itm, idx = self.get_from_uri(p[0])
        op = op_itm.data
        op.inputs[p[2]] = val

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
        #import pdb; pdb.set_trace()
        for itm in self.root_items:
            op_rdy = True
            op = itm.data
            inp_srcs = [il.src for name,il in op.input_locator.items()] 
            if optools.batch_input in inp_srcs: 
                # This Op is not ready until its Batch controller has run.
                op_rdy = False
            elif optools.wf_input in inp_srcs:
                # If the Operation is not a Batch or Realtime, it must not be ready
                if not isinstance(op,Batch) and not isinstance(op,Realtime):
                    op_rdy = False
                else:
                    # If it is Batch or Realtime, check if this is one of the input_routes() 
                    if not il.val in op.input_routes():
                        op_rdy = False
            if op_rdy:
                next_items.append(itm)
        #next_items = self.items_ready(ordered_items)
        while next_items:
            ordered_items = ordered_items + next_items
            item_stack.append( next_items )
            next_items = self.items_ready(ordered_items)
        #print item_stack
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

    def downstream_stack(self,root_item):
        """
        Get the portion of serial_execution_stack() that is level with or downstream of a given item
        """
        stk = self.serial_execution_stack()
        #print stk
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
        return substk

    def items_ready(self,items_done):
        """
        Give a list of Operation items whose inputs are satisfied, given items_done
        """
        rdy = []
        for itm in self.root_items:
            op = itm.data
            op_rdy = True
            for name,il in op.input_locator.items():
                src = il.src
                if src == optools.wf_input:
                    uris = optools.val_list(il)
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

