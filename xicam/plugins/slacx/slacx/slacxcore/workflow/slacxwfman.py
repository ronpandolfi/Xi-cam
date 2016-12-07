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

    # TODO: Make appref a required init arg
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
        self.write_log('Slacx workflow manager started, working with {} threads'.format(self._n_threads))

    def write_log(self,msg):
        if self.logmethod:
            self.logmethod(msg)
        else:
            print(msg)
        if self.appref:
            self.appref.processEvents()

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
        self.write_log( 'dumping current workflow image to {}'.format(filename) )
        f = open(filename, "w")
        #yaml.dump(wf_dict, f, encoding='utf-8')
        yaml.dump(wf_dict, f)
        f.close()
    def op_items_to_dict(self,op_items=None):
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
        if not uri:
            return True
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
        NB: Only effective after most recent data have been stored in the tree. 
        """
        #itm = idx.internalPointer()
        #update_uri = itm.tag()
        for itm in self.root_items:
            op = itm.data
            #op_idx = self.index(row,0,QtCore.QModelIndex())
            for name,il in op.input_locator.items():
                if il:
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
                if not self.is_good_uri(v):
                    self.write_log('--- NB: clearing InputLocator for {} ---'.format(v))
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

    def find_batch_items(self):
        batch_items = [] 
        for item in self.root_items:
            if isinstance(item.data,Batch):
               batch_items.append(item)
        return batch_items

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

#    def run_deps(self,item):
#       deps = self.upstream_stack(item)
#       if deps:
#           self.write_log('Running dependencies for {}: {}'.format(item, [dep.tag() for dep in deps]))
#           self.run_wf_serial(deps)
#           #for dep in deps:
#           #    self.run_and_update(item)

    def is_running(self):
        return self._running

    def stop_wf(self):
        self._running = False

    def run_wf(self):
        self._running = True
        stk = self.execution_stack()
        msg = 'STARTING EXECUTION\n----\nexecution stack: '
        for to_run in stk:
            msg = msg + '\n{}'.format( [itm.tag() for itm in to_run] ) 
        msg += '\n----'
        self.write_log(msg)
        substk = []
        batch_flags = []
        batch_idx = []
        rt_flags = []
        rt_idx = []
        for i in range(len(stk)):
            lst = stk[i]
            if isinstance(lst[0].data,Batch):
                batch_flags.append(True)
                batch_idx.append(i)
                rt_flags.append(False)
            elif isinstance(lst[0].data,Realtime):
                batch_flags.append(False)
                rt_flags.append(True)
                rt_idx.append(i)
            else:   
                batch_flags.append(False)
                rt_flags.append(False)
        if sum(rt_flags) == 1 and not any(batch_flags):
            # Expect only one rt controller at a time.
            rt_itm = stk[rt_idx[0]][0]
            prestk = stk[:rt_idx[0]]
            itms_run = []
            if any(prestk):
                msg = '\n----\n pre-realtime execution stack: '
                for lst in prestk:
                    msg = msg + '\n{}'.format( [itm.tag() for itm in lst] ) 
                    itms_run = itms_run + lst 
                msg += '\n----'
                self.write_log(msg)
                self.run_wf_serial(prestk)
                rtstk = self.downstream_from_batch_item(rt_itm,prestk)
                if any(rtstk):
                    msg = '\n----\n realtime execution stack: '
                    for lst in rtstk:
                        msg = msg + '\n{}'.format( [itm.tag() for itm in lst] ) 
                        itms_run = itms_run + lst 
                    msg += '\n----'
                    self.write_log(msg)
                    self.run_wf_realtime(rt_itm,rtstk)
            poststk = []
            for lst in stk:
                postlst = [itm for itm in lst if not itm in itms_run and not isinstance(itm.data,Realtime)] 
                if any(postlst):
                    poststk.append(postlst)
            if any(poststk):
                msg = '\n----\n post-realtime execution stack: '
                for to_run in poststk:
                    msg = msg + '\n{}'.format( [itm.tag() for itm in to_run] ) 
                msg += '\n----'
                self.write_log(msg)
                self.run_wf_serial(poststk)
        elif any(rt_flags):
            raise ValueError('[{}] only one Realtime op at a time is supported, found {}'.format(
            __name__,sum(rt_flags)+sum(batch_flags)))
        elif any(batch_flags):
            n_batch = sum(batch_flags)
            b_itms = [stk[i][0] for i in batch_idx]
            itms_run = []
            for i in range(n_batch):
                prestk = [] 
                for lst in stk[:batch_idx[i]]:
                    prelst = [itm for itm in lst if not itm in itms_run and not isinstance(itm.data,Batch)]
                    if any(prelst):
                        prestk.append(prelst)
                if any(prestk):
                    msg = '\n----\n pre-batch execution stack: '
                    for lst in prestk:
                        msg = msg + '\n{}'.format( [itm.tag() for itm in lst] ) 
                        itms_run = itms_run + lst 
                    msg += '\n----'
                    self.write_log(msg)
                    self.run_wf_serial(prestk)
                bstk = self.downstream_from_batch_item(b_itms[i],stk[:batch_idx[i]])
                if any(bstk):
                    msg = '\n----\n batch execution stack: '
                    for lst in bstk:
                        msg = msg + '\n{}'.format( [itm.tag() for itm in lst] ) 
                        itms_run = itms_run + lst 
                    msg += '\n----'
                    self.write_log(msg)
                    self.run_wf_batch(b_itms[i],bstk)
            # any more serial ops?
            poststk = []
            for lst in stk:
                postlst = [itm for itm in lst if not itm in itms_run and not isinstance(itm.data,Batch)] 
                if any(postlst):
                    poststk.append(postlst)
            if any(poststk):
                msg = '\n----\n post-batch execution stack: '
                for to_run in poststk:
                    msg = msg + '\n{}'.format( [itm.tag() for itm in to_run] ) 
                msg += '\n----'
                self.write_log(msg)
                self.run_wf_serial(poststk)
        else:
            self.run_wf_serial(stk)
        # if not interrupted, signal done
        if self.is_running():
            self.wfdone.emit()

    def downstream_from_batch_item(self,b_itm,stk_done):
        stk = []
        # The top layer will be strictly the batch input routes
        lst = []
        for uri in b_itm.data.input_routes():
            op_uri = uri.split('.')[0]
            itm,idx = self.get_from_uri(op_uri)
            lst.append(itm)
        while any(lst):
            stk.append(lst)
            lst = []
            for itm in self.root_items:                
                op = itm.data
                if ( not isinstance(op,Batch) 
                and not isinstance(op,Realtime)
                and not any([itm in l for l in stk_done+stk])
                and self.batch_op_ready(op,stk_done+stk) ):
                    lst.append(itm)
        return stk
                
    def batch_op_ready(self,op,stk_done):
        op_rdy = True
        for name,il in op.input_locator.items():
            if il.src == optools.wf_input: 
                op_uri = il.val.split('.')[0]
                itm,idx = self.get_from_uri(op_uri)
                if not any([itm in lst for lst in stk_done]):
                    op_rdy = False
            if il.src == optools.batch_input:
                # assume all ops taking batch input
                # were processed in the top layer of the stack
                op_rdy = False
                
        return op_rdy

    def next_available_thread(self):
        for idx,th in self._wf_threads.items():
            if not th:
                return idx
        # if none found, wait for thread 0
        # TODO: something better
        self.wait_for_thread(0)
        return 0 

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
        if wait_iter > 0:
            self.write_log('... waited {}ms for thread {}'.format(wait_iter*interval,th_idx))

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
        if wait_iter > 0:
            self.write_log('... waited {}ms for threads to finish'.format(wait_iter*interval))

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
        self.write_log('SERIAL EXECUTION STARTING in thread {}'.format(thd))
        if not stk:
            stk = self.execution_stack()
        for lst in stk:
            self.wait_for_thread(thd)
            for itm in lst: 
                op = itm.data
                self.load_inputs(op)
            # Make a new Worker, give None parent so that it can be thread-mobile
            wf_wkr = slacxtools.WfWorker(lst,None)
            wf_wkr.opDone.connect(self.updateOperation)
            wf_thread = QtCore.QThread(self)
            wf_wkr.moveToThread(wf_thread)
            wf_thread.started.connect(wf_wkr.work)
            wf_thread.finished.connect( partial(self.finish_thread,thd) )
            msg = 'running {} in thread {}'.format([itm.tag() for itm in lst],thd)
            self.write_log(msg)
            self._wf_threads[thd] = wf_thread
            wf_thread.start()
            # Let the thread finish
            #self.wait_for_thread(thd)
            # When the thread is finished, update the ops it ran.
            #for itm in lst:
            #    op = itm.data
            #    self.update_op(itm.tag(),op)
        self.wait_for_thread(thd)
        self.write_log('SERIAL EXECUTION FINISHED in thread {}'.format(thd))

    @QtCore.Slot(str,Operation)
    def updateOperation(self,tag,op):
        #print 'updating op for {}'.format(tag)
        self.update_op(tag,op)

    def finish_thread(self,th_idx):
        self.write_log('finished execution in thread {}.'.format(th_idx))
        self._wf_threads[th_idx] = None

    def run_wf_realtime(self,rt_itm,stk):
        """
        Executes the workflow under the control of one Realtime(Operation) 
        """
        self.write_log( 'Preparing Realtime controller... ' )
        rt = rt_itm.data
        self.load_inputs(rt)
        rt.run()
        self.update_op(rt_itm.tag(),rt)
        self.appref.processEvents()
        stk = self.downstream_from_batch_item(rt_itm)
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
                self.write_log( 'REALTIME EXECUTION {} in thread {}'.format(nx,thd))
                self.run_wf_serial(stk,thd)
                opdict = {}
                for lst in stk:
                    opdict.update(self.op_items_to_dict(lst))
                rt.output_list().append(opdict)
                self.update_op(rt_itm.tag(),rt)
            else:
                self.write_log( 'Waiting for new inputs...' )
                waiting_flag = True
                self.loopwait(rt.delay())
        self.write_log( 'REALTIME EXECUTION TERMINATED' )
        return

    def run_wf_batch(self,b_itm,stk):
        """
        Executes the items in the stack stk under the control of one Batch(Operation).
        """
        self.write_log( 'Preparing Batch controller... ' )
        b = b_itm.data
        self.load_inputs(b)
        b.run()
        self.update_op(b_itm.tag(),b)
        self.appref.processEvents()
        # After b.run(), it is expected that b.input_list() will refer to a list of dicts,
        # where each dict has the form [workflow tree uri:input value]. 
        for i in range(len(b.input_list())):
            if self._running:
                input_dict = b.input_list()[i]
                for uri,val in input_dict.items():
                    self.set_op_input_at_uri(uri,val)
                # inputs are set, run in serial 
                thd = self.next_available_thread()
                self.write_log( 'BATCH EXECUTION {} / {} in thread {}'.format(i+1,len(b.input_list()),thd) )
                self.run_wf_serial(stk,thd)
                if any(b.saved_items()):
                    for uri in b.saved_items():
                        itm,idx = self.get_from_uri(uri)
                        b.output_list()[i].update({itm.tag():copy.deepcopy(itm.data)})
                else:
                    for lst in stk:
                        b.output_list()[i].update(self.op_items_to_dict(lst))
                self.update_op(b_itm.tag(),b)
            else:
                self.write_log( 'BATCH EXECUTION TERMINATED' )
        self.write_log( 'BATCH EXECUTION FINISHED' )

    def set_op_input_at_uri(self,uri,val):
        """Set an op input, indicated by uri, to provided value."""
        p = uri.split('.')
        # Allow some structure here: expect no meta-inputs. 
        op_itm, idx = self.get_from_uri(p[0])
        op = op_itm.data
        op.inputs[p[2]] = val

    def execution_stack(self):
        """
        Get a stack (list of lists) of Operations,
        such that each list contains a set of Operations whose dependencies are satisfied
        assuming all operations above them have been executed successfully.
        Give Batch and Realtime execution control Operations special treatment in the stack.
        """
        lst = []            # list of operations in order they are found to be ready
        stk = []            # stack, list of lists, of operations for flattening execution order
        b_rts = []          # list of uris of batch input_routes for batch items in stk
        valid_inputs = []   # list of uris of things available as inputs from stk
        nxt = True
        while nxt:
            nxt_itms = []
            for itm in self.root_items:
                op = itm.data
                op_rdy = True
                for name,il in op.input_locator.items(): 
                    inp_uri = itm.tag()+'.'+optools.inputs_tag+'.'+name
                    if il.src == optools.batch_input:
                        # check if the uri of this input is provided by any input_routes
                        if not inp_uri in b_rts:
                            op_rdy = False
                    elif il.src == optools.wf_input:
                        uri = il.val
                        for uri in optools.val_list(il):
                            f = uri.split('.')
                            # check if the uri of this input is one of the fields of a finished op
                            if not uri in valid_inputs and len(f) < 3:
                                op_rdy = False
                            elif len(f) >= 3:
                                # check if this is pointing to a meta-output.
                                # if so assume it will be generated during execution.
                                if not f[0]+'.'+f[1]+'.'+f[2] in valid_inputs:
                                    op_rdy = False
                            # but wait, also check if this op is a Batch or Realtime 
                            # that uses this uri in an input_route() or one of saved_items()
                            # in either case, it's ok to have this uri pointing down or upstream 
                            if isinstance(op,Realtime) or isinstance(op,Batch):
                                if uri in op.input_routes() or uri in op.saved_items():
                                    op_rdy = True
                if op_rdy:
                    if not itm in lst:
                        nxt_itms.append(itm)
            if not nxt_itms:
                nxt = False
            else:
                # make sure Batch or Realtime ops get special treatment in the stack
                b_rt_itms = [x for x in nxt_itms if isinstance(x.data,Realtime) or isinstance(x.data,Batch)]
                if any(b_rt_itms):
                    # add only one Batch or Realtime at its own level
                    # but make sure it is as low as possible in the stack
                    if len(b_rt_itms) == len(nxt_itms):
                        b_rts += b_rt_itms[0].data.input_routes()
                        nxt_itms = [b_rt_itms[0]]
                    else:
                        nxt_itms = [x for x in nxt_itms if not isinstance(x.data,Realtime) and not isinstance(x.data,Batch)]
                for nxt_itm in nxt_itms:
                    lst.append(nxt_itm)
                    # valid inputs: the operation, its input and output dicts, and their respective entries
                    valid_inputs += [nxt_itm.tag(),nxt_itm.tag()+'.'+optools.inputs_tag,nxt_itm.tag()+'.'+optools.outputs_tag]
                    valid_inputs += [nxt_itm.tag()+'.'+optools.outputs_tag+'.'+k for k in nxt_itm.data.outputs.keys()]
                    valid_inputs += [nxt_itm.tag()+'.'+optools.inputs_tag+'.'+k for k in nxt_itm.data.inputs.keys()]
                stk.append(nxt_itms)
        return stk

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


