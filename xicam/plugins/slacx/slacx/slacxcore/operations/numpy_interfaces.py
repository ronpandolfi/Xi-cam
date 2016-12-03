import numpy as np

from slacxop import Operation
import optools


class Any(Operation):
    """Check whether an array has any non-zero / True elements."""

    def __init__(self):
        input_names = ['ndarray']
        output_names = ['any']
        super(Any, self).__init__(input_names, output_names)
        self.input_doc['ndarray'] = 'ndarray of any type or shape'
        self.output_doc['any'] = 'boolean; existence of at least one nonzero element in *ndarray*'
        # source & type
        self.input_src['ndarray'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['any'] = np.any(self.inputs['ndarray'])

class All(Operation):
    """Check whether an array is entirely non-zero / True elements."""

    def __init__(self):
        input_names = ['ndarray']
        output_names = ['all']
        super(All, self).__init__(input_names, output_names)
        self.input_doc['ndarray'] = 'ndarray of any type or shape'
        self.output_doc['all'] = 'boolean; nonzero existence of all elements in *ndarray*'
        # source & type
        self.input_src['ndarray'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['all'] = np.all(self.inputs['ndarray'])

class Not(Operation):
    """Identify zero-value / False elements."""

    def __init__(self):
        input_names = ['ndarray']
        output_names = ['not_ndarray']
        super(Not, self).__init__(input_names, output_names)
        self.input_doc['ndarray'] = 'ndarray of any type or shape'
        self.output_doc['not_ndarray'] = 'boolean array of same shape as *ndarray*'
        # source & type
        self.input_src['ndarray'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['all'] = np.logical_not(self.inputs['ndarray'])

class And(Operation):
    """Check whether an array is entirely non-zero / True elements."""

    def __init__(self):
        input_names = ['ndarray1', 'ndarray2']
        output_names = ['ndarray1_and_2']
        super(And, self).__init__(input_names, output_names)
        self.input_doc['ndarray1'] = 'ndarray of any type or shape'
        self.input_doc['ndarray2'] = 'ndarray that can be compared to ndarray1'
        self.output_doc['ndarray1_and_2'] = 'boolean array, True where both inputs exist'
        # source & type
        self.input_src['ndarray1'] = optools.wf_input
        self.input_src['ndarray2'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['all'] = np.all(self.inputs['ndarray'])

class Or(Operation):
    """Check whether an array is entirely non-zero / True elements."""

    def __init__(self):
        input_names = ['ndarray1', 'ndarray2']
        output_names = ['ndarray1_or_2']
        super(Or, self).__init__(input_names, output_names)
        self.input_doc['ndarray1'] = 'ndarray of any type or shape'
        self.input_doc['ndarray2'] = 'ndarray that can be compared to ndarray1'
        self.output_doc['ndarray1_or_2'] = 'boolean array, True where at least one input exists'
        # source & type
        self.input_src['ndarray1'] = optools.wf_input
        self.input_src['ndarray2'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['all'] = np.all(self.inputs['ndarray'])

class AnyNaN(Operation):
    """Check whether an array has any NaN elements."""

    def __init__(self):
        input_names = ['ndarray']
        output_names = ['any_nan']
        super(AnyNaN, self).__init__(input_names, output_names)
        self.input_doc['ndarray'] = 'ndarray of any type or shape'
        self.output_doc['any_nan'] = 'existence of any NaN elements'
        # source & type
        self.input_src['ndarray'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['any_nan'] = np.isnan(self.inputs['ndarray']).any()

class IsNaN(Operation):
    """Return boolean array marking NaN elements."""

    def __init__(self):
        input_names = ['ndarray']
        output_names = ['nan']
        super(IsNaN, self).__init__(input_names, output_names)
        self.input_doc['ndarray'] = 'ndarray of any type or shape'
        self.output_doc['nan'] = 'existence of any non-zero / True elements'
        # source & type
        self.input_src['ndarray'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['nan'] = np.isnan(self.inputs['ndarray'])

class AnyZero(Operation):
    """Return boolean array marking zero-value elements."""

    def __init__(self):
        input_names = ['ndarray']
        output_names = ['any_zeros']
        super(AnyZero, self).__init__(input_names, output_names)
        self.input_doc['ndarray'] = 'ndarray of any type or shape'
        self.output_doc['any_zeros'] = 'existence of any zero / False elements'
        # source & type
        self.input_src['ndarray'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['any_zeros'] = np.any(np.logical_not(self.inputs['ndarray']))

class Zip(Operation):
    """Zips two 1d ndarrays together for display in slacx."""

    def __init__(self):
        input_names = ['ndarray_x', 'ndarray_y']
        output_names = ['ndarray_xy']
        super(Zip, self).__init__(input_names, output_names)
        self.input_doc['ndarray_x'] = '1d ndarray, x axis'
        self.input_doc['ndarray_y'] = '1d ndarray, y axis; same size as ndarray_x'
        self.output_doc['ndarray_xy'] = 'n x 2 ndarray for slacx autodisplay fun'
        # source & type
        self.input_src['ndarray_x'] = optools.wf_input
        self.input_src['ndarray_y'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION','DISPLAY']

    def run(self):
        x = self.inputs['ndarray_x']
        y = self.inputs['ndarray_y']
        n = x.size
        if len(x.shape) > 1:
            raise ValueError("ndarray_x and ndarray_y must be 1d arrays")
        if (x.shape != y.shape):  ###
            raise ValueError("ndarray_x and ndarray_y must have the same shape")
        xy = np.zeros((n,2))
        xy[:,0] = x
        xy[:,1] = y
        self.outputs['ndarray_xy'] = xy

class LogLogZip(Operation):
    """Takes the logarithm of two 1d ndarrays, then zips them together for display in slacx.

    Logarithm is taken in base ten.

    Any elements with non-positive values are removed, so this operation
    may not be appropriate for computational purposes."""


    def __init__(self):
        input_names = ['ndarray_x', 'ndarray_y']
        output_names = ['ndarray_logxlogy']
        super(LogLogZip, self).__init__(input_names, output_names)
        self.input_doc['ndarray_x'] = '1d ndarray, x axis'
        self.input_doc['ndarray_y'] = '1d ndarray, y axis; same size as ndarray_x'
        self.output_doc['ndarray_logxlogy'] = 'n x 2 ndarray for slacx autodisplay fun'
        # source & type
        self.input_src['ndarray_x'] = optools.wf_input
        self.input_src['ndarray_y'] = optools.wf_input
        self.categories = ['MISC.NDARRAY MANIPULATION','DISPLAY']

    def run(self):
        x = self.inputs['ndarray_x']
        y = self.inputs['ndarray_y']
        if len(x.shape) > 1:
            print "ndarray_x and ndarray_y must be 1d arrays"
            raise ValueError("ndarray_x and ndarray_y must be 1d arrays")
        if (x.shape != y.shape):  ###
            print "ndarray_x and ndarray_y must have the same shape"
            raise ValueError("ndarray_x and ndarray_y must have the same shape")
        good_vals = ((x > 0) & (y > 0) & (~np.isnan(x)) & (~np.isnan(y)))
        n = good_vals.sum()
        xy = np.zeros((n,2))
        xy[:,0] = np.log10(x[good_vals])
        xy[:,1] = np.log10(y[good_vals])
        self.outputs['ndarray_logxlogy'] = xy

