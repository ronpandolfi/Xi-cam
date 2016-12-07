import numpy as np
import time

from slacxop import Operation
import optools

class ItemFromSequence(Operation):
    """Extract an item from a sequence.

    Intended for use with strings, lists, and tuples, but may work with other data types.

    Uses the python convention that indexing begins with zero, not one."""

    def __init__(self):
        input_names = ['sequence', 'index']
        output_names = ['item']
        super(ItemFromSequence, self).__init__(input_names, output_names)
        self.input_doc['sequence'] = 'list, tuple, string, or other indexable sequence'
        self.input_doc['index'] = 'index of the item you wish to extract'
        self.output_doc['item'] = 'item extracted from *sequence* at position *index*'
        self.categories = ['MISC.PYTHON OBJECT MANIPULATION']
        # source & type
        self.input_src['sequence'] = optools.wf_input
        self.input_src['index'] = optools.user_input
        self.input_type['index'] = optools.int_type

    def run(self):
        self.outputs['item'] = self.inputs['sequence'][self.inputs['index']]


class ItemRangeFromSequence(Operation):
    """Extract consecutive items from a sequence.

    Intended for use with strings, lists, and tuples, but may work with other data types.

    Uses python indexing conventions.
    """

    def __init__(self):
        input_names = ['sequence', 'start_index', 'end_index']
        output_names = ['slice']
        super(ItemRangeFromSequence, self).__init__(input_names, output_names)
        self.input_doc['sequence'] = 'list, tuple, string, or other indexable sequence'
        self.input_doc['start_index'] = 'starting index of the slice you wish to extract'
        self.input_doc['end_index'] = 'ending index of the slice you wish to extract'
        self.output_doc['slice'] = 'items extracted from *sequence*'
        self.categories = ['MISC.PYTHON OBJECT MANIPULATION']

    def run(self):
        start_index, end_index = type_check_item_range_from_sequence(self.inputs['start_index'], self.inputs['end_index'])
        print self.inputs['sequence'], start_index, end_index
        self.outputs['slice'] = self.inputs['sequence'][start_index : end_index]


def type_check_item_range_from_sequence(start_index, end_index):
    if start_index != None:
        start_index = int(start_index)
    if end_index != None:
        end_index = int(end_index)
    return start_index, end_index


class ItemFromMap(Operation):
    """Extract an item from a key-value map.

    Intended for use with dictionaries, but may work with other data types."""

    def __init__(self):
        input_names = ['map', 'key']
        output_names = ['value']
        super(ItemFromMap, self).__init__(input_names, output_names)
        self.input_doc['map'] = 'dictionary or other map'
        self.input_doc['key'] = 'key to the item you wish to extract'
        self.output_doc['value'] = 'value of the item with key *key* in object *map*'
        self.categories = ['MISC.PYTHON OBJECT MANIPULATION']

    def run(self):
        self.outputs['value'] = self.inputs['map'][ self.inputs['key'] ]


class ItemToMap(Operation):
    """Add an item to a key-value map.

    Works IN PLACE, meaning that later functions calling on *map* may find the key-value pair, even if they do not
    explicitly reference *new_map*.  Reference *new_map* to be absolutely certain that the operation has been
    performed.

    Intended for use with dictionaries, but may work with other data types."""

    def __init__(self):
        input_names = ['map', 'key', 'value']
        output_names = ['new_map']
        super(ItemToMap, self).__init__(input_names, output_names)
        self.input_doc['map'] = 'dictionary or other map'
        self.input_doc['key'] = 'key to the item you wish to add'
        self.input_doc['value'] = 'value of the item with key *key* in object *map*'
        self.output_doc['new_map'] = 'map with added key-value pair'
        # source & type
        self.input_src['map'] = optools.wf_input
        self.input_src['key'] = optools.user_input
        self.input_src['value'] = optools.wf_input
        self.input_type['key'] = optools.str_type
        self.categories = ['MISC.PYTHON OBJECT MANIPULATION']

    def run(self):
        self.inputs['map'][ self.inputs['key'] ] = self.inputs['value']
        self.outputs['new_map'] = self.inputs['map']


class PrintMessage(Operation):
    """Prints a message to the console.

    Potentially useful for debug.
    """

    def __init__(self):
        input_names = ['message']
        output_names = []
        super(PrintMessage, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['message'] = 'an object with a print method that will be printed to the console'
        # source and type
        self.input_src['message'] = optools.wf_input
        self.categories = ['MISC.PYTHON OBJECT MANIPULATION','OUTPUT']

    def run(self):
        print self.inputs['message']


class PrintMessageAndTime(Operation):
    """Prints the time and a message to the console.

    Potentially useful for debug.
    """

    def __init__(self):
        input_names = ['message']
        output_names = ['message']
        super(PrintMessageAndTime, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['message'] = 'an object with a print method that will be printed to the console'
        # source and type
        self.input_src['message'] = optools.wf_input
        self.categories = ['MISC.PYTHON OBJECT MANIPULATION','OUTPUT']

    def run(self):
        print time.asctime(), self.inputs['message']
        self.outputs['message'] = self.inputs['message']


class DummySequences(Operation):
    def __init__(self):
        input_names = []
        output_names = ['list', 'tuple', 'string']
        super(DummySequences, self).__init__(input_names, output_names)
        self.output_doc['list'] = ''
        self.output_doc['tuple'] = ''
        self.output_doc['string'] = ''
        self.categories = ['TESTS.OBJECT GENERATION']

    def run(self):
        self.outputs['list'] = ['ab','ra','ca','dab','ra']
        self.outputs['tuple'] = ('ab','ra','ca','dab','ra')
        self.outputs['string'] = 'abracadabra'

