from slacxop import Operation


class Add(Operation):
    """Add two objects."""

    def __init__(self):
        input_names = ['augend', 'addend']
        output_names = ['sum']
        super(Add, self).__init__(input_names, output_names)
        self.input_doc['augend'] = 'array or number'
        self.input_doc['addend'] = 'array or number for which addition with augend is defined'
        self.output_doc['sum'] = 'augend plus addend'

    def run(self):
        self.outputs['sum'] = self.inputs['augend'] + self.inputs['addend']


class Multiply(Operation):
    """Multiply two objects."""

    def __init__(self):
        input_names = ['multiplicand', 'multiplier']
        output_names = ['product']
        super(Multiply, self).__init__(input_names, output_names)
        self.input_doc['multiplicand'] = 'array or number'
        self.input_doc['multiplier'] = 'array or number for which multiplication with multiplicand is defined'
        self.output_doc['product'] = 'multiplicand times multiplier'

    def run(self):
        self.outputs['product'] = self.inputs['multiplicand'] * self.inputs['multiplier']


class Subtract(Operation):
    """Subtract one object from another."""

    def __init__(self):
        input_names = ['minuend', 'subtrahend']
        output_names = ['difference']
        super(Subtract, self).__init__(input_names, output_names)
        self.input_doc['minuend'] = 'array or number'
        self.input_doc['subtrahend'] = 'array or number for which subtraction from minuend is defined'
        self.output_doc['difference'] = 'minuend minus subtrahend'


    def run(self):
        self.outputs['difference'] = self.inputs['minuend'] - self.inputs['subtrahend']


class Divide(Operation):
    """Divide two objects."""

    def __init__(self):
        input_names = ['dividend', 'divisor']
        output_names = ['quotient']
        super(Divide, self).__init__(input_names, output_names)
        self.input_doc['dividend'] = 'array or number'
        self.input_doc['divisor'] = 'array or number for which dividing dividend is defined'
        self.output_doc['quotient'] = 'dividend divided by divisor'

    def run(self):
        self.outputs['quotient'] = self.inputs['divident'] / self.inputs['divisor']


class Exponentiate(Operation):
    """Exponentiate an object by another object."""

    def __init__(self):
        input_names = ['base', 'exponent']
        output_names = ['power']
        super(Exponentiate, self).__init__(input_names, output_names)
        self.input_doc['base'] = 'array or number'
        self.input_doc['exponent'] = 'array or number for which exponentiating base is defined'
        self.output_doc['power'] = 'base raised by exponent'


    def run(self):
        self.outputs['power'] = self.inputs['base'] ** self.inputs['exponent']
