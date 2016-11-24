import numpy as np

from slacxop import Operation
from derivatives import discrete_second_derivative


class Guess(Operation):
    """Guess size and polydispersity of a spherical nanoparticle from its SAXS pattern."""

    def __init__(self):
        input_names = ['x', 'y', ]
        output_names = ['sum']
        super(Guess, self).__init__(input_names, output_names)
        self.input_doc['augend'] = 'array or number'
        self.input_doc['addend'] = 'array or number for which addition with augend is defined'
        self.output_doc['sum'] = 'augend plus addend'
        self.categories = ['ARITHMETIC']

    def run(self):
        self.outputs['sum'] = self.inputs['augend'] + self.inputs['addend']


def guess(q, I):
    smoothedI = smooth(q, I)
    curv = discrete_second_derivative(q, smoothedI)

    dip_array = find_dips(q, smoothedI)



    first_dip_q = q[dip_array][0]
    first_dip_model_intensity = power_law_model(first_dip_q)
    first_dip_intensity = smoothedI[dip_array][0]
    first_dip_curvature = curv[dip_array][0]
    polydispersity = polydispersity_from_first_peak(first_dip_intensity, first_dip_curvature)
    q0 = q0_from(polydispersity, first_dip_q)
    r0 = r0_from_q0(q0)