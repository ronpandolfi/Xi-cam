import base
import numpy as np

import sys
sys.path.append('/usr/local/lib/BornAgain-1.8')

import bornagain as ba
from bornagain import deg, angstrom, nm

phi_min, phi_max = -1.0, 1.0
alpha_min, alpha_max = 0.0, 2.0

####################
#Line Gratings
def get_sample(lattice_rotation_angle):
    """
    Returns a sample with a grating on a substrate,
    modelled by very long boxes forming a 1D lattice with Cauchy correlations.
    """
    # defining materials
    m_ambience = ba.HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate = ba.HomogeneousMaterial("Substrate", 6e-6, 2e-8)
    m_particle = ba.HomogeneousMaterial("Particle", 6e-4, 2e-8)

    box_length, box_width, box_height = 30 * nm, 10000000 * nm, 75 * nm
    lattice_length = 100 * nm

    # collection of particles
    interference = ba.InterferenceFunction1DLattice(
        lattice_length, lattice_rotation_angle)
    pdf = ba.FTDecayFunction1DCauchy(1000.0)
    interference.setDecayFunction(pdf)

    box_ff = ba.FormFactorBox(box_length, box_width, box_height)
    box = ba.Particle(m_particle, box_ff)

    particle_layout = ba.ParticleLayout()
    particle_layout.addParticle(
        box, 1.0, ba.kvector_t(0.0, 0.0, 0.0), ba.RotationZ(lattice_rotation_angle))
    particle_layout.setInterferenceFunction(interference)

    # assembling the sample
    air_layer = ba.Layer(m_ambience)
    air_layer.addLayout(particle_layout)
    substrate_layer = ba.Layer(m_substrate)

    multi_layer = ba.MultiLayer()
    multi_layer.addLayer(air_layer)
    multi_layer.addLayer(substrate_layer)
    return multi_layer

####################
#Multilayer with roughness
def create_bilayer():
    air_material = ba.HomogeneousMaterial("Air", 0.0, 0.0)          # Material with delta and beta
    substrate_material = ba.HomogeneousMaterial("Substrate", 7e-6, 1.8e-7)

    multilayer = ba.MultiLayer()
    multilayer.setCrossCorrLength(10)                               # Cross correlation on the multilayer
    layer_1 = ba.Layer(air_material)
    substrate = ba.Layer(substrate_material)

    roughness_1 = ba.LayerRoughness(1.0, 0.3, 5.0)                  # height, hurst, lateral correlation length

    multilayer.addLayer(layer_1)
    #multilayer.addLayer(substrate)                                 #without roughness
    multilayer.addLayerWithTopRoughness(substrate, roughness_1)     #with roughness

    return multilayer

####################
#Spherical paracrystal
def getSample():
    # Defining Materials
    material_2 = ba.HomogeneousMaterial("example02_Particle", 0.0006, 2e-08)
    material_3 = ba.HomogeneousMaterial("example02_Substrate", 5.99999999995e-06, 2e-08)
    material_1 = ba.HomogeneousMaterial("example02_Air", 0.0, 0.0)

    # Defining Layers
    layer_1 = ba.Layer(material_1)
    layer_2 = ba.Layer(material_3)

    # Defining Form Factors
    formFactor_1 = ba.FormFactorFullSphere(5.0 * nm)        #particle size

    # Defining Particles
    particle_1 = ba.Particle(material_2, formFactor_1)

    # Defining Interference Functions
    interference_1 = ba.InterferenceFunctionRadialParaCrystal(30.0 * nm, 10000.0 * nm)
    interference_1_pdf = ba.FTDistribution1DGauss(7.0)
    interference_1.setProbabilityDistribution(interference_1_pdf)

    # Defining Particle Layouts and adding Particles
    layout_1 = ba.ParticleLayout()
    layout_1.addParticle(particle_1, 1.0)
    layout_1.setInterferenceFunction(interference_1)
    layout_1.setTotalParticleSurfaceDensity(1)

    # Adding layouts to layers
    layer_1.addLayout(layout_1)

    # Defining Multilayers
    multiLayer_1 = ba.MultiLayer()
    multiLayer_1.addLayer(layer_1)
    multiLayer_1.addLayer(layer_2)
    return multiLayer_1

def getSimulation():
    simulation = ba.GISASSimulation()
    simulation.setDetectorParameters(100, -1.0 * deg, 1.0 * deg, 100, 0.0 * deg, 2.0 * deg)
    simulation.setBeamParameters(0.1 * nm, 0.2 * deg, 0.0 * deg)

    #simulation.setDetectorParameters(100, -1.0*deg, 1.0*deg, 100, 0.0*deg, 2.0*deg)     # n_phi (number of pixel), phi_f_min, phi_f_max, n_alpha, alpha_f_min, alpha_f_max :
    #simulation.setBeamParameters(0.1*nm, 0.2*deg, 0.0*deg)                              # wavelength, alpha_i, phi_i
    simulation.setBeamIntensity(1.0e+08)
    return simulation

def get_simulation(monte_carlo_integration=True):
    """
    Create and return GISAXS simulation with beam and detector defined
    """
    simulation = ba.GISASSimulation()
    simulation.setDetectorParameters(5, 0.055 * deg, 0.06 * deg, 100, 0.0 * deg, 1 * deg)
    #simulation.setDetectorParameters(100, phi_min*deg, phi_max*deg, 100, alpha_min*deg, alpha_max*deg)
    #simulation.setBeamParameters(1.0*angstrom, 0.2*deg, 0.0*deg)
    simulation.setBeamParameters(0.1 * nm, 0.4 * deg, 0.0 * deg)

    simulation.getOptions().setMonteCarloIntegration(True, 100)
    return simulation

def run_simulation():
    sample = getSample()
    simulation = getSimulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    img = simulation.getIntensityData()
    EZTest.setImage(np.rot90(np.log(img.getArray()+0.0001),1))
    return

def run_simulation1():
    sample = create_bilayer()
    simulation = getSimulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    img = simulation.getIntensityData()
    EZTest.setImage(np.rot90(img.getArray(),1))
    return

def run_simulation2():
    img_1 = np.zeros((100, 5))
    for i in range(0, 500, 1):
        angle = 85 + 0.02 * i
        sample = get_sample(lattice_rotation_angle = angle * deg)
        simulation = get_simulation(monte_carlo_integration=True)
        simulation.setSample(sample)
        simulation.setTerminalProgressMonitor()
        simulation.runSimulation()
        img = simulation.getIntensityData()
        img_1 += img.getArray()


    '''
    sample = get_sample(lattice_rotation_angle=90 * deg)
    simulation = get_simulation(monte_carlo_integration=True)
    simulation.setSample(sample)
    simulation.setTerminalProgressMonitor()
    simulation.runSimulation()
    #img = simulation.getIntensityData()
    EZTest.setImage(np.rot90(np.log(img.getArray()+0.0001),1))

    '''
    #img = simulation.getIntensityData()
    EZTest.setImage(np.rot90(np.log(img_1+0.0001),1))
    sum_1 = np.sum(img_1, axis = 1)
    EZTest.plot(sum_1)


    return

if __name__ == '__main__':
    result = run_simulation()
    ba.plot_intensity_data(result)

def opentest(filepaths):
    import fabio
    for filepath in filepaths:
        img = fabio.open(filepath).data
        img = np.rot90(np.log(img-np.min(img)+0.01))
        EZTest.setImage(img)


EZTest=base.EZplugin(name='BornAgain',toolbuttons=[('xicam/gui/icons_28.png',run_simulation), ('xicam/gui/icons_28.png',run_simulation1), ('xicam/gui/icons_28.png',run_simulation2)],
                     parameters=[{'name':'number_layer','value':2,'type':'int'},
                                {'name':'Amplitude','value':2200,'type':'int'},
                                 {'name':'Third', 'value': 600, 'type': 'int'},
                                 {'name':'Corelation', 'value': 98, 'type': 'int'},
                                 ],openfileshandler=opentest)

