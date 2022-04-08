try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
from steermd.utils import SMDForceConstForceReporter, steeredBiasConstForce, steeredConsConstForce
import sys
import os

HOME = os.path.dirname(os.path.abspath(__file__))


def steerMDConstForce(args):
    pass