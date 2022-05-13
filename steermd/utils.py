try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
import math


def readIndices(fname):
    with open(fname, "r") as f:
        text = f.readlines()
    text = [i.strip() for i in text if len(i.strip()) > 0]
    res = {}
    key = None
    for line in text:
        if "[" in line and "]" in line:
            key = line[1:-1].strip()
            res[key] = []
        else:
            for v in line.strip().split():
                res[key].append(int(v) - 1)
    return res


class SMDForceConstVelReporter(object):
    """
    Report the pulling force value to append to the simulation reporter.
    The calculation bases on the analytical form of the potential. Get the positions,
    params from simulation, we calculate the force after certain interval.

    Mandatory to have 'describeNextReport' and 'report' functions.

    """
    def __init__(self,
                 file,
                 reportInterval,
                 ref_indices=[],
                 ref_weights=[],
                 pull_indices=[],
                 pull_weights=[],
                 force_constant=0):
        """
        pull_atom_indices = [ref, pulled]
        """
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._force_constant = force_constant
        self._ref_indices = np.array(ref_indices, dtype=int)
        self._pull_indices = np.array(pull_indices, dtype=int)
        self._ref_weights = np.array(ref_weights, dtype=float)
        self._pull_weights = np.array(pull_weights, dtype=float)

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        """
        Tell the program what values need to be returned.
        Coordinates, velocities, forces, energies;

        :param simulation: Simulation object
        :return:
            steps: how many steps away to report;
            True/False: whether to pass coor to State object;
            True/False: whether to pass vel to State object;
            True/False: whether to pass force to State object;
            True/False: whether to pass energy to State object;
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False)

    def report(self, simulation, state):
        """
        Collect params, calculate and report force to output file.

        :param simulation: Simulation object;
        :param state: State object with info specifief by describeNextReport();
        :return:

        """
        time = state.getTime().value_in_unit(unit.picosecond)

        # For CustomCentroidBondForce class
        atom_coords = state.getPositions(
            asNumpy=True)[self._pull_indices].value_in_unit(unit.nanometer)
        x = np.sum(atom_coords[:, 0] * self._pull_weights /
                   self._pull_weights.sum())
        y = np.sum(atom_coords[:, 1] * self._pull_weights /
                   self._pull_weights.sum())
        z = np.sum(atom_coords[:, 2] * self._pull_weights /
                   self._pull_weights.sum())
        atom_coords = state.getPositions(
            asNumpy=True)[self._ref_indices].value_in_unit(unit.nanometer)
        x0 = np.sum(atom_coords[:, 0] * self._ref_weights /
                    self._ref_weights.sum())
        y0 = np.sum(atom_coords[:, 1] * self._ref_weights /
                    self._ref_weights.sum())
        z0 = np.sum(atom_coords[:, 2] * self._ref_weights /
                    self._ref_weights.sum())
        displaceVar = state.getParameters()['displaceVar']

        vec = np.array([x - x0, y - y0, z - z0])
        dist = np.linalg.norm(vec)
        norm = vec / dist

        force = self._force_constant * (displaceVar - dist)
        self._out.write('%g %g %g %g %g %g %g %g %g \n' %
                        (time, x0, y0, z0, x, y, z, displaceVar, force))
        self._out.flush()
        return 0


def steeredBiasConstVel(ref_idx, pull_idx, ref_weights, pull_weights, fc,
                        displaceVar):
    # g1: ref  g2: pull
    SMD_bond = mm.CustomCentroidBondForce(
        2, '0.5*fc_smd*(distance(g1,g2)-displaceVar)^2')
    SMD_bond.addGlobalParameter('displaceVar', displaceVar)
    SMD_bond.addPerBondParameter('fc_smd')
    # Read two atom group lists
    SMD_bond.addGroup(ref_idx, ref_weights)
    SMD_bond.addGroup(pull_idx, pull_weights)
    SMD_bond.addBond([0, 1], [fc])
    return SMD_bond


class SITSLangevinIntegrator(mm.CustomIntegrator):
    def __init__(self, Tlist, logNlist, friction, dt):
        super(SITSLangevinIntegrator, self).__init__(dt)

        self.logNlist = logNlist
        self.Tlist = Tlist

        Aup, Adown = [], []
        for nstate in range(Tlist.shape[0]):
            kTi = 8.314 / 1000.0 * Tlist[nstate]
            logN = logNlist[nstate]
            Aup.append(f"exp({logN:.32f} - energy1 / {kTi:.32f}) / {kTi:.32f}")
            Adown.append(f"exp({logN:.32f} - energy1 / {kTi:.32f})")
        AupT = " + ".join(Aup)
        AdownT = " + ".join(Adown)

        temperature = trep[0]
        kB = 8.314 / 1000.0
        kT = kB * temperature
        ft = friction * dt * 0.001  # friction in 1/ps, dt in fs

        self.addGlobalVariable("a", np.exp(-ft))
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * ft)))
        self.addGlobalVariable("kT", kT)
        self.addPerDofVariable("x1", 0)
        self.addPerDofVariable("fadd", 0)

        # create state K
        self.addGlobalVariable("one_A", 0.0)
        self.addGlobalVariable("Aup", 0.0)
        self.addGlobalVariable("Adown", 0.0)
        #integrator.addPerDofVariable("feff", 0.0)
        self.addUpdateContextState()
        # compute Astate
        self.addComputeGlobal("Aup", "0.0")
        self.addComputeGlobal("Adown", "0.0")
        self.addComputeGlobal("Aup", AupT)
        self.addComputeGlobal("Adown", AdownT)
        self.addComputeGlobal("one_A", "1 - Aup / Adown * kT")
        #integrator.addComputePerDof("fadd", "fadd * (1 - Aup / Adown * kT)")
        self.addComputePerDof("v", "v + dt*f/m")
        self.addComputePerDof("v", "v - dt*f1*one_A/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

    @classmethod
    def genLogNList(cls, Tlist, Eref=0.0):
        blist = 1. / 8.314 / Tlist * 1000.0
        return blist * Eref

    @classmethod
    def genPk(cls, Tlist, logNlist, energylist):
        betarep = (1. / 8.314 / Tlist * 1000.0).reshape((-1, 1))
        ener = energylist.reshape((1, -1))
        logN = logNlist.reshape((-1, 1))
        Pall = np.exp(logN - betarep * ener)
        Pk = Pall / Pall.sum(axis=0).reshape((1, -1))
        return Pk


class SelectEnergyReporter:
    def __init__(self, file: str, reportInterval: int, logNlist):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.logNlist = logNlist
        self._out.write("# logN")
        for n, logn in enumerate(logNlist):
            if n % 10 == 0:
                self._out.write("\n# ")
            self._out.write(f"{logn:15.8f} ")
        self._out.write("\n")

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        state2 = simulation.context.getState(getEnergy=True, groups={1})
        ener = state2.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        self._out.write(f"{ener:16.8f}\n")
        self._out.flush()