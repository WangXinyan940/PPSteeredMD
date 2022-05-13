try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
from mdtraj.reporters import DCDReporter


class SelectEnergyReporter:
    def __init__(self, file: str, reportInterval: int):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

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


def regularMD(args):
    platNames = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    if "CUDA" in platNames:
        plat = mm.Platform.getPlatformByName("CUDA")
    #elif "OpenCL" in platNames:
    #    plat = mm.Platform.getPlatformByName("OpenCL")
    else:
        plat = mm.Platform.getPlatformByName("CPU")

    print("Build...")

    if args.water:
        ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                            "amber14/tip3p_standard.xml")
        pdbinit = app.PDBFile(args.input)
        pdb = app.Modeller(pdbinit.topology, pdbinit.positions)
        pdb.addSolvent(ff, padding=1.0 * unit.nanometer)
        idx = [a.index for a in pdbinit.topology.atoms()]
    else:
        ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                            "%s/prm/gbn2.xml" % HOME)
        pdb = app.PDBFile(args.input)
        idx = [a.index for a in pdb.topology.atoms()]
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    if args.water:
        system = ff.createSystem(pdb.topology,
                                 nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.0 * unit.nanometer,
                                 constraints=app.AllBonds,
                                 hydrogenMass=2.0 * unit.amu)
    else:
        system = ff.createSystem(pdb.topology,
                                 nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.6 * unit.nanometer,
                                 constraints=app.AllBonds,
                                 hydrogenMass=2.0 * unit.amu)

    # indices = readIndices(args.index)
    # rec_idx = indices["receptor"]
    # lig_idx = indices["ligand"]
    # atoms = [a for a in pdb.topology.atoms()]
    # rec_weights = np.array(
    #     [atoms[i].element.mass.value_in_unit(unit.amu) for i in rec_idx])
    # lig_weights = np.array(
    #     [atoms[i].element.mass.value_in_unit(unit.amu) for i in lig_idx])

    if args.sits:
        for force in system.getForces():
            if isinstance(force, mm.PeriodicTorsionForce):
                force.setForceGroup(1)

        trep = np.linspace(300., 600., 101)
        t = trep - trep.min()
        t = t / t.std()
        nk = np.exp(-t)

        Aup, Adown = [], []
        for nstate in range(trep.shape[0]):
            ni = nk[nstate]
            kTi = 8.314 / 1000.0 * trep[nstate]
            Aup.append(f"{ni:.32f} * exp(- energy / {kTi:.32f}) / {kTi:.32f}")
            Adown.append(f"{ni:.32f} * exp(- energy / {kTi:.32f})")
        AupT = " + ".join(Aup)
        AdownT = " + ".join(Adown)

        temperature = trep[0]
        friction = 5.0
        dt = args.delta * 0.001
        kB = 8.314 / 1000.0
        kT = kB * temperature

        integrator = mm.CustomIntegrator(dt)
        integrator.addGlobalVariable("a", np.exp(-friction * dt))
        integrator.addGlobalVariable("b",
                                     np.sqrt(1 - np.exp(-2 * friction * dt)))
        integrator.addGlobalVariable("kT", kT)
        integrator.addPerDofVariable("x1", 0)

        # create state K
        integrator.addGlobalVariable("Aup", 0.0)
        integrator.addGlobalVariable("Adown", 0.0)
        integrator.addPerDofVariable("feff", 0.0)
        integrator.addUpdateContextState()
        # compute Astate
        integrator.addComputeGlobal("Aup", "0.0")
        integrator.addComputeGlobal("Adown", "0.0")
        integrator.addComputeGlobal("Aup", AupT)
        integrator.addComputeGlobal("Adown", AdownT)
        integrator.addComputePerDof("feff", "f * Aup / Adown * kT")
        integrator.addComputePerDof("v", "v + dt*feff/m")
        integrator.addConstrainVelocities()
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        integrator.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "v + (x-x1)/dt")

    else:
        integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                                 5.0 / unit.picosecond,
                                                 args.delta * unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integrator, plat)

    simulation.context.setPositions(pdb.getPositions())

    nstep = int(args.length * 1000 / args.delta)

    simulation.reporters.append(
        app.StateDataReporter(sys.stdout,
                              int(10.0 * 1000 / args.delta),
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              temperature=True,
                              progress=True,
                              remainingTime=True,
                              speed=True,
                              totalSteps=nstep))
    simulation.reporters.append(
        DCDReporter(args.traj, int(10.0 * 1000 / args.delta), atomSubset=idx))
    if args.sits:
        simulation.reporters.append(
            SelectEnergyReporter(args.sits_out, int(10.0 * 1000 / args.delta)))

    simulation.minimizeEnergy()
    simulation.step(nstep)