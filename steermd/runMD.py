try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
from steermd.utils import SITSLangevinIntegrator, SelectEnergyReporter
from mdtraj.reporters import DCDReporter
import sys
import os

HOME = os.path.dirname(os.path.abspath(__file__))


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
                            "amber14/tip3p.xml")
        pdbinit = app.PDBFile(args.input)
        pdb = app.Modeller(pdbinit.topology, pdbinit.positions)
        pdb.addSolvent(ff, padding=1.0 * unit.nanometer)
        idx = [a.index for a in pdbinit.topology.atoms()]
    else:
        ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                            "%s/prm/gbn2.xml" % HOME)
        pdb = app.PDBFile(args.input)
        idx = [a.index for a in pdb.topology.atoms()]

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

        # equilibrate system
        print("Start EQ...")
        integEQ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                              5.0 / unit.picosecond,
                                              args.delta * unit.femtosecond)
        simEQ = app.Simulation(pdb.topology, system, integEQ, plat)
        simEQ.context.setPositions(pdb.getPositions())
        nstep = int(1.0 * 1000 * 1000 / args.delta)
        simEQ.reporters.append(
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
        simEQ.step(nstep)
        state = simEQ.context.getState(getPositions=True,
                                       getVelocities=True,
                                       getEnergy=True,
                                       groups={1})
        Eref = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

        # run MD
        Tlist = np.linspace(300., 600., 91)
        logNlist = SITSLangevinIntegrator.genLogNList(Tlist, Eref=Eref)
        integrator = SITSLangevinIntegrator(Tlist, logNlist, 5.0, args.delta)
        simulation = app.Simulation(pdb.topology, system, integrator, plat)
        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocities(state.getVelocities())
    else:
        integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                                 5.0 / unit.picosecond,
                                                 args.delta * unit.femtosecond)
        simulation = app.Simulation(pdb.topology, system, integrator, plat)
        simulation.context.setPositions(pdb.getPositions())
        simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

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
            SelectEnergyReporter(args.sits_out, int(10.0 * 1000 / args.delta),
                                 integrator.logNlist))

    simulation.minimizeEnergy()
    simulation.step(nstep)