try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
from steermd.utils import SITSLangevinIntegrator, SelectEnergyReporter, isDihBackbone, isDihSidechain
from mdtraj.reporters import DCDReporter
import mdtraj as md
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

    if not args.rerun:
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
                                     constraints=app.HBonds,
                                     hydrogenMass=2.0 * unit.amu)
        else:
            system = ff.createSystem(pdb.topology,
                                     nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.6 * unit.nanometer,
                                     constraints=app.HBonds,
                                     hydrogenMass=2.0 * unit.amu)

        integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                                 5.0 / unit.picosecond,
                                                 args.delta * unit.femtosecond)
        simulation = app.Simulation(pdb.topology, system, integrator, plat)
        simulation.context.setPositions(pdb.getPositions())
        simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
        simulation.minimizeEnergy()

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
            DCDReporter(args.traj,
                        int(10.0 * 1000 / args.delta),
                        atomSubset=idx))
        simulation.step(nstep)

    traj = md.load(args.traj, top=args.input)
    ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                        "amber14/tip3p.xml")
    pdb = app.PDBFile(args.input)
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer,
                             constraints=app.AllBonds,
                             hydrogenMass=2.0 * unit.amu)
    integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                             5.0 / unit.picosecond,
                                             args.delta * unit.femtosecond)
    ctx = mm.Context(system, integrator, plat)
    with open(args.output, "w") as f:
        for frame in tqdm(traj):
            ctx.setPositions(frame.xyz[0, :, :] * unit.nanometer)
            ener = ctx.getState(
                getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole)
            f.write(f"{ener:16.8f}\n")