try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
import argparse
import sys
import os


HOME = os.path.dirname(os.path.abspath(__file__))


def genIndices(args):
    rec = app.PDBFile(args.receptor)
    lig = app.PDBFile(args.ligand)
    tot = app.Modeller(rec.topology, rec.positions)
    tot.add(lig.topology, lig.positions)
    rec_idx = [i.index for i in rec.topology.atoms()]
    lig_idx = [
        i.index + rec.topology.getNumAtoms() for i in lig.topology.atoms()
    ]
    with open(args.output, "w") as f:
        f.write("[ receptor ]\n")
        for n, i in enumerate(rec_idx):
            f.write("%i " % (i+1))
            if n > 0 and n % 20 == 0:
                f.write("\n")
        f.write("\n")
        f.write("[ ligand ]\n")
        for n, i in enumerate(lig_idx):
            f.write("%i " % (i+1))
            if n > 0 and n % 20 == 0:
                f.write("\n")


def genStates(args):
    platNames = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    if "CUDA" in platNames:
        plat = mm.Platform.getPlatformByName("CUDA")
    elif "OpenCL" in platNames:
        plat = mm.Platform.getPlatformByName("OpenCL")
    else:
        plat = mm.Platform.getPlatformByName("CPU")

    pdb = app.PDBFile(args.input)
    ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                        "%s/prm/gbn2.xml" % HOME)
    system = ff.createSystem(pdb.getTopology(),
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=2.0 * unit.amu)
    caidx = [i.index for i in pdb.topology.atoms() if i.name == "CA"]
    rmsd = mm.RMSDForce(pdb.getPositions(), caidx)
    cv = mm.CustomCVForce("k^2 * cv^2")
    cv.addCollectiveVariable("cv", rmsd)
    cv.addGlobalParameter("k", 100.0)
    system.addForce(cv)
    for force in system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            force.setForceGroup(1)
    system.addForce(
        mm.AndersenThermostat(300.0 * unit.kelvin, 5.0 / unit.picosecond))
    integ = mm.AMDForceGroupIntegrator(2.0 * unit.femtosecond, 1, args.alpha,
                                       args.ecut * unit.kilojoule_per_mole)
    simulation = app.Simulation(pdb.topology, system, integ, plat)
    simulation.reporters.append(
        app.StateDataReporter(sys.stdout,
                              1000,
                              step=True,
                              potentialEnergy=True,
                              temperature=True,
                              speed=True,
                              remainingTime=True,
                              totalSteps=args.nstate * args.nstep))
    simulation.reporters.append(app.DCDReporter(args.traj, 1000))
    simulation.context.setPositions(pdb.positions)
    for ns in range(1, args.nstate + 1):
        simulation.step(args.nstep)
        state = simulation.context.getState(getPositions=True,
                                            getVelocities=True)
        with open("%s-%i.rst" % (args.pname, ns), "w") as f:
            f.write(mm.XmlSerializer.serialize(state))