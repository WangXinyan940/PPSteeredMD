try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
import sys
import os

HOME = os.path.dirname(os.path.abspath(__file__))


def optSideChain(args):
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

    print("Optimization...")
    ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                        "%s/prm/gbn2.xml" % HOME)
    rec = app.PDBFile(args.receptor)
    lig = app.PDBFile(args.ligand)
    tot = app.Modeller(rec.topology, rec.positions)
    tot.add(lig.topology, lig.positions)
    system = ff.createSystem(tot.getTopology(),
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=2.0 * unit.amu)
    rec_caidx = [i.index for i in rec.topology.atoms() if i.name == "CA"]
    lig_caidx = [
        i.index + rec.topology.getNumAtoms() for i in lig.topology.atoms()
        if i.name == "CA"
    ]
    tot_caidx = rec_caidx + lig_caidx
    rmsd_rec = mm.RMSDForce(tot.getPositions(), rec_caidx)
    rmsd_lig = mm.RMSDForce(tot.getPositions(), lig_caidx)
    rmsd_tot = mm.RMSDForce(tot.getPositions(), tot_caidx)
    cv = mm.CustomCVForce(
        "k^2 * ((rcv^2 + lcv^2) * alpha + 2 * tcv^2 * (1-alpha))")
    cv.addCollectiveVariable("rcv", rmsd_rec)
    cv.addCollectiveVariable("lcv", rmsd_lig)
    cv.addCollectiveVariable("tcv", rmsd_tot)
    cv.addGlobalParameter("k", 100.0)
    cv.addGlobalParameter("alpha", 0.0)
    system.addForce(cv)
    integ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                        5.0 / unit.picosecond,
                                        2.0 * unit.femtosecond)
    simulation = app.Simulation(tot.topology, system, integ, plat)
    simulation.reporters.append(
        app.StateDataReporter(sys.stdout,
                              1000,
                              step=True,
                              potentialEnergy=True,
                              temperature=True,
                              speed=True))
    simulation.reporters.append(app.DCDReporter(args.rlxtrj, 1000))
    simulation.context.setPositions(tot.positions)
    simulation.minimizeEnergy()
    for ii in np.linspace(0.0, 1.0, 20):
        simulation.context.setParameter("alpha", ii)
        simulation.step(args.nstep)
    state = simulation.context.getState(getPositions=True)
    minpos = state.getPositions(asNumpy=True)
    with open(args.output, "w") as f:
        app.PDBFile.writeFile(tot.topology, minpos, f)
    print("Done.")
