try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np


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
    ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                        "%s/prm/gbn2.xml" % HOME)
    pdb = app.PDBFile(args.input)
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

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

    integ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                        5.0 / unit.picosecond,
                                        args.delta * unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integ, plat)

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

    simulation.reporters.append(app.DCDReporter(args.traj, int(10.0 * 1000 / args.delta),))
    simulation.minimizeEnergy()
    simulation.step(nstep)