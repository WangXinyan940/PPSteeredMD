try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np
from steermd.utils import SMDForceConstVelReporter, steeredBiasConstVel, readIndices
import sys
import os

HOME = os.path.dirname(os.path.abspath(__file__))


def steerMDConstVel(args):
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

    print("Steer...")
    ff = app.ForceField("%s/prm/amberff14SBonlysc.xml" % HOME,
                        "%s/prm/gbn2.xml" % HOME)
    pdb = app.PDBFile(args.topol)
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=2.0 * unit.amu)

    indices = readIndices(args.index)
    rec_idx = indices["receptor"]
    lig_idx = indices["ligand"]
    atoms = [a for a in pdb.topology.atoms()]
    rec_weights = np.array(
        [atoms[i].element.mass.value_in_unit(unit.amu) for i in rec_idx])
    lig_weights = np.array(
        [atoms[i].element.mass.value_in_unit(unit.amu) for i in lig_idx])

    refx = (pos[rec_idx][:, 0] * rec_weights / rec_weights.sum()).sum()
    refy = (pos[rec_idx][:, 1] * rec_weights / rec_weights.sum()).sum()
    refz = (pos[rec_idx][:, 2] * rec_weights / rec_weights.sum()).sum()

    pullx = (pos[lig_idx][:, 0] * lig_weights / lig_weights.sum()).sum()
    pully = (pos[lig_idx][:, 1] * lig_weights / lig_weights.sum()).sum()
    pullz = (pos[lig_idx][:, 2] * lig_weights / lig_weights.sum()).sum()

    displacex = pullx - refx
    displacey = pully - refy
    displacez = pullz - refz
    displace = np.sqrt(displacex**2 + displacey**2 + displacez**2)

    pvec = np.array([displacex, displacey, displacez])
    pvec = pvec / np.linalg.norm(pvec)

    bias = steeredBiasConstVel(rec_idx, lig_idx, rec_weights, lig_weights,
                               args.fconst, displace)
    system.addForce(bias)

    integ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                        5.0 / unit.picosecond,
                                        2.0 * unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integ, plat)
    with open(args.input, "r") as f:
        state = mm.XmlSerializer.deserialize(f.read())
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())

    nstep = int(args.length * 500)

    simulation.reporters.append(
        app.StateDataReporter(sys.stdout,
                              10000,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              temperature=True,
                              progress=True,
                              remainingTime=True,
                              speed=True,
                              totalSteps=nstep))
    simulation.reporters.append(app.DCDReporter(args.traj, args.nprint))
    simulation.reporters.append(
        SMDForceConstVelReporter(args.output,
                                 args.nprint,
                                 ref_indices=rec_idx,
                                 ref_weights=rec_weights,
                                 pull_indices=lig_idx,
                                 pull_weights=lig_weights,
                                 force_constant=args.fconst))
    for nround in range(int(nstep/10)):
        simulation.step(10)
        displace += 10 * 0.002 * args.velocity
        simulation.context.setParameter('displaceVar', displace)