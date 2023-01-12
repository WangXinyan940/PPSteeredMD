import os
import numpy as np
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
from steermd.utils import readIndices
from openmmplumed import PlumedForce
import sys

HOME = os.path.dirname(os.path.abspath(__file__))


def runSteerMDPlumed(args):
    platNames = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    if "CUDA" in platNames:
        plat = mm.Platform.getPlatformByName("CUDA")
    # elif "OpenCL" in platNames:
    #    plat = mm.Platform.getPlatformByName("OpenCL")
    else:
        plat = mm.Platform.getPlatformByName("CPU")

    nstep = int(args.length * 1000 / args.delta)

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
    displace_fin = displace + args.length * args.velocity

    # plumed dat
    rcomstr = ",".join(["%i" % (i + 1) for i in rec_idx])
    lcomstr = ",".join(["%i" % (i + 1) for i in lig_idx])
    plumedtxt = f"""
c1: COM ATOMS={rcomstr}
c2: COM ATOMS={lcomstr}
dist: DISTANCE ATOMS=c1,c2
MOVINGRESTRAINT ...
    label=bias
    ARG=dist
    STEP0=0 AT0={displace} KAPPA0={args.fconst}
    STEP1={nstep} AT1={displace_fin} KAPPA1={args.fconst}
...
FLUSH STRIDE={args.nprint}
PRINT ARG=dist,bias.dist_cntr,bias.dist_work,bias.force2 STRIDE={args.nprint} FILE={args.output}"""
    system.addForce(PlumedForce(plumedtxt))

    import mdtraj as md
    traj = md.load(args.topol)
    bb_index = np.array([1 if a.name.strip() in ["CA", "C", "N", "O"] else 0 for a in traj.topology.atoms])
    if args.restraint == "none":
        print("No restraint.")
    elif args.restraint == "all":
        idx0 = np.zeros((pos.shape[0],))
        idx0[rec_idx] = 1.0
        rmsd_r = mm.RMSDForce(pos * unit.nanometer, idx0 * bb_index)
        idx0 = np.zeros((pos.shape[0],))
        idx0[lig_idx] = 1.0
        rmsd_l = mm.RMSDForce(pos * unit.nanometer, idx0 * bb_index)
        res_bias = mm.CustomCVForce("0.5*2000*(cv1^2+cv2^2)")
        res_bias.addCollectiveVariable("cv1", rmsd_r)
        res_bias.addCollectiveVariable("cv2", rmsd_l)
        system.addForce(res_bias)
    elif args.restraint == "ss":
        dssp = md.compute_dssp(traj)
        ss_idx = np.array([1 if dssp[0, atom.residue.index] in [
            "H", "E"] else 0 for atom in traj.topology.atoms])

        idx0 = np.zeros((pos.shape[0],))
        idx0[rec_idx] = 1.0
        rmsd_r = mm.RMSDForce(pos * unit.nanometer, idx0 * ss_idx * bb_index)
        idx0 = np.zeros((pos.shape[0],))
        idx0[lig_idx] = 1.0
        rmsd_l = mm.RMSDForce(pos * unit.nanometer, idx0 * ss_idx * bb_index)
        res_bias = mm.CustomCVForce("0.5*2000*(cv1^2+cv2^2)")
        res_bias.addCollectiveVariable("cv1", rmsd_r)
        res_bias.addCollectiveVariable("cv2", rmsd_l)
        system.addForce(res_bias)
    else:
        raise BaseException(f"Unsupported restraint type {args.restraint}.")

    integ = mm.LangevinIntegrator(300.0 * unit.kelvin,
                                  5.0 / unit.picosecond,
                                  args.delta * unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integ, plat)
    with open(args.input, "r") as f:
        state = mm.XmlSerializer.deserialize(f.read())
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())

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
    simulation.reporters.append(app.DCDReporter(args.traj, args.nprint))
    simulation.step(nstep)
