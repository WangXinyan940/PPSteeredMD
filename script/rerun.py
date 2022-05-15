try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
from tqdm import trange, tqdm
import numpy as np
import mdtraj as md
import argparse
import sys


class RerunError(BaseException):
    pass


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="Input trajectory file.")
    parser.add_argument("-t",
                        "--topol",
                        type=str,
                        required=True,
                        help="Topology PDB file.")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="Output energy file.")
    parser.add_argument(
        "--forcefield",
        type=str,
        default="amber14",
        help="Forcefield to use. Only amber99, amber14 and amoeba are allowed."
    )

    args = parser.parse_args()
    return args


def rerun(args):
    traj = md.load(args.input, top=args.topol)

    # set system
    pdb = app.PDBFile(args.topol)
    if args.forcefield == "amber99":
        ff = app.ForceField("amber99sbildn.xml", "implicit/gbn2.xml")
    elif args.forcefield == "amber14":
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    elif args.forcefield == "amoeba":
        ff = app.ForceField("amoeba2018.xml", "amoeba2018_gk.xml")
    else:
        raise RerunError(
            f"Forcefield type {args.forcefield} is not supported. Please use amber99, amber14 or amoeba."
        )
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer)
    integ = mm.VerletIntegrator(1e-8)
    context = mm.Context(system, integ)

    with open(args.output, "w") as f:
        for nframe in traj.n_frame:
            frame = traj[nframe]
            context.setPositions(frame.xyz)
            energy = context.getState(
                getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole)
            f.write(f"{energy:.16e}\n")


if __name__ == "__main__":
    args = parseArgs()
    rerun(args)