import argparse
from steermd.optChain import optSideChain
from steermd.generate import genIndices, genStates
from steermd.steerConstVel import steerMDConstVel
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Pulling ligand COM along receptor-ligand COM vector')
    subparsers = parser.add_subparsers()
    parser_optsc = subparsers.add_parser(
        "optsc",
        help="Optimize side chain structure under implicit solvent model.")
    parser_optsc.add_argument("-r",
                              "--receptor",
                              dest="receptor",
                              type=str,
                              required=True,
                              help="Receptor PDB file.")
    parser_optsc.add_argument("-l",
                              "--ligand",
                              dest="ligand",
                              type=str,
                              required=True,
                              help="Ligand PDB file.")
    parser_optsc.add_argument("-o",
                              "--output",
                              default="relaxed.pdb",
                              help="Sidechain relaxed conformation.")
    parser_optsc.add_argument(
        "--nstep",
        type=int,
        default=5000,
        help="Number of steps. Total steps is 20 * nstep.")
    parser_optsc.add_argument("-x",
                              "--traj",
                              dest="rlxtrj",
                              type=str,
                              default="relaxed.dcd",
                              help="Relaxed trajectory.")
    parser_optsc.set_defaults(func=optSideChain)

    parser_genidx = subparsers.add_parser(
        "genidx", help="Generate indices of receptor and ligand.")
    parser_genidx.add_argument("-r",
                               "--receptor",
                               dest="receptor",
                               type=str,
                               required=True,
                               help="Receptor PDB file.")
    parser_genidx.add_argument("-l",
                               "--ligand",
                               dest="ligand",
                               type=str,
                               required=True,
                               help="Ligand PDB file.")
    parser_genidx.add_argument("-o",
                               "--output",
                               type=str,
                               default="index.txt",
                               help="Indices of receptor and ligand.")
    parser_genidx.set_defaults(func=genIndices)

    parser_genstate = subparsers.add_parser(
        "genstate", help="Generate state using enhanced sampling.")
    parser_genstate.add_argument("-i",
                                 "--input",
                                 type=str,
                                 required=True,
                                 help="Input PDB structure.")
    parser_genstate.add_argument("--alpha",
                                 type=float,
                                 default=0.5,
                                 help="Alpha of each state.")
    parser_genstate.add_argument("--ecut",
                                 type=float,
                                 default=100.0,
                                 help="Energy cut of each state (kJ/mol).")
    parser_genstate.add_argument("-n",
                                 "--pname",
                                 type=str,
                                 default="state",
                                 help="Project name.")
    parser_genstate.add_argument("-x",
                                 "--traj",
                                 type=str,
                                 default="genstate.dcd",
                                 help="Trajectory.")
    parser_genstate.add_argument("--nstate",
                                 type=int,
                                 default=10,
                                 help="Number of states.")
    parser_genstate.add_argument("--nstep",
                                 type=int,
                                 default=50000,
                                 help="Num. of steps per state.")
    parser_genstate.set_defaults(func=genStates)

    parser_steer = subparsers.add_parser(
        "steer",
        help="Steer ligand to leave receptor using constant velocity.")

    parser_steer.add_argument("-i",
                              "--input",
                              type=str,
                              required=True,
                              help="Input state.")
    parser_steer.add_argument("-t",
                              "--topol",
                              type=str,
                              required=True,
                              help="Input topology PDB file.")
    parser_steer.add_argument(
        "-n",
        "--index",
        type=str,
        required=True,
        help="Input index file to distinguish receptor and ligand.")
    parser_steer.add_argument("-o",
                              "--output",
                              type=str,
                              default="steer.txt",
                              help="Steered force data.")
    parser_steer.add_argument("-x",
                              dest="traj",
                              type=str,
                              default="steered.dcd",
                              help="Pulling trajectory.")
    parser_steer.add_argument("-v",
                              "--vel",
                              dest="velocity",
                              type=float,
                              default=0.04,
                              help="Pulling velocity (nm/ps)")
    parser_steer.add_argument("-l",
                              "--length",
                              dest="length",
                              type=float,
                              default=1000.0,
                              help="Simulation length (ps)")
    parser_steer.add_argument("-f",
                              "--fconst",
                              dest="fconst",
                              type=float,
                              default=2000.0,
                              help="Force constant (kJ/mol/nm^2)")
    parser_steer.add_argument("--nprint",
                              type=int,
                              default=100,
                              help="Steps to print steering force.")
    parser_steer.add_argument("--delta",
                              type=float,
                              default=2.0,
                              help="Timestep in fs.")
    parser_steer.set_defaults(func=steerMDConstVel)

    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()