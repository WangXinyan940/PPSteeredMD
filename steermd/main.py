import argparse
from steermd.optChain import optSideChain
from steermd.generate import genIndices, genStates
from steermd.steerConstVel import steerMDConstVel
from steermd.analysis import genSpect
from steermd.steerMartini import runMSteer
from steermd.steerPlumed import runSteerMDPlumed
from steermd.runMD import regularMD
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

    parser_msteer = subparsers.add_parser(
        "msteer",
        help="Steer ligand to leave receptor using constant velocity.")

    parser_msteer.add_argument("-i",
                               "--input",
                               type=str,
                               required=True,
                               help="Input PDB.")
    parser_msteer.add_argument(
        "-n",
        "--index",
        type=str,
        required=True,
        help="Input index file to distinguish receptor and ligand.")
    parser_msteer.add_argument("-o",
                               "--output",
                               type=str,
                               default="steer.txt",
                               help="Steered force data.")
    parser_msteer.add_argument("-v",
                               "--vel",
                               dest="velocity",
                               type=float,
                               default=0.04,
                               help="Pulling velocity (nm/ps)")
    parser_msteer.add_argument("-f",
                               "--fconst",
                               dest="kconst",
                               type=float,
                               default=2000.0,
                               help="Force constant (kJ/mol/nm^2)")
    parser_msteer.add_argument("--nprint",
                               type=int,
                               default=10,
                               help="Steps to print steering force.")
    parser_msteer.add_argument("--gmx", type=str, default="gmx", help="")
    parser_msteer.add_argument("--nthreads", type=int, default=8, help="")
    parser_msteer.set_defaults(func=runMSteer)

    parser_psteer = subparsers.add_parser(
        "psteer",
        help="Steer ligand to leave receptor using constant velocity.")

    parser_psteer.add_argument("-i",
                               "--input",
                               type=str,
                               required=True,
                               help="Input state.")
    parser_psteer.add_argument("-t",
                               "--topol",
                               type=str,
                               required=True,
                               help="Input topology PDB file.")
    parser_psteer.add_argument(
        "-n",
        "--index",
        type=str,
        required=True,
        help="Input index file to distinguish receptor and ligand.")
    parser_psteer.add_argument("-o",
                               "--output",
                               type=str,
                               default="steer.txt",
                               help="Steered force data.")
    parser_psteer.add_argument("-x",
                               dest="traj",
                               type=str,
                               default="steered.dcd",
                               help="Pulling trajectory.")
    parser_psteer.add_argument("-v",
                               "--vel",
                               dest="velocity",
                               type=float,
                               default=0.001,
                               help="Pulling velocity (nm/ps)")
    parser_psteer.add_argument("-l",
                               "--length",
                               dest="length",
                               type=float,
                               default=4000.0,
                               help="Simulation length (ps)")
    parser_psteer.add_argument("-f",
                               "--fconst",
                               dest="fconst",
                               type=float,
                               default=2500.0,
                               help="Force constant (kJ/mol/nm^2)")
    parser_psteer.add_argument("--nprint",
                               type=int,
                               default=100,
                               help="Steps to print steering force.")
    parser_psteer.add_argument("--delta",
                               type=float,
                               default=2.0,
                               help="Timestep in fs.")
    parser_psteer.set_defaults(func=runSteerMDPlumed)

    parser_spect = subparsers.add_parser("spect", help="")
    parser_spect.add_argument("-i",
                              "--input",
                              type=str,
                              nargs="+",
                              required=True,
                              help="SteerMD output(s).")
    parser_spect.add_argument("-o",
                              "--output",
                              type=str,
                              default="out.txt",
                              help="PMF.")
    parser_spect.add_argument("-t",
                              "--temperature",
                              type=float,
                              default=300.0,
                              help="Temperature.")
    parser_spect.set_defaults(func=genSpect)

    parser_bfmd = subparsers.add_parser("bfmd",
                                        help="Brute-force MD for sampling.")

    parser_bfmd.add_argument("-i",
                             "--input",
                             type=str,
                             required=True,
                             help="Input PDB.")
    # parser_steer.add_argument(
    #     "-n",
    #     "--index",
    #     type=str,
    #     required=True,
    #     help="Input index file to distinguish receptor and ligand.")
    parser_bfmd.add_argument("-x",
                             dest="traj",
                             type=str,
                             default="steered.dcd",
                             help="Pulling trajectory.")
    parser_bfmd.add_argument("-l",
                             "--length",
                             dest="length",
                             type=float,
                             default=1000.0,
                             help="Simulation length (ps)")
    parser_bfmd.add_argument("--delta",
                             type=float,
                             default=2.0,
                             help="Timestep in fs.")
    parser_bfmd.set_defaults(func=regularMD)

    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()