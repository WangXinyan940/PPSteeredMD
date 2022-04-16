import os
import numpy as np
try:
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm.app as app
    import simtk.unit as unit
from steermd.utils import readIndices

HOME = os.path.dirname(os.path.abspath(__file__))


def buildMartiniSys(pdbname,
                    recidx,
                    ligidx,
                    geomfile,
                    topfile,
                    plumedfile,
                    output,
                    kconst,
                    vel,
                    nprint=100,
                    gmx="gmx",
                    nt=8):
    # Build martini
    os.system(
        f"martinize2 -f {pdbname} -x {geomfile} -o {topfile} -ignh -p backbone -ff martini3001 -elastic -ef 700.0 -el 0.5 -eu 0.9 -ea 0 -ep 0 -scfix -cys auto"
    )
    # move files
    os.system(f"cp {HOME}/prm/martini.itp .")
    os.system(f"cp {HOME}/prm/martini_v3.0.0_ions_v1.itp .")
    os.system(f"cp {HOME}/prm/martini_v3.0.0.itp .")
    os.system(f"cp {HOME}/prm/martini_v3.0.0_solvents_v1.itp .")
    os.system(f"cp {HOME}/prm/water.gro .")
    os.system(f"cp {HOME}/prm/minim.mdp .")
    os.system(f"cp {HOME}/prm/eq.mdp .")
    os.system(f"cp {HOME}/prm/dynamic.mdp .")

    # build res idx
    pdbfile = app.PDBFile(pdbname)
    res = [r for r in pdbfile.topology.residues()]
    recres = [
        r.index for r in res if [a for a in r.atoms()][0].index in recidx
    ]
    ligres = [
        r.index for r in res if [a for a in r.atoms()][0].index in ligidx
    ]

    cgpdb = app.PDBFile(geomfile)
    cgrecidx = [
        a.index for a in cgpdb.topology.atoms() if a.residue.index in recres
    ]
    cgligidx = [
        a.index for a in cgpdb.topology.atoms() if a.residue.index in ligres
    ]

    # calc COM distance
    geom = cgpdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    rcom = geom[cgrecidx, :].mean(axis=0)
    lcom = geom[cgligidx, :].mean(axis=0)
    dist = np.linalg.norm(rcom - lcom)

    stepfin = int(4.0 / vel / 0.02 + 1e-3)

    # Build plumed file
    rcomstr = ",".join(["%i" % (i + 1) for i in cgrecidx])
    lcomstr = ",".join(["%i" % (i + 1) for i in cgligidx])
    with open(plumedfile, "w") as f:
        f.write(f"""
c1: COM ATOMS={rcomstr}
c2: COM ATOMS={lcomstr}
dist: DISTANCE ATOMS=c1,c2
MOVINGRESTRAINT ...
    label=bias
    ARG=dist
    STEP0=0 AT0={dist} KAPPA0={kconst}
    STEP1={stepfin} AT1={dist+4} KAPPA1={kconst}
...
FLUSH STRIDE={nprint}
PRINT ARG=dist,bias.dist_cntr,bias.dist_work,bias.force2 STRIDE={nprint} FILE={output}""")

    os.system(f"{gmx} editconf -f {geomfile} -d 4.0 -bt cubic -o box.gro")
    os.system(
        f"{gmx} grompp -f minim.mdp -c box.gro -p {topfile} -o min.tpr -r box.gro"
    )
    os.system(f"{gmx} mdrun -v -deffnm min -nt {nt}")
    os.system(
        f"{gmx} solvate -cp min.gro -cs water.gro -radius 0.21 -o solvated.gro -p {topfile}"
    )
    os.system(
        f"{gmx} grompp -f minim.mdp -c solvated.gro -p {topfile} -o min2.tpr -r solvated.gro -maxwarn 1"
    )
    os.system(f"{gmx} mdrun -v -deffnm min2 -nt {nt} --nsteps 5000")
    os.system(
        f"{gmx} grompp -f eq.mdp -c min2.gro -p {topfile} -o eq.tpr -r min2.gro -maxwarn 1"
    )
    os.system(f"{gmx} mdrun -v -deffnm eq -nt {nt}")
    os.system(
        f"{gmx} grompp -f dynamic.mdp -c eq.gro -p {topfile} -t eq.cpt -o dyn.tpr -maxwarn 1"
    )
    os.system(
        f"{gmx} mdrun -v -deffnm dyn -nt {nt} --nsteps {stepfin} --plumed {plumedfile}"
    )


def runMSteer(args):
    indices = readIndices(args.index)
    rec_idx = indices["receptor"]
    lig_idx = indices["ligand"]
    buildMartiniSys(args.input,
                    rec_idx,
                    lig_idx,
                    "cg_complex.pdb",
                    "cg_complex.top",
                    "plumed.dat",
                    args.output,
                    args.kconst,
                    args.velocity,
                    nprint=args.nprint,
                    gmx=args.gmx,
                    nt=args.nthreads)
