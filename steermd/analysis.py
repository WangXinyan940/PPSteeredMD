try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import numpy as np


def readOutput(fname):
    with open(fname, "r") as f:
        text = f.readlines()
    data = [[float(j) for j in i.strip().split()] for i in text]
    data = np.array(data)
    time = data[:,0]
    dt = time[1] - time[0]
    ref = data[:,-2]
    dx = ref[1] - ref[0]
    vel = dx / dt
    force = data[:,-1]
    work = np.zeros((data.shape[0],))
    for ii in range(data.shape[0]):
        work[ii] = force[:ii].sum() * dt * vel # ps * nm / ps * kJ/mol / nm = 
    return time, force, work

def genSpect(args):
    nfiles = len(args.input)
    if nfiles == 1:
        time, fout, wout = readOutput(args.input[0])
    else:
        works = []
        forces = []
        for fname in args.input:
            time, force, work = readOutput(fname)
            works.append(work)
            forces.append(force)
        # ensemble average of work
        beta = 1. / 8.314 / args.temperature * 1000.0
        works = np.array(work)
        wout = 1. / beta * np.log(np.exp(- beta * works).mean(axis=0))
        forces = np.array(forces)
        fout = forces.mean(axis=0)

    with open(args.output, "w") as f:
        for ii in range(time.shape[0]):
            f.write("%.8f %.8f %.8f\n"%(time[ii], fout[ii], wout[ii]))

