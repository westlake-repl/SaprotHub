from pyrosetta import *
from pyrosetta.rosetta import *


init("-beta_nov16 -output_pose_energies_table false -ex1 -ex2aro -mute all")


# optimize af2 predicted pdb
def optimize_pdb(pdb_path, save_path):
    pose = pose_from_file(pdb_path)
    scorefxn = pyrosetta.create_score_function("beta_nov16")
    fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=3)
    tf_relax = core.pack.task.TaskFactory()
    tf_relax.push_back(core.pack.task.operation.InitializeFromCommandline())
    tf_relax.push_back(core.pack.task.operation.RestrictToRepacking())
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    # mm.set_bb(True)
    mm.set_chi(True)
    fast_relax.set_task_factory(tf_relax)
    fast_relax.set_movemap(mm)
    fast_relax.apply(pose)
    pose.dump_pdb(save_path)
    
    