import json
import numpy as np
import pandas as pd
import os

import torch
import subprocess

from .data_transform import make_dist_map
from tqdm import tqdm
from utils.mpr import MultipleProcessRunner


# Calculate TMscore between two structures. TMscore command tool is required to be installed.
def get_tmscore(TMscore: str, pdb_path1: str, pdb_path2: str):
	"""
	Args:
		TMscore: path to TMscore command line tool
		pdb_path1: path to pdb file 1
		pdb_path2: path to pdb file 2

	Returns:
		TMscore value
	"""
	cmd = f"{TMscore} {pdb_path1} {pdb_path2} | grep 'TM-score.*d0'"

	r = os.popen(cmd)
	text = r.read()
	value = float(text.split("=")[1].strip().split(' ')[0])
	return value


# Calculate lddt between true structure and predicted structure:
def get_lddt(coords1, coords2, threshold=15):
	"""

	Args:
		coords1: [seq_len, 4, 3]. 4: N CA C O
		threshold: Threshold of distance between two atoms

	Returns:
		lddt value
	"""
	dist1 = make_dist_map(coords1)
	dist2 = make_dist_map(coords2)
	
	mask = dist1 < threshold
	gap = torch.abs(dist1 - dist2)[mask]
	
	lddt = sum([(gap < t).sum() / gap.numel() for t in [0.5, 1, 2, 4]]) / 4
	return lddt.to('cpu').item()


# Get plddt from alphafold2 predicted pdb file
def get_plddt(plddt_path):
	"""
	
	Args:
		plddt_path: File path

	Returns:
		Mean plddt value
		
	"""
	
	with open(plddt_path, 'r') as r:
		info = json.load(r)
		mean_plddt = np.array(info["confidenceScore"]).mean()
	
	return mean_plddt


# Get plddts from a directory
def get_plddts(plddt_dir) -> pd.DataFrame:
	"""
	
	Args:
		plddt_dir: Directory path of plddt files

	Returns:
		Dataframe that contains mean plddt of all files
		
	"""
	
	files = [os.path.join(plddt_dir, file) for file in os.listdir(plddt_dir)]
	df = pd.DataFrame(columns=["file", "mean_plddt"])
	
	for file in tqdm(files, "Parsing plddt files..."):
		mean_plddt = get_plddt(file)
		df = df.append({
			"file": file,
			"mean_plddt": mean_plddt,
		}, ignore_index=True)
	
	return df


# Extracting plddts using multiple processes
class PlddtCalculator(MultipleProcessRunner):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def _target(self, process_id, data, sub_path, *args):
		with open(sub_path, "w") as w:
			w.write("name\tmean_plddt\n")
			
			for i, data_path in enumerate(data):
				_, file = os.path.split(data_path)
				name, _ = os.path.splitext(file)
				mean_plddt = get_plddt(data_path)
				
				w.write(f"{name}\t{mean_plddt}\n")
				
				self.terminal_progress_bar(process_id, i+1, len(data), f"Process{process_id}: calculating plddt...")
	
	def __len__(self):
		return len(self.data)
	
	def _aggregate(self, final_path: str, sub_paths):
		with open(final_path, 'w') as w:
			w.write("name\tmean_plddt\n")
			
			for sub_path in sub_paths:
				with open(sub_path, 'r') as r:
					# Skip header
					r.readline()
					for line in tqdm(r, f"Aggregating plddts..."):
						w.write(line)
				
				os.remove(sub_path)