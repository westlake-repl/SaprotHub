import torch


from Bio import pairwise2
from .data_transform import *
from .parse import parse_structure


# Convert pdb to features
def pdb2feature(pdb_path: str, device: str = 'cpu'):
	coords = parse_structure("pdb", pdb_path)['coords']
	pair_feature = make_rbf(coords, 16, device=device)
	pair_feature = torch.cat([pair_feature, make_orientations(coords, device)], dim=-1)

	return pair_feature


# Convert coordinates to features
def coords2feature(coords: dict, device: str = 'cpu'):
	pair_feature = make_rbf(coords, device=device)
	pair_feature = torch.cat([pair_feature, make_orientations(coords, device)], dim=-1)
	# pair_feature = make_orientations(coords, device)
	return pair_feature


# Convert coordinates to distance maps
def coords2dist(coords: dict, device: str = 'cpu'):
	tensor_coords = dict2tensor(coords, device=device)
	dist_map = make_dist_map(tensor_coords)
	return dist_map


# Convert a distance map to pair features
def dist2feature(dist, device: str = "cpu"):
	dist = dist.to(device)
	pair_feature = torch.cat([rbf(dist[..., i], 2., 22., 16) for i in range(dist.shape[-1])], dim=-1)
	# pair_feature = torch.cat([pair_feature, make_orientations(coords, device)], dim=-1)

	return pair_feature


# Convert a batch of coordinates to features
def batch_coords2feature(coords: list, device: str = 'cpu'):
	padding_length = max([len(coord["N"]) for coord in coords])

	# Feature has 407 channels
	feature_size = 407
	padded_feature = torch.zeros(len(coords), padding_length, padding_length, feature_size, device=device)

	for i, coord in enumerate(coords):
		if coord is not None:
			pair_feature = coords2feature(coord, device=device)
			l = pair_feature.size(0)
			padded_feature[i, :l, :l] = pair_feature

	return padded_feature


# Convert a batch of coordinates to distance maps
def batch_coords2dist(coords: list, device: str = 'cpu'):
	padding_length = max([len(coord["N"]) for coord in coords])

	# Distance map has 25 channels
	feature_size = 25
	padded_maps = torch.zeros(len(coords), padding_length, padding_length, feature_size, device=device)

	for i, coord in enumerate(coords):
		if coord is not None:
			dist_map = coords2dist(coord, device=device)
			l = dist_map.size(0)
			padded_maps[i, :l, :l] = dist_map

	return padded_maps


# Align mutated and wild tpye sequences and get aligned features
def get_aligned_feature(mut_seq: str, wt_seq: str, wt_feature: torch.Tensor, align: bool = True):
	if align:
		alignments = pairwise2.align.globalxx(mut_seq, wt_seq)
		aligned_mut_seq, aligned_wt_seq, _, _, _ = alignments[0]

	else:
		aligned_mut_seq, aligned_wt_seq = mut_seq, wt_seq

	r, _, f = wt_feature.size()
	
	# For gap in mutated sequence, we remove the corresponding feature in wt_feature.
	# For gap in wild type sequence, we add a zero feature.

	# expand wild type feature
	np_wt_seq = np.array(list(aligned_wt_seq))
	for n, index in enumerate((np_wt_seq == '-').nonzero()[0]):
		wt_feature = torch.cat([wt_feature[:index],
								torch.zeros(1, r+n, f).to(wt_feature),
								wt_feature[index:]], dim=0)

		wt_feature = torch.cat([wt_feature[:, :index],
								torch.zeros(r+n+1, 1, f).to(wt_feature),
								wt_feature[:, index:]], dim=1)

	np_mut_seq = np.array(list(aligned_mut_seq))
	# Mask is used to remove the corresponding feature in wt_feature for gap in mutated sequence.
	mask = np_mut_seq != '-'

	# calculate the outer product of mask
	mask_2d = np.outer(mask, mask)

	# remove the corresponding feature in wt_feature for gap in mutated sequence.
	wt_feature = wt_feature[mask_2d].view(-1, mask.sum(), f)

	return wt_feature

			