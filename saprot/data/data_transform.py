import torch
import torch.nn.functional as F
import numpy as np

aa = 'ACDEFGHIKLMNPQRSTVWY'
aa2id = {aa: i for i, aa in enumerate(aa)}


# Get one-hot encoding of a sequence
def get_one_hot(seq, device="cpu"):
	one_hot = torch.zeros(len(seq), 20, device=device)
	for i, aa in enumerate(seq):
		one_hot[i, aa2id[aa]] = 1
	
	return one_hot


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
	batch_size = len(sequences)
	shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
	
	if dtype is None:
		dtype = sequences[0].dtype
	
	if isinstance(sequences[0], np.ndarray):
		array = np.full(shape, constant_value, dtype=dtype)
	elif isinstance(sequences[0], torch.Tensor):
		device = sequences[0].device
		array = torch.full(shape, constant_value, dtype=dtype, device=device)
	
	for arr, seq in zip(array, sequences):
		arrslice = tuple(slice(dim) for dim in seq.shape)
		arr[arrslice] = seq
	
	return array


def rbf(values, v_min, v_max, n_bins=16):
	"""

	Args:
		values:
		v_min:
		v_max:
		n_bins:

	Returns: RBF encodings in a new dimension at the end

	"""

	rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
	rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
	rbf_std = (v_max - v_min) / n_bins
	z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
	return torch.exp(-z ** 2)


def make_dist_map(coords, eps=1e-6):
	"""

		Args:
			coords: [seq_len, 4, 3]. 4: N CA C O
			eps:    add a small value to avoid square root error

		Returns:
			Distance map

		"""

	# transform dict of coordinates to tensor
	# coords: [seq_len, 4, 3]. 4: N CA C O
	seq_len = coords.size(0)

	# calculate pseudo CB coordinate
	vec_NtoCA = coords[..., 1, :] - coords[..., 0, :]
	vec_CAtoC = coords[..., 2, :] - coords[..., 1, :]

	vec_cross_product = torch.cross(vec_NtoCA, vec_CAtoC, dim=-1)
	CB = -0.58273431 * vec_NtoCA + 0.56802827 * vec_CAtoC - 0.54067466 * vec_cross_product + coords[..., 1, :]

	# coords: [seq_len, 5, 3]. 5: N CA C O CB
	coords = torch.cat([coords, CB.unsqueeze(dim=-2)], dim=-2)

	# calculate pairwise atom distance between residues
	# coords_view: [seq_len, seq_len, 5, 5, 3]
	coords_view1 = coords[..., None, :, None, :].repeat(1, seq_len, 1, 5, 1)
	coords_view2 = coords[..., None, :, None, :, :].repeat(seq_len, 1, 5, 1, 1)

	# coords_dist: [seq_len, seq_len, 25]
	coords_dist = ((coords_view1 - coords_view2).square().sum(dim=-1) + eps).sqrt().reshape(seq_len, seq_len, -1)

	return coords_dist


def make_rbf(coord_dict, num_rbf=16, device='cpu'):
	"""
	
	Args:
		coord_dict: dict of a sequence that has keys: "N", "CA", "C", "O"
		num_rbf:    number of dimension of rbf
		eps:        add a small value to avoid square root error

	Returns:
		rbf feature [seq_len, seq_len, 25 * num_rbf]

	"""
	
	coords = torch.stack([torch.tensor(coord_dict[k]) for k in ["N", "CA", "C", "O"]], dim=1)
	coords = coords.to(device)
	coords_dist = make_dist_map(coords)

	# RBFs  : [seq_len, seq_len, 25 * num_rbf]
	RBFs = torch.cat([rbf(coords_dist[..., i], 2., 22., num_rbf) for i in range(coords_dist.shape[-1])], dim=-1)

	return RBFs


def make_orientations(coord_dict, device='cuda'):
	"""
	
	Args:
		coord_dict: dict of a sequence that has the order: "N", "CA", "C", "O"

	Returns:
		orientation feature [seq_len, seq_len, 25 * num_rbf]

	"""
	coord_dict = {k: torch.tensor(v).to(device) for k, v in coord_dict.items()}
	
	X_N, X_CA, X_C, _ = [atom_coords for atom_coords in coord_dict.values()]
	vec_CAtoC = F.normalize(X_C - X_CA, dim=-1)
	vec_CAtoN = F.normalize(X_N - X_CA, dim=-1)
	
	vec_cross_product = F.normalize(torch.cross(vec_CAtoN, vec_CAtoC), dim=-1)
	
	# O [..., seq_len, 3, 3]
	O = torch.stack((vec_CAtoN, vec_cross_product, torch.cross(vec_CAtoN, vec_cross_product)), -2)
	
	# O [..., seq_len, seq_len, 3]
	dX = X_CA[..., None, :] - X_CA[..., None, :, :]
	
	# [seq_len, 1, 3, 3] * [seq_len, seq_len, 3, 1] => [seq_len, seq_len, 3 ,1]
	dU = torch.matmul(O[..., None, :, :].transpose(-1, -2), dX.unsqueeze(-1)).squeeze(-1)
	dU = F.normalize(dU, dim=-1)
	
	# Rotate into local reference frames
	rots = torch.matmul(O.unsqueeze(-3).transpose(-1, -2), O.unsqueeze(-4))
	quats = rot2quat(rots)
	
	# Orientation features
	O_features = torch.cat((dU, quats), dim=-1)
	return O_features


def rot2quat(rots):
	""" Convert a batch of 3D rotations [R] to quaternions [Q]
		rots [N, L, 3, 3]
		quats [N, L, 4]
	"""
	diag = torch.diagonal(rots, dim1=-2, dim2=-1)
	Rxx, Ryy, Rzz = diag.unbind(-1)
	magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
			Rxx - Ryy - Rzz,
		- Rxx + Ryy - Rzz,
		- Rxx - Ryy + Rzz
	], -1)))
	_R = lambda i, j: rots[..., i, j]
	signs = torch.sign(torch.stack([
		_R(2,1) - _R(1,2),
		_R(0,2) - _R(2,0),
		_R(1,0) - _R(0,1)
	], -1))
	xyz = signs * magnitudes
	# The relu enforces a non-negative trace
	w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
	Q = torch.cat((xyz, w), -1)
	Q = F.normalize(Q, dim=-1)

	return Q
	# rots = [[rots[..., i, j] for j in range(3)] for i in range(3)]
	# [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rots
	#
	# k = [
	# 	[xx + yy + zz, zy - yz, xz - zx, yx - xy, ],
	# 	[zy - yz, xx - yy - zz, xy + yx, xz + zx, ],
	# 	[xz - zx, xy + yx, yy - xx - zz, yz + zy, ],
	# 	[yx - xy, xz + zx, yz + zy, zz - xx - yy, ]
	# ]
	#
	# k = (1. / 3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)
	#
	# _, vectors = torch.linalg.eigh(k)
	# return vectors[..., -1]


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
	mat = np.zeros((4, 4))
	for pair in pairs:
		key, value = pair
		ind = _qtr_ind_dict[key]
		mat[ind // 4][ind % 4] = value

	return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat2rot(quats):
	"""
	        Converts a quaternion to a rotation matrix.

	        Args:
	            quat: [*, 4] quaternions
	        Returns:
	            [*, 3, 3] rotation matrices
	    """
	# [*, 4, 4]
	quats = quats[..., None] * quats[..., None, :]
	
	# [4, 4, 3, 3]
	mat = torch.tensor(_QTR_MAT, dtype=quats.dtype, device=quats.device)
	
	# [*, 4, 4, 3, 3]
	shaped_qtr_mat = mat.view((1,) * len(quats.shape[:-2]) + mat.shape)
	quats = quats[..., None, None] * shaped_qtr_mat
	
	# [*, 3, 3]
	return torch.sum(quats, dim=(-3, -4))


def normalize_vector(v, dim, eps=1e-6):
	return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
	"""
	Description:
	Project vector `v` onto vector `e`.
	Args:
	v: (N, L, 3).
	e: (N, L, 3).
	"""
	return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
	"""
	Args:
	center: (N, L, 3), usually the position of CA.
	p1:   (N, L, 3), usually the position of C.
	p2:   (N, L, 3), usually the position of N.
	Returns
		R: A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
			The matrix is composed of 3 column vectors: [e1, e2, e3].
		t: A batch of translation vector, (N, L, 3).
	"""
	
	v1 = p1 - center  # (N, L, 3)
	e1 = normalize_vector(v1, dim=-1)

	v2 = p2 - center  # (N, L, 3)
	u2 = v2 - project_v2v(v2, e1, dim=-1)
	e2 = normalize_vector(u2, dim=-1)
	
	e3 = torch.cross(e1, e2, dim=-1)  # (N, L, 3)
	
	R = torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1) # (N, L, 3, 3_index)
	
	return R, center


def local_to_global(R, t, p):
	"""
	Description:
	Convert local (internal) coordinates to global (external) coordinates q.
	q <- Rp + t
	Args:
	R: (N, L, 3, 3).
	t: (N, L, 3).
	p: Local coordinates, (N, L, A, 3).
	Returns:
	q: Global coordinates, (N, L, A, 3).
	"""
	assert p.size(-1) == 3
	p_size = p.size()
	N, L = p_size[0], p_size[1]
	
	p = p.view(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
	q = torch.matmul(R, p) + t.unsqueeze(-1)  # (N, L, 3, *)
	q = q.transpose(-1, -2).reshape(p_size)   # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
	return q


def global_to_local(R, t, q):
	"""
	Description:
	Convert global (external) coordinates q to local (internal) coordinates p.
	p <- R^{T}(q - t)
	Args:
	R: (N, L, 3, 3).
	t: (N, L, 3).
	q: Global coordinates, (N, L, A, 3).
	Returns:
	p: Local coordinates, (N, L, A, 3).
	"""
	assert q.size(-1) == 3
	q = q.unsqueeze(dim=-2).transpose(-1, -2)
	p = torch.matmul(R[:, :, None, :].transpose(-1, -2), (q - t[:, :, None, :, None])) # (N, L, 3, *)
	p = p.transpose(-1, -2).squeeze(dim=-2)   # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)

	return p


# Add noise to frames of coordinate
def add_frame_noise(coords: torch.Tensor, mean, std, mask: torch.Tensor = None):
	"""
	
	Args:
		coords: [N, L, 4, 3]. 4 means N, CA, C, O and 3 means coordinates
		mean: Mean value of normal distribution
		std: Standard error of normal distribution
		mask: [N, L]. For all positions that need to be noised, set the value to 1

	Returns:
		New coords with noise: [N, L, 4, 3]

	"""
	center = coords[..., 1, :]
	p1 = coords[..., 2, :]
	p2 = coords[..., 0, :]
	
	rots, t = construct_3d_basis(center, p1, p2)
	quats = rot2quat(rots)
	
	transform_vec = torch.cat([quats, t], dim=-1)
	mean = torch.full_like(transform_vec, mean)
	std = torch.full_like(transform_vec, std)
	transform_vec += torch.normal(mean, std)
	
	noised_quats, noised_t = transform_vec[..., :4], transform_vec[..., 4:]
	noised_quats = noised_quats / torch.norm(noised_quats, dim=-1, keepdim=True)
	noised_rots = quat2rot(noised_quats)
	
	local_coords = global_to_local(rots, t, coords)
	noised_coords = local_to_global(noised_rots, noised_t, local_coords)

	if mask is None:
		N, L, _, _ = coords.shape
		mask = torch.ones(N, L, device=coords.device, dtype=torch.bool)
	
	coords[mask] = noised_coords[mask]
	return coords


def dict2tensor(dict_coords: dict, device="cpu"):
	"""
	Convert dict of coordinates to tensor

	Args:
		dict_coords: dict, keys are N, CA, C, O
		device:

	Returns:
		coords: [L, 4, 3] tensor

	"""
	coords = torch.stack([torch.tensor(dict_coords[k]) for k in ["N", "CA", "C", "O"]], dim=1)
	coords = coords.to(device)

	return coords

def tensor2dict(coords: torch.Tensor):
	"""
	Convert tensor of coordinates to dict

	Args:
		coords: [L, 4, 3] tensor
		device:

	Returns:
		dict_coords: dict, keys are N, CA, C, O
	"""
	
	dict_coords = {}
	for i, atom in enumerate(["N", "CA", "C", "O"]):
		dict_coords[atom] = coords[:, i, :].cpu().numpy().tolist()
	
	return dict_coords


def dict2trans(dict_coords: dict, device="cpu"):
	"""
	Get rotation matrix and translation vector from dict of coordinates
	
	Args:
		dict_coords: dict, keys are N, CA, C, O
		device: torch.device

	Returns:
		coords: [N, L, 4, 3] tensor
		rots: [N, L, 3, 3] tensor
		t: [N, L, 3] tensor

	"""
	coords = dict2tensor(dict_coords, device).unsqueeze(dim=0)
	
	center = coords[..., 1, :]
	p1 = coords[..., 2, :]
	p2 = coords[..., 0, :]
	
	rots, trans = construct_3d_basis(center, p1, p2)
	return coords, rots, trans