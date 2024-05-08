import os
import json
import copy
import numpy as np


from tqdm import tqdm
from Bio import pairwise2
from Bio.PDB import PDBParser, FastMMCIFParser, Atom, Model, Structure, Chain, Residue
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from utils.mpr import MultipleProcessRunner


mmcif_parser = FastMMCIFParser(QUIET=True)
mmcif_io = MMCIFIO()
pdb_parser = PDBParser(QUIET=True)
pdb_io = PDBIO()
aa3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
          'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
          'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
          'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
aa1to3 = {v: k for k, v in aa3to1.items()}


def create_pdb_from_backbone(backbone_coords_dict: dict, output_file: str, residue_types: list = None):
    """
    Create a PDB file from the backbone coordinates
    Args:
        backbone_coords_dict: Keys are in ["N", "CA", "C", "O"], values are the list of coordinates
        output_file: Path to the output file
        residue_types: List of residue types, if None, all residues are set to be "ALA"

    """
    # Create a Structure object
    structure = Structure.Structure('backbone_structure')
    
    # Create a Model object within the structure
    model = Model.Model(0)
    
    # Create a Chain object within the model
    chain = Chain.Chain('A')
    
    # Get the length of the protein
    for value in backbone_coords_dict.values():
        length = len(value)
        break
    
    if residue_types is None:
        residue_types = ['ALA'] * length
    
    for i in range(length):
        residue = Residue.Residue((' ', i+1, ' '), residue_types[i], 1)
    
        # Create Atom objects for N, CA, C, and O with their coordinates
        for atom_name, coordinates in backbone_coords_dict.items():
            atom = Atom.Atom(atom_name, coordinates[i], 0, 0, ' ', atom_name, i+1, atom_name)
            atom.set_coord(coordinates[i])
            residue.add(atom)
        
        # Add the residue to the chain
        chain.add(residue)
    
    # Add the chain to the model
    model.add(chain)
    
    # Add the model to the structure
    structure.add(model)
    
    io = pdb_io if output_file.endswith("pdb") else mmcif_io
    
    # Save the structure to a PDB file
    io.set_structure(structure)
    io.save(output_file)
    

def is_residue_valid(residue, CA_only: bool = False) -> bool:
    """
    Check if the residue has all atoms (N, CA, C, O)
    Args:
        residue: Bio.PDB.Residue object.

        CA_only: If True, only check if the residue has CA atom.

    Returns:
        True if the residue has all atoms
    """
    atoms = ['N', 'CA', 'C', 'O'] if not CA_only else ['CA']
    
    res_name = residue.get_resname()
    if res_name not in aa3to1 or sum([0 if atom in residue else 1 for atom in atoms]) > 0:
        return False
    
    else:
        return True
    

# Refine a pdb structure given rotation matrices and translation vectors
def save_refined_pdb(path, format, chain, new_coords: np.ndarray, save_path=None):
    """
    
    Args:
        path: path to pdb file
        format: pdb or mmcif
        chain: chain id
        new_coords: [seq_len, 4, 3]. 4: N CA C O
        save_path: path to save refined pdb file

    """
    assert format in ['pdb', 'mmcif'], "Only support pdb and mmcif format"
    
    chain = get_structure(path, format, chain)
    atoms = ["N", "CA", "C", "O"]
    
    cnt = 0
    for residue in chain.get_residues():
        res_name = residue.get_resname()
        if res_name not in aa3to1:
            continue
        
        # Ensure that the residue has all atoms
        if sum([0 if atom in residue else 1 for atom in atoms]) > 0:
            continue
        
        atom_coords = new_coords[cnt]
        for atom, new_atom_coord in zip(residue, atom_coords):
            atom.set_coord(new_atom_coord)
            
        cnt += 1

    if format == 'pdb':
        pdb_io.set_structure(chain)
        pdb_io.save(save_path)
    else:
        mmcif_io.set_structure(chain)
        mmcif_io.save(save_path)


# Get the sequence and id dict of a chain
def get_seq(chain):
    seqs = []
    delete_id = []
    index2id = {}
    atoms = ['N', 'CA', 'C', 'O']
    residues = chain.get_residues()

    for residue in residues:
        res_name = residue.get_resname()
        if res_name not in aa3to1:
            delete_id.append(residue.get_id())
            continue

        # Ensure that the residue has all atoms
        if sum([0 if atom in residue else 1 for atom in atoms]) > 0:
            delete_id.append(residue.get_id())
            continue

        seqs.append(aa3to1[res_name])
        index2id[len(seqs) - 1] = residue.get_id()

    seq = "".join(seqs)
    return seq, index2id, delete_id


# Get the structure of a protein with a given chain
def get_chain(path, chain):
    _, file = os.path.split(path)
    name, format = os.path.splitext(file)
    assert format in ['.pdb', '.cif'], "Only support pdb and mmcif format"
    
    parser = pdb_parser if format == '.pdb' else mmcif_parser
    structure = parser.get_structure(name, path)
    return structure[0][chain]


# Align two chains and save the aligned structures as new pdb files.
def align_structure_output(chain1,
                           chain2,
                           save_path1=None,
                           save_path2=None,
                           plddt1=None,
                           plddt2=None,
                           plddt_save_path1=None,
                           plddt_save_path2=None,):
    """
    plddt file is for AF2 structures. We remove corresponding positions in plddt file.

    """
    seq1, index2id1, delete_id1 = get_seq(chain1)
    seq2, index2id2, delete_id2 = get_seq(chain2)

    alignments = pairwise2.align.globalxx(seq1, seq2)
    aligned_seq1, aligned_seq2, _, _, _ = alignments[0]

    seq1_pos, seq2_pos = 0, 0
    seq1_ignore, seq2_ignore = [], []

    for i, (aa_seq1, aa_seq2) in enumerate(zip(aligned_seq1, aligned_seq2)):
        if aa_seq1 != "-" and aa_seq2 != "-":
            seq1_pos += 1
            seq2_pos += 1

        elif aa_seq1 == "-":
            # Amino acid of seq2 matches a gap. Ignore this amino acid.
            seq2_ignore.append(seq2_pos)
            seq2_pos += 1

        else:
            # Amino acid of seq1 matches a gap. Ignore this amino acid.
            seq1_ignore.append(seq1_pos)
            seq1_pos += 1
    
    # Remove the ignored amino acids
    for i in [1, 2]:
        chain = eval(f"chain{i}")
        seq_ignore = eval(f"seq{i}_ignore")
        index2id = eval(f"index2id{i}")
        save_path = eval(f"save_path{i}")
        delete_id = eval(f"delete_id{i}")
        plddt_path = eval(f"plddt{i}")
        plddt_save_path = eval(f"plddt_save_path{i}")
        
        if plddt_path is not None:
            plddt = json.load(open(plddt_path, "r"))
            tmp_dict = {}
            for k, v in plddt.items():
                np_v = np.array(v)
                
                selector = np.ones(len(v), dtype=bool)
                selector[seq_ignore] = False
                np_v = np_v[selector]
                
                tmp_dict[k] = np_v.tolist()
            
            json.dump(tmp_dict, open(plddt_save_path, "w"))
        
        for index in seq_ignore:
            chain.detach_child(index2id[index])

        for id in delete_id:
            chain.detach_child(id)

        # Save the aligned structure
        if save_path is not None:
            io = pdb_io if save_path.endswith("pdb") else mmcif_io
            io.set_structure(chain)
            io.save(save_path)


# Align sequences of two parsed dicts, removing all mismatches
def align_structure(dict1, dict2):
    dict1 = copy.deepcopy(dict1)
    dict2 = copy.deepcopy(dict2)
    
    seq1 = dict1["seq"]
    seq2 = dict2["seq"]
    
    alignments = pairwise2.align.globalxx(seq1, seq2)
    aligned_seq1, aligned_seq2, _, _, _ = alignments[0]
    
    seq1_pos, seq2_pos = 0, 0
    seq1_ignore, seq2_ignore = [], []
    for i, (aa_seq1, aa_seq2) in enumerate(zip(aligned_seq1, aligned_seq2)):
        if aa_seq1 != "-" and aa_seq2 != "-":
            seq1_pos += 1
            seq2_pos += 1
        
        elif aa_seq1 == "-":
            # Amino acid of seq2 matches a gap. Ignore this amino acid.
            seq2_ignore.append(seq2_pos)
            seq2_pos += 1
        
        else:
            # Amino acid of seq1 matches a gap. Ignore this amino acid.
            seq1_ignore.append(seq1_pos)
            seq1_pos += 1
    
    return_dict = []
    for parsed_dict, seq_ignore in zip((dict1, dict2), (seq1_ignore, seq2_ignore)):
        np_seq = np.array(list(parsed_dict["seq"]))
        np_coords = {k: np.array(v) for k, v in parsed_dict["coords"].items()}
        
        selected = np.ones_like(np_seq).astype(bool)
        selected[seq_ignore] = False
        
        parsed_dict["seq"] = "".join(np_seq[selected])
        parsed_dict["coords"] = {k: v[selected].tolist() for k, v in np_coords.items()}
        
        return_dict.append(parsed_dict)

    return return_dict

    
def split_chain(in_path, out_path, chain_id, new_chain_id: str = None) -> None:
    """
    Split a chain from a pdb file and save it as a new pdb file.
    Args:
        in_path: Path to the input pdb file.
        out_path: Path to the output pdb file.
        chain_id: Chain id to be split.
        new_chain_id: New chain id. If None, the new chain id will be the same as the old chain id.
    """
    
    in_format = os.path.splitext(in_path)[1]
    out_format = os.path.splitext(out_path)[1]
    assert in_format in [".pdb", ".cif"], "Input format must be pdb or cif."
    assert out_format in [".pdb", ".cif"], "Output format must be pdb or cif."
    
    parser = pdb_parser if in_format == ".pdb" else mmcif_parser
    structure = parser.get_structure("input", in_path)
    io = pdb_io if out_format == ".pdb" else mmcif_io

    chain = structure[0][chain_id]
    if new_chain_id is not None:
        chain.id = new_chain_id
    io.set_structure(chain)
    io.save(out_path)


def get_chain_ids(path: str) -> list:
    """
    Get the chains of a pdb file.
    Args:
        path:  Path to the pdb file.
        format:  Format of the pdb file. Only support "pdb" and "mmcif".

    Returns:
        A list of chain ids.

    """
    
    _, file = os.path.split(path)
    name, format = os.path.splitext(file)
    assert format in ['.pdb', '.cif'], "Only support pdb and mmcif format"
    
    parser = pdb_parser if format == ".pdb" else mmcif_parser
    structure = parser.get_structure(name, path)
    
    return [chain.get_id() for chain in structure[0].get_chains()]


def extract_pdb_section(input_pdb,
                        output_pdb,
                        chain_id,
                        start_residue,
                        end_residue) -> None:
    """
    Extract a section of a pdb file.
    Args:
        input_pdb: Path to the input pdb file.
        output_pdb: Path to the output pdb file.
        chain_id: Chain id of the section to be extracted.
        start_residue: Start residue id of the section to be extracted. Starts from 1.
        end_residue: End residue id of the section to be extracted. Starts from 1.
    """

    in_format = input_pdb.split(".")[-1]
    parser = pdb_parser if in_format == "pdb" else mmcif_parser
    structure = parser.get_structure('input', input_pdb)

    # Iterate over the chains in the structure
    for chain in structure[0]:
        if chain.id == chain_id:
            # Create a list of residues to remove
            residues_to_remove = []
            cnt = 0
            for residue in chain:
                if cnt < start_residue - 1 or cnt > end_residue - 1:

                    residues_to_remove.append(residue.id)
                cnt += 1

            # Remove the residues
            for residue_id in residues_to_remove:
                chain.detach_child(residue_id)

            # Save the aligned structure
            out_format = output_pdb.split(".")[-1]
            io = pdb_io if out_format == "pdb" else mmcif_io
            io.set_structure(chain)
            io.save(output_pdb)

def remove_pdb_section(input_pdb, output_pdb, chain_id, remove_residues: list) -> None:
    """
    Remove residues from a pdb file.
    Args:
        input_pdb: Path to the input pdb file.
        output_pdb: Path to the output pdb file.
        chain_id: Chain id of the residues to be removed.
        remove_residues: A index list of residue ids to be removed. Starts from 1.
    """
    
    in_format = input_pdb.split(".")[-1]
    parser = pdb_parser if in_format == "pdb" else mmcif_parser
    structure = parser.get_structure('input', input_pdb)

    # Iterate over the chains in the structure
    for chain in structure[0]:
        if chain.id == chain_id:
            # Create a list of residues to remove
            remove_sets = set(remove_residues)
            residues_to_remove = []
            cnt = 1
            for residue in chain:
                if is_residue_valid(residue):
                    if cnt in remove_sets:
                        residues_to_remove.append(residue)
                    cnt += 1

            # Remove the residues
            for residue in residues_to_remove:
                chain.detach_child(residue.id)

            # Write the modified chain to the output file
            out_format = output_pdb.split(".")[-1]
            io = pdb_io if out_format == "pdb" else mmcif_io
            io.set_structure(chain)
            io.save(output_pdb)


def parse_structure(path, chains: list = None, CA_only: bool = False) -> dict:
    """
    Parse a pdb file into a list of dict.

    Args:
        path: Path to the pdb file.

        chains: A list of chains to be parsed. If None, all chains will be parsed.

        CA_only: If True, only CA atoms will be parsed.

    Returns:
        A dict of parsed chains. The keys are chain ids and the values are dicts of parsed chains.
            seq: Amino acid sequence of the chain.
            coords: A dict of coordinates of the chain. The keys are "N", "CA", "C", "O".
            name: Name of the pdb.
            chain: Chain ID.
    """
    _, file = os.path.split(path)
    name, format = os.path.splitext(file)

    assert format in ['.pdb', '.cif'], "Only support pdb and mmcif format"
    
    parser = pdb_parser if format == ".pdb" else mmcif_parser
    structure = parser.get_structure(name, path)

    parsed_dicts = {}
    chains = structure[0].get_chains() if chains is None else [structure[0][chain_id] for chain_id in chains]
    for chain in chains:
        residues = chain.get_residues()
        atoms = ['N', 'CA', 'C', 'O'] if not CA_only else ['CA']
        coords = {atom: [] for atom in atoms}

        seq = []
        for residue in residues:
            if is_residue_valid(residue, CA_only):
                res_name = residue.get_resname()
                seq.append(aa3to1[res_name])
                for atom in atoms:
                    coords[atom].append(residue[atom].get_coord().tolist())

        parsed_dict = {"name": name,
                       "chain": chain.get_id(),
                       "seq": "".join(seq),
                       "coords": coords}
        
        # Skip empty chains
        if len(parsed_dict["seq"]) != 0:
            parsed_dicts[chain.get_id()] = parsed_dict

    return parsed_dicts

    
class ProteinStructureParser(MultipleProcessRunner):
    def __init__(self, format, *args, **kwargs):
        """
        Parse protein files into coordinates and sequences using multiple processes.
        Supported file formats: pdb, mmcif

        Args:
            data: List of files. Each element Should be a tuple of (pdb, chains) to have specific chain parsed.
            path: Path to save the parsed data as jsonl file
            n_process: Number of processes to use
        """
        super().__init__(*args, **kwargs)
        self.format = format
        assert self.format in ['pdb', 'mmcif'], "Only support pdb and mmcif format"
    
    def _aggregate(self, final_path: str, sub_paths):
        with open(final_path, 'w') as w:
            for sub_path in sub_paths:
                with open(sub_path, 'r') as r:
                    for line in tqdm(r, f"Aggregating parsed {self.format}s..."):
                        w.write(line)
                
                os.remove(sub_path)
    
    def _target(self, process_id, data, sub_path, *args):
        with open(sub_path, 'w') as f:
            for i, (path, chains) in enumerate(data):
                try:
                    parsed_dicts = parse_structure(path, self.format, chains)
                    for parsed_dict in parsed_dicts:
                        parsed_str = json.dumps(parsed_dict)
                        f.write(parsed_str + '\n')
                    self.terminal_progress_bar(process_id,
                                               i + 1,
                                               len(data),
                                               f"Process{process_id}: parsing {self.format}s...")
                
                except Exception as e:
                    print(f"Error in {path}: {e} Skip it.")
                    continue
    
    def __len__(self):
        return len(self.data)
