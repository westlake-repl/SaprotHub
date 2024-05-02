import requests
import json
import os


from tqdm import tqdm
from easydict import EasyDict
from utils.mpr import MultipleProcessRunner


# Convert the pdb id to the uniprot id. For every matched uniprot id,
# we only select one of corresponding pdb chains to return.
def pdb2uniprot(pdb):
    """
    Args:
        pdb: pdb id.

    Returns:
        A list contains all matched uniprot ids and corresponding chains
    """
    response = requests.get(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}")
    res = EasyDict(json.loads(response.text))
    
    results = []
    keys = set()
    for uniprot, info in res[pdb].UniProt.items():
        for chain in info.mappings:
            if chain.chain_id not in keys:
                results.append((f"{pdb}_{chain.chain_id}", uniprot))
                keys.add(chain.chain_id)
    
    return results


# Convert the pdb id with specific chain to the uniprot id
def chain2uniprot(pdb, chain):
    response = requests.get(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}")
    res = EasyDict(json.loads(response.text))
    
    for uniprot, info in res[pdb].UniProt.items():
        for mappings in info.mappings:
            if mappings.chain_id == chain:
                return uniprot


class PdbMapper(MultipleProcessRunner):
    def __init__(self, data, save_path, n_process=1, **kwargs):
        super().__init__(data, save_path, n_process=n_process, **kwargs)

    def _aggregate(self, final_path: str, sub_paths):
        with open(final_path, "w") as w:
            w.write("pdb_chain\tuniprot\n")

            for sub_path in sub_paths:
                with open(sub_path, 'r') as r:
                    for line in tqdm(r, f"Aggregating {sub_path}..."):
                        w.write(line)
                os.remove(sub_path)

    def _target(self, process_id, data, sub_path, *args):
        with open(sub_path, "w") as w:
            for i, pdb_chain in enumerate(data):
                try:
                    pdb, chain = pdb_chain.split(".")
                    uniprot = pdb2uniprot(pdb, chain)
                    w.write(f"{pdb_chain}\t{uniprot}\n")
                    
                    self.terminal_progress_bar(process_id, i + 1, len(data), f"Process{process_id} Mapping PDBs...")
                
                except Exception as e:
                    print(f"Error: {e}. {pdb_chain} cannot be mapped!")

    def __len__(self):
        return len(self.data)