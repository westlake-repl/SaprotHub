import yaml
from easydict import EasyDict
from pathlib import Path

current_file_path = Path(__file__).resolve()  # 获取当前脚本的绝对路径
current_dir_path = current_file_path.parent

with open(current_dir_path / 'default.yaml', 'r', encoding='utf-8') as r:
        Default_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'DeepLoc/cls2/saprot.yaml', 'r', encoding='utf-8') as r:
        DeepLoc_cls2_config = EasyDict(yaml.safe_load(r))
with open(current_dir_path / 'DeepLoc/cls10/saprot.yaml', 'r', encoding='utf-8') as r:
        DeepLoc_cls10_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'EC/saprot.yaml', 'r', encoding='utf-8') as r:
        EC_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'GO/BP/saprot.yaml', 'r', encoding='utf-8') as r:
        GO_BP_config = EasyDict(yaml.safe_load(r))
with open(current_dir_path / 'GO/CC/saprot.yaml', 'r', encoding='utf-8') as r:
        GO_CC_config = EasyDict(yaml.safe_load(r))
with open(current_dir_path / 'GO/MF/saprot.yaml', 'r', encoding='utf-8') as r:
        GO_MF_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'HumanPPI/saprot.yaml', 'r', encoding='utf-8') as r:
        HumanPPI_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'MetalIonBinding/saprot.yaml', 'r', encoding='utf-8') as r:
        MetalIonBinding_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'Thermostability/saprot.yaml', 'r', encoding='utf-8') as r:
        Thermostability_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'ClinVar/saprot.yaml', 'r', encoding='utf-8') as r:
        ClinVar_config = EasyDict(yaml.safe_load(r))

with open(current_dir_path / 'ProteinGym/saprot.yaml', 'r', encoding='utf-8') as r:
        ProteinGym_config = EasyDict(yaml.safe_load(r))
