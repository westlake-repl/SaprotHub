## Quick start

1. Click the link to jump to the section you're interested in.
2. Follow the instruction and video to prepare your **task**, **model** and **dataset**.
3. Finish your task with only a few clicks!

| Instruction                                                  | Video                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <a href="#instruction-train">How to train your model</a>     | -[YouTube](https://www.youtube.com/watch?v=r42z1hvYKfw)<br />- [Bilibili](https://www.bilibili.com/video/BV1HDhHeTEmH/?spm_id_from=333.337.search-card.all.click&vd_source=a418185fadee73ac65d8fab69eee0b52) |
| <a href="#instruction-prediction">How to use model for classification/regression prediction</a> | -[YouTube](https://www.youtube.com/watch?v=N5VMBwM_ukQ)      |
| <a href="#instruction-mutational_effect_prediction">How to use model for mutational effect prediction</a> |                                                              |
| <a href="#instruction-inverse_folding_prediction">How to use model for inverse folding prediction</a> |                                                              |
| <a href="#instruction-contribute">How to contribute to SaprotHub</a> |                                                              |

## Overview

### Task <a name="overview-task"></a>

Different models are designed for different tasks, so it's essential to understand **which type your task belongs to**.

To view the full list of tasks supported by ColabSaprot, please refer to [task_list.md](https://github.com/westlake-repl/SaProtHub/blob/main/task_list.md).

#### Task type

Here are the task types and their description, so you can recognize your task type based on your task description and objectives.

For Classification and Regression prediction task:

1. Classification Task
2. Regression Task
3. Amino Acid Classification Task
4. Pair Classification Task
5. Pair Regression Task

For Zero-shot prediciton task:

1. <a href="#mutational_effect_prediction">Mutational effect prediction</a>
2. <a href="#inverse_folding_prediction">Inverse folding prediction</a>

#### Classification and Regression prediction task

<a href="#train">Train</a> a model based on SaProt and use it to make <a href="#prediction">prediction</a>.

| Task Type                                                    | Task Description                                             | Example                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Classification**<br />(Protein-level Classification)       | Classify protein sequences.                                  | - Fold Class Prediction<br />- Localization Prediction<br />- Function Prediction |
| **Regression**<br />(Protein-level Regression)               | Predict the value of some property of a protein sequence.    | - Thermal Stability Prediction<br />- Fluorescence Intensity Prediction <br />- Binding Affinity Prediction |
| **Amino Acid Classification**<br />(Residue-level Classification) | Classify the amino acids in a protein sequence.              | - Secondary Structure Prediction<br />- Binding Site Prediction <br />- Active Site Prediction |
| **Pair Classification**                                      | Predict if there is interaction between the two proteins.    | - Protein-Protein Interaction (PPI) Prediction<br />- Interaction Type Classification Disease<br />- Associated Interaction Prediction |
| **Pair Regression**                                          | Predict the ability of interaction between the two proteins. | - Interaction Strength Prediction<br />- Binding Free Energy Calculation<br />- Interaction Affinity Prediction |

#### Zero-shot prediciton task

Directly use SaProt (650M) to make prediction.

| Task Type                              | Task Description                                                                        | Example                                                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Mutational Effect Prediction** | Predict the mutational effect based on the wild type sequence and mutation information. | - Enzyme Activity Prediction<br />- Virus Fitness Prediction<br />- Driver Mutation Prediction |
| **Inverse Folding Prediction**   | Predict the residue sequence given the structure backbone.                              | - Enzyme Function Optimization<br />- Protein Stability Enhancement <br />- Protein Folding Prediction |

### Dataset <a name="overview-dataset"></a>

You can use your private data to train and predict. Below are the various data formats corresponding to different **data types**.

#### What is SA(Structure-aware) Sequence

We combine the residue and structure tokens at each residue site to create a **Structure-aware sequence** (SA sequence), merging both residue and structural information.

The structure tokens are generated by encoding the 3D structure of proteins using Foldseek.

<a href="#script-get_sa">Here</a> you can **convert your data into SA Sequence** format.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/SA_Sequence.png?raw=true" height="300" width="600px" align="center">

#### Data Type

1. Single AA Sequence
2. Single SA Sequence
3. Single UniProt ID
4. Single PDB/CIF Structure
5. Multiple AA Sequences
6. Multiple SA Sequences
7. Multiple UniProt IDs
8. Multiple PDB/CIF Structures
9. SaprotHub Dataset

For tasks that require **two protein sequences as input** (pair classification & pair regression) :

1. A pair of AA Sequences
2. A pair of SA Sequences
3. A pair of UniProt IDs
4. A pair of PDB/CIF Structures
5. Multiple pairs of AA Sequences
6. Multiple pairs of SA Sequences
7. Multiple pairs of UniProt IDs
8. Multiple pairs of PDB/CIF Structures

#### How to find a SaprotHub Dataset

1. Go to [Official SaProtHub Repository](https://huggingface.co/SaProtHub) to find some datasets.
2. Copy the `Dataset ID` for future use.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Huggingface_ID.png?raw=true" height="200" width="700px" align="center">

#### Scripts for dataset preparation

|                                                              | Link                                     |
| ------------------------------------------------------------ | ---------------------------------------- |
| Get Structure-Aware Sequence                                 | <a href="#script-get_sa">here</a>        |
| Convert .fa file to .csv dataset (data type:`Multiple AA sequences`) | <a href="#script-fa2aa">here</a>         |
| Randomly split your dataset                                  | <a href="#script-split_dataset">here</a> |

### Model <a name="overview-model"></a>

#### Model type

1. Official pretrained SaProt (35M)
2. Official pretrained SaProt (650M)
3. Trained by yourself on ColabSaprot
4. Shared by peers on SaprotHub
5. Saved in your local computer
6. Multi-model on SaprotHub

| Model type                           | Used for                         | Description                                                  | Input                                   |
| ------------------------------------ | -------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| `Official pretrained SaProt (35M)`   | Training                         | Train a protein language model based on SaProt(35M) with your dataset | -                                       |
| `Official pretrained SaProt (650M)`  | Training                         | Train a protein language model based on SaProt(650M) with your dataset | -                                       |
| `Trained by yourself on ColabSaprot` | Continually training, Prediction | Once you have completed training the model, select this option to use the model you have trained on ColabSaprot for continual training or prediction | Select the model from the dropdown menu |
| `Shared by peers on SaprotHub`       | Continually training, Prediction | Use models shared on [SaprotHub](https://huggingface.co/SaProtHub) for continual training or prediction | Enter the model ID                      |
| `Saved in your local computer`       | Continually training, Prediction | Use models saved on your local computer (.zip file which were saved when finishing training) for continual training or prediction | Upload the .zip file                    |
| `Multi-models on SaprotHub`          | Prediction                       | Ensemble multiple models shared on [SaprotHub](https://huggingface.co/SaProtHub) for prediction<br />Each sample will be predicted using multiple models. <br /><font color="red"> **Note that:**</font> For classification tasks, voting will be used to determine the final predicted category; for regression tasks, the predicted values from each model will be averaged. | Enter the model IDs                     |

#### How to find a model on SaprotHub

1. Go to [Official SaProtHub Repository](https://huggingface.co/SaProtHub) to find some model based on your requirements.
2. Copy the `Model ID` for future use.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/train/model/SaprotHubModel.png?raw=true" align="center">

---



## How to train your model <a name="instruction-train"></a>

For classification or regression task, you can **train** your model based on SaProt, or **continually train** a SaprotHub model (trained on ColabSaprot)

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/r42z1hvYKfw/0.jpg)](https://www.youtube.com/watch?v=r42z1hvYKfw)

### Task type

1. Classification Task
2. Regression Task
3. Amino Acid Classification Task
4. Pair Classification Task
5. Pair Regression Task

### Base model

Click <a href="#overview-model">here</a> for detailed information on each model type.

1. Official pretrained SaProt (35M)
2. Official pretrained SaProt (650M)
3. Trained by yourself on ColabSaprot
4. Shared by peers on SaprotHub
5. Saved in your local computer

### Training dataset

Dataset should be a .csv file with three required columns: `sequence`, `label` and `stage`

- The content of column `sequence` depends on your **data type**. See the table
- The content of column `label` depends on your **task type**. See the table
- The column `stage` indicate whether the sample is used for training, validation, or testing. Ensure your dataset includes samples for all three stages. The values are: `train`, `valid`, `test`.

| Data type                       | Interface         | Input                                                                                                                                                         | Example                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Multiple AA Sequences`       | An upload button  | `file`: the .csv file                                                                                                                                       | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/train/dataset/aa.png?raw=true" height="200" width="700px" align="center">                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `Multiple SA Sequences`       | An upload button  | `file`: the .csv file                                                                                                                                       | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/train/dataset/sa.png?raw=true" height="200" width="700px" align="center">                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `Multiple UniProt IDs`        | An upload button  | `file`: the .csv file                                                                                                                                       | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/train/dataset/uniprot_id.png?raw=true" height="250" width="700px" align="center">                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `Multiple PDB/CIF Structures` | Two upload button | `file`: a .csv file containing three columns: `Sqeuence`, `type` and `chain<br />sturcture files`: a .zip file containing all the structure files | `type`: Indicate whether the structure file is a real PDB structure or an AlphaFold 2 predicted structure. For AF2 (AlphaFold 2) structures, we will apply pLDDT masking. The value must be either "PDB" or "AF2".<br />`chain`: For real PDB structures, since multiple chains may exist in one .pdb file, it is necessary to specify which chain is used. For AF2 structures, the chain is assumed to be A by default.<br /><img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/train/dataset/pdb.png?raw=true" height="150" width="700px" align="center"> |
| `SaprotHub Dataset`           | An input box      | `Dataset ID`: SaprotHub Dataset ID                                                                                                                          | Find more datasets on[SaprotHub](https://huggingface.co/SaProtHub)<br /><img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Huggingface_ID.png?raw=true" height="100" width="700px" align="center">                                                                                                                                                                                                                                                                                                                                                                 |

Example of comlum `label` for different task type (the data type in these examples is `Multiple SA sequences`)

| Task type                    | Label                                          | Example                                                      | Description                                                  |
| ---------------------------- | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Protein-level classification | Category index starting from zero              | <img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/train/dataset/label-cls.png?raw=true" height="100" width="700px" align="center"> | - The task have 2 protein sequence categories: 0, 1.<br />- Each protein sequence has a corresponding category index. |
| Protein-level regression     | Numerical values                               | <img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/train/dataset/label-regr.png?raw=true" height="100" width="700px" align="center"> | - Each protein sequence has a corresponding numerical label to represent the value of some property. |
| Residue-level classification | A list of category indices for each amino acid | <img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/train/dataset/label-aa_cls.png?raw=true" height="100" width="700px" align="center"> | - The task have 3 animo acid categories: 0, 1, 2.<br />- Each animo acid has a corresponding category index. |

### Training config

| Training config | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `batch_size`    | `batch_size` depends on the number of training samples. "Adaptive" (default choice) refers to automatic batch size according to your data size. <br />If your training data set is large enough, you can use 32, 64, 128, 256, ..., others can be set to 8, 4, 2 (Note that you can not use a larger batch size if you use the Colab default T4 GPU. <br /><font color="red"> **Note that:**</font> Strongly suggest you subscribe to Colab Pro for an A100 GPU.). |
| `max_epochs`    | `max_epochs`  refers to the maximum number of training iterations. A larger value needs more training time. The best model will be saved after each iteration. You can adjust `max_epochs` to control training duration. <br /><font color="red"> **Note that:**</font> The max running time of colab is 12hrs for unsubscribed user or 24hrs for Colab Pro+ user |
| `learning_rate` | `learning_rate` affects the convergence speed of the model. Through experimentation, we have found that `5.0e-4` is a good default value for base model `Official pretrained SaProt (650M)` and `1.0e-3` for `Official pretrained SaProt (35M)`. |

<font color="red"> **Note that:**</font> You can expand the code cell to adjust `GPU_batch_size` and `accumulate_grad_batches` to control the number of samples used for each training step. If you do this, the `batch_size` selected in the dropdown menu will be overridden.



### Upload model

You can upload the model to your Huggingface repository and then contribute it to SaprotHub.



You need to add some description for your model:

- `name`: The name of your model.
- `description`: The description of your model (which task is your model used for).
- `label_meanings`: For classification model, please provide detailed information about the meanings of all labels; for regression model, please provide the numerical range of the value.



For example, in a Subcellular Localization Classification Task with 10 categories, label=0 means the protein is located in the Nucleus, label=1 means the protein is located in the Cytoplasm, and so on. The information should be provided as follows:

`Nucleus, Cytoplasm, Extracellular, Mitochondrion, Cell.membrane, Endoplasmic.reticulum, Plastid, Golgi.apparatus, Lysosome/Vacuole, Peroxisome`



You can also edit the model card (readme.md) to provide more information such as `Dataset description`, `Performance` and so on.






### Instruction

Step 1

Complete the input and selection of Task Configs

- `task_name` is the name of the training task you're working on.
- `task_objective` describes the goal of your task, like sorting protein sequences into categories or predicting the values of some protein properties.
- `base_model` is the base model you use for training. By default, it's set to the officially pretrained SaProt, but you can use models either retrained (by yourself) by ColabSaprot or shared on [SaprotHub](https://huggingface.co/SaProtHub). For example, you can choose `Trained-by-peers` with your own data if you want to retrain on SaProt models shared by others.  There are a wide range of retrained models available on [SaprotHub](https://huggingface.co/SaProtHub).
- `data_type` indicates the kind of data you're using, which is determined by the dataset file you upload. You can find more details about the formats for different types of data in the provided <a href="#data_format">instruction</a>.

Step 2

Click the run button to apply the configs.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/train-1.png?raw=true" height="300" width="600px" align="center">

Step 3

After clicking the "Run" button, additional input boxes will appear.

Complete the input of additional information and upload files.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/train-2.png?raw=true" height="300" width="400px" align="center">

(Note: Do not click the "Run" button of the next cell before completing the input and upload.)

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/train-3.png?raw=true" height="300" width="300px" align="center">

Step 4

Complete the input of training configs

- `batch_size` depends on the number of training samples. If your training data set is large enough, we recommend using 32, 64,128,256, ..., others can be set to 8, 4, 2. (Note that you can not use a larger batch size if you the Colab default T4 GPU. Strongly suggest you subscribe to Colab Pro for an A100 GPU.)
- `max_epochs` refers to the maximum number of training iterations. A larger value needs more training time. The best model will be saved after each iteration. You can adjust `max_epochs` to control training duration. (Note that the max running time of Colab is 12hrs for unsubscribed user or 24hrs for Colab Pro+ user)
- `learning_rate` affects the convergence speed of the model. Through experimentation, we have found that `5.0e-4` is a good default value for base model `Official pretrained SaProt (650M)` and `1.0e-3` for `Official pretrained SaProt (35M)`.

Step 5

Click the "Run" button to start training.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/train-4.png?raw=true" height="300" width="400px" align="center">

You can monitor the training process by these plots. After training, check the training results and the saved model.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/train-5.png?raw=true" height="300" width="400px" align="center">

---



## How to use model for classification/regression prediction <a name="instruction-prediction"></a>

### Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/N5VMBwM_ukQ/0.jpg)](https://www.youtube.com/watch?v=N5VMBwM_ukQ)

### Task type

1. Classification Task
2. Regression Task
3. Amino Acid Classification Task
4. Pair Classification Task
5. Pair Regression Task

### Model

Click <a href="#overview-model">here</a> for detailed information on each model type.

1. Trained by yourself on ColabSaprot
2. Shared by peers on SaprotHub
3. Saved in your local computer
4. Multi-model on SaprotHub

### Dataset

| Data type                       | Interface                          | Input                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Example                                                                                                                                                                                                                |
| ------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Single AA Sequence`          | An input box                       | `sequence`: the amino acid sequence                                                                                                                                                                                                                                                                                                                                                                                                                                              | `sequence`: MEETMKLATM |
| `Single SA Sequence`          | An input box                       | `sequence`: the structure-aware sequence                                                                                                                                                                                                                                                                                                                                                                                                                                         | `sequence`: MdEvEvTvMpKpLpApTaMp |
| `Single UniProt ID`           | An input box                       | `sequence`: the UniProt ID                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `sequence`: O95905 |
| `Single PDB/CIF structure`    | Two input box and an upload button | `type`: Indicate whether the structure file is a real PDB structure or an AlphaFold 2 predicted structure. For AF2 (AlphaFold 2) structures, we will apply pLDDT masking. The value must be either "PDB" or "AF2".<br />`chain`: For real PDB structures, since multiple chains may exist in one .pdb file, it is necessary to specify which chain is used. For AF2 structures, the chain is assumed to be A by default.<br />`structure file`: the .pdb/.cif structure file | `type`: AF2<br />`chain`: A<br />`structure file`: O95905.pdb |
| `Multiple AA Sequences`       | An upload button                   | `file`: the .csv file                                                                                                                                                                                                                                                                                                                                                                                                                                                            | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/prediction/dataset/aa.png?raw=true" height="200" width="700px" align="center">                                                               |
| `Multiple SA Sequences`       | An upload button                   | `file`: the .csv file                                                                                                                                                                                                                                                                                                                                                                                                                                                            | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/prediction/dataset/sa.png?raw=true" height="200" width="700px" align="center">                                                               |
| `Multiple UniProt IDs`        | An upload button                   | `file`: the .csv file                                                                                                                                                                                                                                                                                                                                                                                                                                                            | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/prediction/dataset/uniprot_id.png?raw=true" height="200" width="700px" align="center">                                                       |
| `Multiple PDB/CIF Structures` | Two upload button                  | `file`: a .csv file containing three columns: `Sqeuence`, `type` and `chain`<br />`structure files`: a .zip file containing all the structure files                                                                                                                                                                                                                                                                                                                    | <img src="https://github.com/westlake-repl/SaprotHub/blob/main/Figure/prediction/dataset/pdb.png?raw=true" height="100" width="700px" align="center">                                                              |
| `SaprotHub Dataset`           | An input box                       | `Dataset ID`: SaprotHub Dataset ID                                                                                                                                                                                                                                                                                                                                                                                                                                               | Find more datasets on [SaprotHub](https://huggingface.co/SaProtHub)<br /><img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Huggingface_ID.png?raw=true" height="100" width="700px" align="center"> |

### Instruction

Step 1

Complete the input and selection of Task Configs, and then

- `task_objective` describes the goal of your task, like sorting protein sequences into categories or predicting the values of some protein properties.
- `use_model_from` depends on whether you want to use a local model or a Huggingface model. If you choose `Shared by peers on SaprotHub`, please enter the Hugging Face model ID in the input box. If you choose `Local Model`, simply select your local model from the options. Additionally, there's a wide range of models available on SaprotHub.
- `data_type` indicates the kind of data you're using, which determines the dataset file you should upload. You can find more details about the formats for different types of data in the provided <a href="#data_format">instruction</a>.

Step 2

Click the run button to apply the configs.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/cls_regr-1.png?raw=true" height="300" width="500px" align="center">

Step 3

After clicking the "Run" button, additional input boxes and upload button will appear.

Complete the input of additional information and upload files.

(Note: Do not click the "Run" button of the next cell before completing the input and upload.)

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/cls_regr-2.png?raw=true" height="300" width="400px" align="center">

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/cls_regr-3.png?raw=true" height="300" width="400px" align="center">

Step 4

Click the run button to start predicting. Check your results after finishing prediction.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/cls_regr-4.png?raw=true" height="300" width="500px" align="center">

---



## How to use model for mutational effect prediction <a name="instruction-mutational_effect_prediction"></a>

### Mutation Task

- Single-site or Multi-site mutagenesis
- Saturation mutagenesis

### Model

Default model is `Official pretrained SaProt (650M)`.

### Mutation information

Here is the detail about the representation of **mutation information**: <a name="mutation_info"></a>

| mode                    | mutation information   |
| ----------------------- | ---------------------- |
| Single-site mutagenesis | H87Y                   |
| Multi-site mutagenesis  | H87Y:V162M:P179L:P179R |

- For `Single-site mutagenesis`, we use a term like "H87Y" to denote the mutation, where the first letter represents the **original amino acid**, the number in the middle represents the **mutation site** (indexed starting from 1), and the last letter represents the **mutated amino acid**,
- For `Multi-site mutagenesis`, we use a colon ":" to connect each single-site mutations, such as "H87Y:V162M:P179L:P179R".

### Mutation dataset

- For `Saturation mutagenesis`, the mutation dataset is the same as <a href="#instruction-prediction">the dataset used for classification/regression prediction tasks</a>.
- For `Single-site or Multi-site mutagenesis`, **one more information** are required: `mutation`.

| Data type                       | Interface                            | Input                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Example                                                                                                                                                                                                                |
| ------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Single SA Sequence`          | Two input box                        | `sequence`: the structure-aware sequence<br />`mutation`: the mutation information                                                                                                                                                                                                                                                                                                                                                                                                                                        | `sequence`: MdEvEvTvMpKpLpAp<br />`mutation`: M1H:E2L:E3Q:T4A:M5P:K6Y:L7V:A8P |
| `Single UniProt ID`           | Two input box                        | `sequence`: the UniProt ID<br />`mutation`: the mutation information                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `sequence`: O95905<br />`mutation`: H87Y:V162M:P179L |
| `Single PDB/CIF structure`    | Three input box and an upload button | `type`: Indicate whether the structure file is a real PDB structure or an AlphaFold 2 predicted structure. For AF2 (AlphaFold 2) structures, we will apply pLDDT masking. The value must be either "PDB" or "AF2".<br />`chain`: For real PDB structures, since multiple chains may exist in one .pdb file, it is necessary to specify which chain is used. For AF2 structures, the chain is assumed to be A by default.<br />`structure file`: the .pdb/.cif structure file<br />`mutation`: the mutation information | `type`: AF2<br />`chain`: A<br />`structure file`: O95905.pdb<br />`mutation`: H87Y:V162M:P179L |
| `Multiple SA Sequences`       | An upload button                     | `file`: the .csv file containing two columns: `sequence` and `mutation`                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/mutational_effect_prediction/dataset/sa.png?raw=true" height="100" width="700px" align="center"> |
| `Multiple UniProt IDs`        | An upload button                     | `file`: the .csv file containing two columns: `sequence` and `mutation`                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/mutational_effect_prediction/dataset/uniprot_id.png?raw=true" height="100" width="700px" align="center"> |
| `Multiple PDB/CIF Structures` | Two upload button                    | `file`: a .csv file containing four columns: `Sqeuence`, `type`, `chain` and `mutation`<br />`structure files`: a .zip file containing all the structure files                                                                                                                                                                                                                                                                                                                                                   | <img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/mutational_effect_prediction/dataset/pdb.png?raw=true" height="50" width="1000px" align="center"> |

### Instruction

Step 1

Complete the selection of Task Configs.

- `mutation_task` indicates the type of mutation task. You can choose from `Single-site or Multi-site mutagenesis` and `Saturation mutagenesis`.
- `data_type` indicates the kind of data you're using, which determines the dataset file you should upload. You can find more details about the formats for different types of data in the provided <a href="#data_format">instruction</a>.

Step 2

Click the run button to apply the configs.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/mep-1.png?raw=true" height="300" width="800px" align="center">

Step 3

After clicking the "Run" button, additional input boxes and upload button will appear.

For a single sequence, enter the sequence and the mutation information into the corresponding input fields. (Note that for Saturation mutagenesis, you won't see the Mutation input box.)

For multiple sequences, click the upload button to upload your dataset. (Note that for Saturation mutagenesis, you don’t need to provide mutation information in your dataset, which means only `sequence` column is required in the .csv dataset.)

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/mep-2.png?raw=true" height="300" width="800px" align="center">

Step 4

Click the run button to start predicting. Check your results after finishing prediction.

- For a single sequence, the predicted score will be show in the output.
- For multiple sequences, the predicted score will be saved in a .csv file.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/mep-3.png?raw=true" height="300" width="600px" align="center">

![Untitled](https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/mutation-3-3.png?raw=true)

![Untitled](https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/mutation-3-4.png?raw=true)

## How to use model for inverse folding prediction <a name="instruction-inverse_folding_prediction"></a>

### Task config

- `method` refers to the prediction method. It could be either `argmax` or `multinomial`.
  - `argmax` selects the amino acid with the highest probability.
  - `multinomial` samples an amino acid from the multinomial distribution.
- `num_samples` refers to the number of output amino acid sequences.

### Model

Default model is `Official pretrained SaProt (650M)`.

### Inverse folding dataset

PDB/CIF file

### After generating the sequence

#### Predict the structure of generated sequence

#### Align proteins using TMalign

### Instruction

Step 1

Click the run button to upload the structure file, which could be in the format of .pdb or .cif file.

Step 2

After clicking the "Run" button, additional input boxes and upload button will appear.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/if-1.png?raw=true" height="300" width="500px" align="center">

Step 3

After uploading the structure file, it will be transformed into AA sequence and structure sequence.

Use '#' to mask some amino acids for prediction.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/if-2.png?raw=true" height="300" width="800px" align="center">

Step 4

Choose the prediction method.

Step 5

Click the run button to make prediction.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/if-3.png?raw=true" height="300" width="1000px" align="center">

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/Instruction/v1/if-4.png?raw=true" height="300" width="600px" align="center">

## How to contribute to SaprotHub <a name="instruction-contribute"></a>



### Join SaprotHub Organization

Before contributing to SaprotHub, you need to join the SaprotHub Huggingface Organization to gain write access to the subset of repos within the Organization that you have created.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/contribute/join.png?raw=true" align="center">



### Contribute to SaprotHub

You have two ways to contribute to SaprotHub:

1. Transfer your model to SaprotHub (Recommended)
2. Create a new model repository and upload model files



**Transfer your model to SaprotHub (Recommended)**

Once you have uploaded the model to your Huggingface repository using ColabSaprot, you can directly transfer your model to SaprotHub.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/contribute/transfer.png?raw=true" align="center">



**Create a new model repository and upload model files**

You can manually create a new model repository on SaprotHub, and then upload the model files to this repository.

<img src="https://github.com/westlake-repl/SaProtHub/blob/main/Figure/contribute/create.png?raw=true" align="center">
