# SaprotHub: Making Protein Modeling Accessible to All Biologists
<a href="https://www.biorxiv.org/content/10.1101/2024.05.24.595648v3"><img src="https://img.shields.io/badge/Paper-bioRxiv-green" style="max-width: 100%;"></a>
<a href="https://huggingface.co/SaProtHub"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red?label=SaprotHub" style="max-width: 100%;"></a>
<a href="https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub.ipynb"><img src="Figure/colab-badge.svg" style="max-width: 100%;"></a>
<a href="https://cbirt.net/no-coding-required-saprothub-brings-protein-modeling-to-every-biologist/" alt="blog"><img src="https://img.shields.io/badge/Blog-Medium-purple" /></a> 
<a href="https://x.com/sokrypton/status/1795525127653986415"><img src="https://img.shields.io/badge/Twitter-blue?logo=twitter" style="max-width: 100%;"></a>

The repository is an official implementation of [SaprotHub: Making Protein Modeling Accessible to All Biologists](https://www.biorxiv.org/content/10.1101/2024.05.24.595648v4).

We are pleased to announce that [ColabSaprot](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub.ipynb) and [SaprotHub](https://huggingface.co/SaProtHub) are now ready for use.

## Open Protein Modeling Consortium (OPMC)
The Open Protein Modeling Consortium (OPMC) is a collaborative initiative designed to unify the efforts of the protein research community. Its mission is to facilitate the sharing and co-construction of resources, with a particular focus on individually trained decentralized models, thereby advancing protein modeling through collective contributions. OPMC offers a platform that supports a wide range of protein  predictions, aiming to make advanced protein modeling accessible to researchers irrespective of their machine learning expertise.

Join us as an OPMC author [here](https://github.com/westlake-repl/SaprotHub/blob/main/Figure/OPMC.jpg) 
visit our OPMC wetbsite  [here](https://theopmc.github.io/)

### ColabPLM of OPMC
- [ColabSaprot](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub.ipynb) (AA+3Di)
- [ColabProTrek](https://colab.research.google.com/drive/1On2xQU0d7351bIBgZpz2T0VUp2gZium0?usp=sharing) (AA/3Di align 3Di/AA)
- [ColabProtT5](https://colab.research.google.com/drive/1agJCwW8EXyB0xmY5xjbR9iiNE5JAPMOo?usp=sharing)
### OPMC authors (senior)
- Sergey Ovchinnikov, MIT 
- Martin Steinegger, Seoul National University 
- Kevin Yang, Microsoft
- Michael Heinzinger, Technische Universität München
- Pascal Notin, Harvard University
- Pranam Chatterjee, Duke University
- Jia Zheng, Westlake University
- Stan Z. Li, Westlake University
- Xing Chang, Westlake University
- Huaizong Shen, Westlake University
- Noelia Ferruz, The Centre for Genomic Regulation (CRG)
- Rohit Singh, Duke University
- Debora Marks, Harvard University
- Anping Zeng, Westlake University
- Jijie Chai, Westlake University
- Anthony Gitter, University of Wisconsin-Madison
- Anum Glasgow, Columbia University
- Milot Mirdita, Seoul National University
- Philip M. Kim, University of Toronto     
- Christopher Snow, Colorado State University   
- Vasilis Ntranos, University of California 
- Philip A. Romero,  Duke University
- Jianyi Yang, Shandong University
- Caixia Gao, Chinese Academy of Sciences
- Michael Bronstein, University of Oxford




## Deploy ColabSaprot on local server (for linux os and Windows)
For users who want to deploy ColabSaprot on their local server, please refer to [here](https://github.com/westlake-repl/SaprotHub/blob/main/local_server).

## FAQs
### Q1: It seems like OPMC and SaprotHub are intertwined but not exactly the same.
Yes, OPMC is a grand goal, and in this paper, it is primarily presented as a concept and vision. The paper introduces 
OPMC and implements SaprotHub as a pioneering example to drive the initial realization of OPMC. 
Achieving a broader implementation of OPMC requires continuous efforts from the entire community.

### Q2:  I'm very interested in the OPMC side of this project? Would I be able to support OMPC independently?
Yes, you can. OPMC is not tied exclusively to SaprotHub. SaprotHub serves as an initial implementation case within the
broader OPMC concept. We also welcome the inclusion of new protein models in OPMC. There are generally two ways to
contribute: either independently of SaprotHub, such as building ESMHub or ProtTransHub, or by joining SaprotHub. While
SaprotHub is named after its first model, Saprot, it is not limited to Saprot alone and welcomes the inclusion of 
other  language models. The concept of OPMC originated from the SaprotHub paper, so if you would like your protein 
model to be part of OPMC or if you adopt the similar construction approach of SaprotHub, we encourage you to cite the
source paper. Also see Q9.

### Q3: What's the relation between OPMC and the OpenFold Consortium?
The goal of the OpenFold Consortium is to develop free and open-source software tools. This differs from the goals of
OPMC. OPMC aims to make it easy for all biologists (especially those without machine learning backgrounds and coding
skills) to train their own protein models, and to share these models with the community members, allowing for 
integration and collaborative development on top of the existing community models.

Additionally, so far, the OpenFold Consortium seems to be focusing more on protein structure prediction, while OPMC
is more focused on protein function prediction. Furthermore, the number of protein function task categories is far 
greater than the number of structure tasks. As a result, biologists often have to fine-tune large pre-trained protein 
models based on their own training data, which is a key feature of OPMC.

### Q4: Is the idea to create a company that provides the resources for biologists to do model training? I'm unsure the vision here, since a lot of model training is resource and data constrained. It would be hard to create something where "every biologist to train their own AI models with just a few clicks." Who would provide the resources in this case?
No, the primary motivation behind OPMC is to enable biologists to participate in protein model training and 
collaborative development, without direct involvement of creating a company or commercial operation.

Currently, we do not provide free training resources. Users have the option to purchase GPUs, such as the A100, on 
platforms like Colab. OPMC primarily supports fine-tuning tasks or direct prediction tasks for protein language models,
rather than pre-training. These tasks typically do not require excessively expensive computational power. With a 
budget of around $10, one can easily complete training and prediction tasks on several thousand samples. This cost 
is manageable for most individuals and academic institutions. There are also free GPU resources in Colab but they 
may not be sufficient for some of your tasks.

In the future, we may explore options such as applying for funding or accepting donations to provide some free 
computational resources to users. Please note that the purpose of SaprotHub and OPMC is not to provide free computational
resources. 

### Q5: We are open sourcing models as well, so it would be interesting to collaborate once we release these models. However, it seems like currently the hub is geared towards SaProt as the main model of choice.
Saprot is the first model to join the hub, so we named it SaprotHub. However, SaprotHub can also accept other protein
models, such as ESM. Of course, you can also independently develop your own model hub. In the future, we will create
a webpage for OPMC that will include all the participating models.

The reason we adopted the Saprot model is that it is a near-universal model, capable of supporting any protein and 
residue-level prediction task, including regression, classification, ranking, as well as zero-shot mutational effect
prediction and sequence design tasks. Saprot is also the state-of-the-art protein language model in the community.

Additionally, we also hope to include as many other protein language models as possible, but due to limited human
resources, we are unable to integrate all the existing protein language models. This is precisely the purpose of 
building a community (the development of ColabSaprot took us approximately 4 months. Of course, with the open-sourcing
of ColabSaprot, it will be much easier to implement similar functionality for other protein language models).

We believe that with the joint efforts of the entire community, the OPMC community store can become more diverse, 
and biologists can choose the models that best suit their needs. Therefore, we sincerely invite researchers who are
interested in OPMC to join us, and if you have better suggestions, we welcome you to join us in co-building OPMC.

### Q6: ESM, AlphaFold, and Openfold models are not mentioned in the hub.
OPMC mainly focuses on protein function prediction. So ESM is a good fit, AlphaFold and OpenFold target at protein 
structure prediction, and could be independent of OPMC or SaprotHub. But please note that developing another ColabESM 
will also take some time, and we welcome researchers to integrate the ESM or other models onto the hub.

### Q7: How do you foresee collaborating and sharing other open models?
The SaprotHub paper primarily focuses on collaboration and sharing within the framework of one backbone model, as this
can greatly reduce storage and communication costs by leveraging Adapter technique – users just need to operate on the
adapters rather than the large backbone model. Since these models are based on the same backbone network, the input
format, network and parameter interfaces are more consistent, which serves as the basic for community sharing, 
collaboration and co-construction.

As for sharing between different backbone models, this remains a challenge at the moment, although it is an 
interesting research direction without an ideal solution yet. For example, if users want to collaborate between 
ESM15B and Saprot650M, they would need to upload and download the two complete models, significantly increasing 
communication and maintenance costs. As the number of models integrated increases, the demand on GPU performance 
would also increase rapidly. 

However, by using the same backbone model and the Adapter mechanism, these difficulties can be elegantly solved. The
Adapters in SaprotHub is just like grafting techniques in biology. Just as a single tree can bear different kinds of 
fruit, Saprot acts like the trunk, and the adapters for various downstream tasks resemble the different fruits on the
Saprot tree.

Additionally, due to the differences in model architecture, input, and output across different models, it is difficult
to design a unified interface. Therefore, this paper serves as an initial exploration of OPMC and does not cover 
collaboration between different backbone models. This may require more effort from the community, but holds promise 
for the future.

### Q8: How does this differentiate SaprotHub from Hugging Face?
SaprotHub primarily focuses on storing lightweight Adapters, whereas Hugging Face stores complete pre-trained models.
SaprotHub adopts the Adapter mechanism, which enables biologists to easily share, co-develop, and collaborate within
ColabSaprot.

The goal of SaprotHub is to allow all biologists to train their own protein models even without machine learning and 
coding background, while Hugging Face's objective is to open-source the model weights without considering the easy 
training aspect.

SaprotHub is dedicated to establishing the AI model community for proteins, while Hugging Face has a broader scope.
Therefore, SaprotHub's Adapter store can be built on top of Hugging Face or developed independently.


### Q9: If I develop other protein language models (PLMs) and online platforms following SaprotHub and ColabSaprot, can I be an author on the SaprotHub paper?

All OPMC members will be listed as authors in the SaprotHub paper before its final revision, which is expected to take 4-12 months. Author's name will be included in the paper. Please note that eligibility to become an OPMC regular member is determined by the steering committee.

If an OPMC author is primarily granted authorship recognition for developing other PLMHubs, they are required to acknowledge that this model automatically becomes part of the OPMC framework.
In case they publish a paper, they should menton this somewhere in the paper. Researchers using this new PLMHub should also cite the original OPMC literature.

### Q10: How to be a member of OPMC or an author of SaprotHub.

Before the final revision of SaprotHub, all OPMC members will be automatically included as authors. However, after the publication of SaprotHub, individuals can still join OPMC but will not be able to be listed as authors in the paper, as it is subject to the requirements of the journal publication - the final version needs to determine the author list.

Regarding how to join OPMC, please refer to: [here](https://github.com/westlake-repl/SaprotHub/blob/main/Figure/OPMC.jpg). 

All in all, if you can come up with some cool ideas or novel ways to enhance the impact and influence of Saprot, ColabSaprot, SaprotHub, or OPMC, you may have the opportunity to be listed as an author.

### Q11: If I have made a lot of valuable contributions to OPMC, can I become an OPMC member together with my supervisor?

Yes. If you have made significant valuable contributions to OPMC, it is possible to become an OPMC member alongside your PhD/postgraduate or internship supervisor. However, the acceptance of your contributions and membership status will be determined by the steering committee. Generally, in such cases, you would need to demonstrate more substantial contributions compared to regular OPMC members. You need to provide official documentation of your relationship with your supervisor, such as an official letter or document confirming their role and support.

### Q12: Can I  copy the code of ColabSaprot for my own protein language model to build another ColabPLM?

Yes, you can. But you need to agree to join OPMC, and papers that use your ColabPLM for research are encouraged to mention the SaprotHub paper, which is the source literature of OPMC.

### Q13: Would the paper be submitted to a journal.
Yes, the paper is under review of Nature Methods and you can still participate OPMC during the revision stage.
