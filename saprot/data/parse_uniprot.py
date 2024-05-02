import json
import random
import re


def extract_texts(data_dict: dict):
    uniprot_id = data_dict["primaryAccession"]
    seq_len = data_dict["sequence"]["length"]
    sep = "|"
    
    records = []
    for comment in data_dict.get("comments", []):
        # Section: Function. Subsection: Function
        if comment["commentType"] == "FUNCTION":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Function", "Function", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Miscellaneous
        if comment["commentType"] == "MISCELLANEOUS":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Function", "Miscellaneous", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Caution
        if comment["commentType"] == "CAUTION":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Function", "Caution", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Catalytic activity
        if comment["commentType"] == "CATALYTIC ACTIVITY":
            reaction = comment["reaction"]
            text = reaction["name"]
            evidences = json.dumps(reaction.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            physiologicalReactions = comment["physiologicalReactions"][0][
                "directionType"] if "physiologicalReactions" in comment else "None"
            note = f"{sep}".join(["molecule", molecule, "physiologicalReactions", physiologicalReactions])
            
            record = [uniprot_id, seq_len, "Function", "Catalytic activity", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Cofactor
        if comment["commentType"] == "COFACTOR":
            for cofactor in comment.get("cofactors", []):
                text = cofactor["name"]
                evidences = json.dumps(cofactor.get("evidences", None))
                
                molecule = comment["molecule"] if "molecule" in comment else "None"
                note = f"{sep}".join(["molecule", molecule])
                
                record = [uniprot_id, seq_len, "Function", "Cofactor", evidences, text, note]
                records.append(record)
            
            if comment.get("note", None) is not None:
                text_dict = comment["note"]["texts"][0]
                text = text_dict["value"]
                evidences = json.dumps(text_dict.get("evidences", None))
                note = "note"
                record = [uniprot_id, seq_len, "Function", "Cofactor", evidences, text, note]
                records.append(record)
        
        # Section: Function. Subsection: Activity regulation
        if comment["commentType"] == "ACTIVITY REGULATION":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Function", "Activity regulation", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Biophysicochemical properties
        if comment["commentType"] == "BIOPHYSICOCHEMICAL PROPERTIES":
            subsubsections = ["phDependence", "temperatureDependence"]
            for subsubsection in subsubsections:
                if comment.get(subsubsection, None) is not None:
                    text_dict = comment[subsubsection]["texts"][0]
                    text = text_dict["value"]
                    evidences = json.dumps(text_dict.get("evidences", None))
                    note = subsubsection
                    record = [uniprot_id, seq_len, "Function", "Biophysicochemical properties", evidences, text, note]
                    records.append(record)
        
        # Section: Function. Subsection: Pathway
        if comment["commentType"] == "PATHWAY":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Function", "Pathway", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Involvement in disease
        if comment["commentType"] == "DISEASE":
            if "disease" not in comment:
                continue
            
            disease = comment["disease"]
            text = f"{disease['diseaseId']} ({disease['acronym']})"
            evidences = json.dumps(disease.get("evidences", None))
            note = None
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Involvement in disease", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Allergenic properties
        if comment["commentType"] == "ALLERGEN":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Allergenic properties", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Toxic dose
        if comment["commentType"] == "TOXIC DOSE":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Toxic dose", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Pharmaceutical use
        if comment["commentType"] == "PHARMACEUTICAL":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Pharmaceutical use", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Disruption phenotype
        if comment["commentType"] == "DISRUPTION PHENOTYPE":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Disruption phenotype", evidences, text, note]
            records.append(record)
        
        # Section: Subcellular location. Subsection: Subcellular location
        if comment["commentType"] == "SUBCELLULAR LOCATION":
            for location in comment.get("subcellularLocations", []):
                (location)
                loc_dict = location["location"]
                text = loc_dict["value"]
                evidences = json.dumps(loc_dict.get("evidences", None))
                
                molecule = comment.get("molecule", "None")
                topology = location["topology"]["value"] if location.get("topology", None) is not None else "None"
                note = f"{sep}".join(["molecule", molecule, "topology", topology])
                
                record = [uniprot_id, seq_len, "Subcellular location", "Subcellular location", evidences, text, note]
                records.append(record)
            
            if "note" in comment:
                text_dict = comment["note"]["texts"][0]
                text = text_dict["value"]
                evidences = json.dumps(text_dict.get("evidences", None))
                note = "note"
                
                record = [uniprot_id, seq_len, "Subcellular location", "Subcellular location", evidences, text, note]
                records.append(record)
        
        # Section: PTM/Processing. Subsection: Post-translational modification
        if comment["commentType"] == "PTM":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Post-translational modification", evidences, text, note]
            records.append(record)
        
        # Section: Interaction. Subsection: Subunit
        if comment["commentType"] == "SUBUNIT":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Interaction", "Subunit", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Domain (non-positional annotation)
        if comment["commentType"] == "DOMAIN":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Family and Domains", "Domain (non-positional annotation)", evidences, text,
                      note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Sequence similarities
        if comment["commentType"] == "SIMILARITY":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Family and Domains", "Sequence similarities", evidences, text, note]
            records.append(record)
        
        # Section: Sequence. Subsection: RNA Editing
        if comment["commentType"] == "RNA EDITING":
            if "positions" in comment:
                for position in comment["positions"]:
                    text = ""
                    evidences = json.dumps(position.get("evidences", None))
                    
                    note = f"{sep}".join(["position", position["position"]])
                    
                    record = [uniprot_id, seq_len, "Sequence", "RNA Editing", evidences, text, note]
                    records.append(record)
            
            if "note" in comment:
                text_dict = comment["note"]["texts"][0]
                text = text_dict["value"]
                evidences = json.dumps(text_dict.get("evidences", None))
                note = "note"
                
                record = [uniprot_id, seq_len, "Sequence", "RNA Editing", evidences, text, note]
                records.append(record)
        
        # Section: Expression. Subsection: Tissue specificity
        if comment["commentType"] == "TISSUE SPECIFICITY":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Expression", "Tissue specificity", evidences, text, note]
            records.append(record)
        
        # Section: Expression. Subsection: Developmental stage
        if comment["commentType"] == "DEVELOPMENTAL STAGE":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Expression", "Developmental stage", evidences, text, note]
            records.append(record)
        
        # Section: Expression. Subsection: Induction
        if comment["commentType"] == "INDUCTION":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            molecule = comment["molecule"] if "molecule" in comment else "None"
            note = f"{sep}".join(["molecule", molecule])
            
            record = [uniprot_id, seq_len, "Expression", "Induction", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Biotechnology
        if comment["commentType"] == "BIOTECHNOLOGY":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            note = None
            
            record = [uniprot_id, seq_len, "Function", "Biotechnology", evidences, text, note]
            records.append(record)
        
        # Section: Sequence. Subsection: Polymorphism
        if comment["commentType"] == "POLYMORPHISM":
            text_dict = comment["texts"][0]
            text = text_dict["value"]
            evidences = json.dumps(text_dict.get("evidences", None))
            
            note = None
            
            record = [uniprot_id, seq_len, "Sequence", "Polymorphism", evidences, text, note]
            records.append(record)
    
    for feature in data_dict.get("features", []):
        # Section: Function. Subsection: Active site
        if feature["type"] == "Active site":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Function", "Active site", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Binding site
        if feature["type"] == "Binding site":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            desc = feature["description"]
            ligand = feature["ligand"]["name"]
            ligand_note = feature["ligand"].get("note", None)
            ligand_label = feature["ligand"].get("label", None)
            ligand_part = feature["ligandPart"]["name"] if "ligandPart" in feature else None
            evidences = json.dumps(feature.get("evidences", None))
            
            text = sep.join(["description", str(desc),
                             "ligand", str(ligand),
                             "note", str(ligand_note),
                             "label", str(ligand_label),
                             "part", str(ligand_part)])
            
            note = sep.join(["start", str(st),
                             "end", str(ed)])
            
            record = [uniprot_id, seq_len, "Function", "Binding site", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: Site
        if feature["type"] == "Site":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Function", "Site", evidences, text, note]
            records.append(record)
        
        # Section: Function. Subsection: DNA binding
        if feature["type"] == "DNA binding":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Function", "DNA binding", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Natural variant
        if feature["type"] == "Natural variant":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            text = feature["description"]
            
            if "in dbSNP" in text:
                continue
            else:
                text = text.split("dbSNP")[0]
            
            # We don't consider variants that do not have the mutation information
            if len(feature["alternativeSequence"]) == 0:
                continue
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            ori_seq = feature["alternativeSequence"]["originalSequence"]
            alt_seq = feature["alternativeSequence"]["alternativeSequences"][0]
            
            note = f"pos{sep}{ed}{sep}ori{sep}{ori_seq}{sep}mut{sep}{alt_seq}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Natural variant", evidences, text, note]
            records.append(record)
        
        # Section: Disease and Variants. Subsection: Mutagenesis
        if feature["type"] == "Mutagenesis":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            
            # We don't consider variants that do not have the mutation information
            if len(feature["alternativeSequence"]) == 0:
                continue
            
            ori_seq = feature["alternativeSequence"]["originalSequence"]
            alt_seq = feature["alternativeSequence"]["alternativeSequences"][0]
            
            note = f"{sep}".join(["start", str(st), "end", str(ed), "ori", ori_seq, "mut", alt_seq])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Disease and Variants", "Mutagenesis", evidences, text, note]
            records.append(record)
        
        # Section: Subcellular location. Subsection: Transmembrane
        if feature["type"] == "Transmembrane":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Subcellular location", "Transmembrane", evidences, text, note]
            records.append(record)
        
        # Section: Subcellular location. Subsection: Topological domain
        if feature["type"] == "Topological domain":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Subcellular location", "Topological domain", evidences, text, note]
            records.append(record)
        
        # Section: Subcellular location. Subsection: Intramembrane
        if feature["type"] == "Intramembrane":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Subcellular location", "Intramembrane", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Signal peptide
        if feature["type"] == "Signal":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Signal peptide", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Propeptide
        if feature["type"] == "Propeptide":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Propeptide", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Transit peptide
        if feature["type"] == "Transit peptide":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Transit peptide", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Chain
        if feature["type"] == "Chain":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Chain", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Peptide
        if feature["type"] == "Peptide":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            note = f"start{sep}{st}{sep}end{sep}{ed}"
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Peptide", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Modified residue
        if feature["type"] == "Modified residue":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            aa = data_dict["sequence"]["value"][st - 1: ed]
            
            note = f"{sep}".join(["start", str(st), "end", str(ed), "aa", aa])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Modified residue", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Lipidation
        if feature["type"] == "Lipidation":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            aa = data_dict["sequence"]["value"][st - 1: ed]
            
            note = f"{sep}".join(["start", str(st), "end", str(ed), "aa", aa])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Lipidation", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Glycosylation
        if feature["type"] == "Glycosylation":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            text = feature["description"]
            aa = data_dict["sequence"]["value"][st - 1: ed]
            
            note = f"{sep}".join(["start", str(st), "end", str(ed), "aa", aa])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Glycosylation", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Disulfide bond
        if feature["type"] == "Disulfide bond":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            st_aa = data_dict["sequence"]["value"][st - 1]
            ed_aa = data_dict["sequence"]["value"][ed - 1]
            
            note = f"{sep}".join(["start", str(st), "end", str(ed), "start_aa", st_aa, "end_aa", ed_aa])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Disulfide bond", evidences, text, note]
            records.append(record)
        
        # Section: PTM/Processing. Subsection: Cross-link
        if feature["type"] == "Cross-link":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            st_aa = data_dict["sequence"]["value"][st - 1]
            ed_aa = data_dict["sequence"]["value"][ed - 1]
            
            note = f"{sep}".join(["start", str(st), "end", str(ed), "start_aa", st_aa, "end_aa", ed_aa])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "PTM/Processing", "Cross-link", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Domain
        if feature["type"] == "Domain":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            note = f"{sep}".join(["start", str(st), "end", str(ed)])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Family and Domains", "Domain", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Repeat
        if feature["type"] == "Repeat":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            note = f"{sep}".join(["start", str(st), "end", str(ed)])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Family and Domains", "Repeat", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Compositional bias
        if feature["type"] == "Compositional bias":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            note = f"{sep}".join(["start", str(st), "end", str(ed)])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Family and Domains", "Compositional bias", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Region
        if feature["type"] == "Region":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            note = f"{sep}".join(["start", str(st), "end", str(ed)])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Family and Domains", "Region", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Coiled coil
        if feature["type"] == "Coiled coil":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            note = f"{sep}".join(["start", str(st), "end", str(ed)])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Family and Domains", "Coiled coil", evidences, text, note]
            records.append(record)
        
        # Section: Family and Domains. Subsection: Motif
        if feature["type"] == "Motif":
            st = feature["location"]["start"]["value"]
            ed = feature["location"]["end"]["value"]
            text = feature["description"]
            
            if st is None or ed is None or "sequence" in feature["location"]:
                continue
            
            note = f"{sep}".join(["start", str(st), "end", str(ed)])
            evidences = json.dumps(feature.get("evidences", None))
            
            record = [uniprot_id, seq_len, "Family and Domains", "Motif", evidences, text, note]
            records.append(record)
    
    for db in data_dict.get("uniProtKBCrossReferences", []):
        # Section: Function. Subsection: GO annotation
        if db['database'] == "GO":
            aspect_term = db["properties"][0]["value"]
            aspect, term = aspect_term.split(":", 1)
            aspect_dict = {"C": "Cellular component", "F": "Molecular function", "P": "Biological process"}
            aspect = aspect_dict[aspect]
            
            evidences = json.dumps(db.get("evidences", None))
            note = f"{aspect}{sep}{db['id']}"
            
            record = [uniprot_id, seq_len, "Function", "GO annotation", evidences, term, note]
            records.append(record)
        
        # Section: Names and Taxonomy. Subsection: Proteomes
        if db['database'] == "Proteomes":
            evidences = db["id"]
            text = db["properties"][0]["value"]
            
            note = None
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Proteomes", evidences, text, note]
            records.append(record)
    
    # Section: Names and Taxonomy. Subsection: Protein names
    for name, sub_dict in data_dict["proteinDescription"].items():
        if name == "recommendedName":
            # Full Name
            text_dict = sub_dict["fullName"]
            evidences = json.dumps(text_dict.get("evidences", None))
            text = text_dict["value"]
            note = sep.join(["recommendedName", "fullName"])
            
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
            records.append(record)
            
            # Short Names
            for short_name_dict in sub_dict.get("shortNames", []):
                evidences = json.dumps(short_name_dict.get("evidences", None))
                text = short_name_dict["value"]
                note = sep.join(["recommendedName", "shortNames"])
                
                record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                records.append(record)
        
        if name == "alternativeNames":
            for alt_name_dict in sub_dict:
                # Full Name
                text_dict = alt_name_dict["fullName"]
                evidences = json.dumps(text_dict.get("evidences", None))
                text = text_dict["value"]
                note = sep.join(["alternativeNames", "fullName"])
                
                record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                records.append(record)
                
                # Short Names
                for short_name_dict in alt_name_dict.get("shortNames", []):
                    evidences = json.dumps(short_name_dict.get("evidences", None))
                    text = short_name_dict["value"]
                    note = sep.join(["alternativeNames", "shortNames"])
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                    records.append(record)
        
        if name == "includes":
            for includes in sub_dict:
                if "recommendedName" in includes:
                    # Full Name
                    text_dict = includes["recommendedName"]["fullName"]
                    evidences = json.dumps(text_dict.get("evidences", None))
                    text = text_dict["value"]
                    note = sep.join(["includes", "recommendedName", "fullName"])
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                    records.append(record)
                    
                    # Short Names
                    for short_name_dict in includes["recommendedName"].get("shortNames", []):
                        evidences = json.dumps(short_name_dict.get("evidences", None))
                        text = short_name_dict["value"]
                        note = sep.join(["includes", "recommendedName", "shortNames"])
                        
                        record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                        records.append(record)
                
                for alt_name_dict in includes.get("alternativeNames", []):
                    # Full Name
                    text_dict = alt_name_dict["fullName"]
                    evidences = json.dumps(text_dict.get("evidences", None))
                    text = text_dict["value"]
                    note = sep.join(["includes", "alternativeNames", "fullName"])
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                    records.append(record)
                    
                    # Short Names
                    for short_name_dict in alt_name_dict.get("shortNames", []):
                        evidences = json.dumps(short_name_dict.get("evidences", None))
                        text = short_name_dict["value"]
                        note = sep.join(["includes", "alternativeNames", "shortNames"])
                        
                        record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                        records.append(record)
        
        if name == "cdAntigenNames":
            for cdAntigenName in sub_dict:
                evidences = json.dumps(cdAntigenName.get("evidences", None))
                text = cdAntigenName["value"]
                note = "cdAntigenNames"
                
                record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                records.append(record)
        
        if name == "contains":
            for contain in sub_dict:
                if "recommendedName" in contain:
                    # Full Name
                    text_dict = contain["recommendedName"]["fullName"]
                    evidences = json.dumps(text_dict.get("evidences", None))
                    text = text_dict["value"]
                    note = sep.join(["contains", "recommendedName", "fullName"])
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                    records.append(record)
                    
                    # Short Names
                    for short_name_dict in contain["recommendedName"].get("shortNames", []):
                        evidences = json.dumps(short_name_dict.get("evidences", None))
                        text = short_name_dict["value"]
                        note = sep.join(["contains", "recommendedName", "shortNames"])
                        
                        record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                        records.append(record)
                
                for alt_name_dict in contain.get("alternativeNames", []):
                    # Full Name
                    text_dict = alt_name_dict["fullName"]
                    evidences = json.dumps(text_dict.get("evidences", None))
                    text = text_dict["value"]
                    note = sep.join(["contains", "alternativeNames", "fullName"])
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                    records.append(record)
                    
                    # Short Names
                    for short_name_dict in alt_name_dict.get("shortNames", []):
                        evidences = json.dumps(short_name_dict.get("evidences", None))
                        text = short_name_dict["value"]
                        note = sep.join(["contains", "alternativeNames", "shortNames"])
                        
                        record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                        records.append(record)
        
        if name == "allergenName":
            evidences = json.dumps(sub_dict.get("evidences", None))
            text = sub_dict["value"]
            note = "allergenName"
            
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
            records.append(record)
        
        if name == "innNames":
            for innName in sub_dict:
                evidences = json.dumps(innName.get("evidences", None))
                text = innName["value"]
                note = "innNames"
                
                record = [uniprot_id, seq_len, "Names and Taxonomy", "Protein names", evidences, text, note]
                records.append(record)
    
    # Section: Names and Taxonomy. Subsection: Gene names
    for genes in data_dict.get("genes", []):
        for name, sub_dict in genes.items():
            if name == "geneName":
                evidences = json.dumps(sub_dict.get("evidences", None))
                text = sub_dict["value"]
                note = "geneName"
                
                record = [uniprot_id, seq_len, "Names and Taxonomy", "Gene names", evidences, text, note]
                records.append(record)
            
            if name == "synonyms":
                for synonym in sub_dict:
                    evidences = json.dumps(synonym.get("evidences", None))
                    text = synonym["value"]
                    note = "synonyms"
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Gene names", evidences, text, note]
                    records.append(record)
            
            if name == "orderedLocusNames":
                for orderedLocusName in sub_dict:
                    evidences = json.dumps(orderedLocusName.get("evidences", None))
                    text = orderedLocusName["value"]
                    note = "orderedLocusNames"
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Gene names", evidences, text, note]
                    records.append(record)
            
            if name == "orfNames":
                for orfName in sub_dict:
                    evidences = json.dumps(orfName.get("evidences", None))
                    text = orfName["value"]
                    note = "orfNames"
                    
                    record = [uniprot_id, seq_len, "Names and Taxonomy", "Gene names", evidences, text, note]
                    records.append(record)
    
    if "organism" in data_dict:
        organism = data_dict["organism"]
        
        # Section: Names and Taxonomy. Subsection: Organism
        if "scientificName" in organism:
            evidences = "null"
            text = organism["scientificName"]
            note = "scientificName"
            
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Organism", evidences, text, note]
            records.append(record)
        
        if "commonName" in organism:
            evidences = "null"
            text = organism["commonName"]
            note = "commonName"
            
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Organism", evidences, text, note]
            records.append(record)
        
        if "synonyms" in organism:
            for synonym in organism["synonyms"]:
                evidences = "null"
                text = synonym
                note = "synonyms"
                
                record = [uniprot_id, seq_len, "Names and Taxonomy", "Organism", evidences, text, note]
                records.append(record)
        
        # Section: Names and Taxonomy. Subsection: Taxonomic lineage
        if "lineage" in organism:
            evidences = "null"
            text = "->".join(organism["lineage"])
            note = sep.join(["number", str(len(organism["lineage"]))])
            
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Taxonomic lineage", evidences, text, note]
            records.append(record)
    
    if "organismHosts" in data_dict:
        # Section: Names and Taxonomy. Subsection: Virus host
        for host in data_dict["organismHosts"]:
            evidences = json.dumps(host.get("evidences", None))
            note = sep.join(["scientificName", host.get("scientificName", "None"),
                             "commonName", host.get("commonName", "None"),
                             "synonyms", host.get("synonyms", ["None"])[0]])
            
            record = [uniprot_id, seq_len, "Names and Taxonomy", "Virus host", evidences, "", note]
            records.append(record)
    
    # Post processing for records
    for record in records:
        # Replace all '\n' in text with ' '
        record[5] = record[5].replace("\n", " ")
    
    return records


def record2text(record: list, template: dict, template_id: int = None, return_info: bool = False) -> str or list:
    """
    Convert a record into a text based on the subsection type
    Args:
        record: A list of record containing the following information
        ["subsection", "text", "note"]

        template: A dictionary of templates for each subsection type

        template_id: An integer indicating which template to use. If None, it will randomly select a template.
        
        return_info: A boolean indicating whether to return the used template and inserted information

    Returns:
        text: a text converted from a record
        Note that for subsection "Taxonomic lineage", it will return a list of texts.
    """

    global raw_text, ori, mut, st, ed, pos, cls, st_aa, ed_aa
    subsection, raw_text, note = record
    text, used_template = None, None
    sep = "|"

    def fill_template(template, template_id, extra_key=None):
        candidates = template[subsection] if extra_key is None else template[subsection][extra_key]

        if template_id is None:
            template_id = random.randint(0, len(candidates) - 1)

        return eval('f"' + candidates[template_id] + '"'), candidates[template_id]

    # Section: Function. Subsection: Function
    if subsection == "Function":
        text = raw_text

    # Section: Function. Subsection: Miscellaneous
    if subsection == "Miscellaneous":
        text = raw_text

    # Section: Function. Subsection: Caution
    if subsection == "Caution":
        text = raw_text

    # Section: Function. Subsection: Catalytic activity
    if subsection == "Catalytic activity":
        text, used_template = fill_template(template, template_id)

        physiologicalReactions = note.split(sep)[-1]
        if physiologicalReactions != "None":
            if physiologicalReactions == "left-to-right":
                plus = "This reaction proceeds in the forward direction."
            else:
                plus = "This reaction proceeds in the reverse direction."

            text = text + " " + plus

    # Section: Function. Subsection: Cofactor
    if subsection == "Cofactor":
        if note == "note":
            text = raw_text
        else:
            text, used_template = fill_template(template, template_id)

    # Section: Function. Subsection: Activity regulation
    if subsection == "Activity regulation":
        text = raw_text

    # Section: Function. Subsection: Biophysicochemical properties
    if subsection == "Biophysicochemical properties":
        text = raw_text

    # Section: Function. Subsection: Pathway
    if subsection == "Pathway":
        text, used_template = fill_template(template, template_id)

    # Section: Function. Subsection: Active site
    if subsection == "Active site":
        pos = note.split(sep)[1]
        text, used_template = fill_template(template, template_id)

    # Section: Function. Subsection: Binding site
    if subsection == "Binding site":
        segments = raw_text.split(sep)
        _, desc, _, ligand, _, ligand_note, _, label, _, part = segments
        raw_text = ligand
        if label != "None":
            raw_text = raw_text + " " + label

        if part != "None":
            raw_text = f"{part} of {raw_text}"

        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Function. Subsection: Site
    if subsection == "Site":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        # Lowercase the first letter of the raw_text
        raw_text = raw_text[0].lower() + raw_text[1:]
        text, used_template = fill_template(template, template_id)

    # Section: Function. Subsection: DNA binding
    if subsection == "DNA binding":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        if raw_text == "":
            raw_text = "unknown"

        # Lowercase the first letter of the raw_text
        raw_text = raw_text[0].lower() + raw_text[1:]
        text, used_template = fill_template(template, template_id)

    # Section: Function. Subsection: Biotechnology
    if subsection == "Biotechnology":
        text = raw_text

    # Section: Function. Subsection: GO annotation
    if subsection == "GO annotation":
        cls = note.split(sep)[0].lower()
        text, used_template = fill_template(template, template_id)

    # Section: Names and Taxonomy. Subsection: Protein names
    if subsection == "Protein names":
        text, used_template = fill_template(template, template_id)

    # Section: Names and Taxonomy. Subsection: Gene names
    if subsection == "Gene names":
        if note == "orfNames" or note == "orderedLocusNames":
            text, used_template = fill_template(template, template_id, note)
        else:
            text, used_template = fill_template(template, template_id, "normal")

    # Section: Names and Taxonomy. Subsection: Organism
    if subsection == "Organism":
        text, used_template = fill_template(template, template_id)

    # Section: Names and Taxonomy. Subsection: Taxonomic lineage
    if subsection == "Taxonomic lineage":
        classes = raw_text.split("->")
        text = []
        for raw_text in classes:
            single_text, used_template = fill_template(template, template_id)
            text.append(single_text)

    # Section: Names and Taxonomy. Subsection: Proteomes
    if subsection == "Proteomes":
        text, used_template = fill_template(template, template_id)

    # Section: Names and Taxonomy. Subsection: Virus host
    if subsection == "Virus host":
        _, scientificName, _, commonName, _, synonyms = note.split(sep)
        text = []
        for raw_text in [scientificName, commonName, synonyms]:
            if raw_text != "None":
                text.append(fill_template(template, template_id))

    # Section: Disease and Variants. Subsection: Involvement in disease
    if subsection == "Involvement in disease":
        text, used_template = fill_template(template, template_id)

    # Section: Disease and Variants. Subsection: Natural variant
    if subsection == "Natural variant":
        _, pos, _, ori, _, mut = note.split(sep)

        # Lowercase the first letter of the raw_text
        raw_text = raw_text[0].lower() + raw_text[1:] if raw_text != "" else ""
        if raw_text == "":
            text, used_template = fill_template(template, template_id, extra_key="without raw_text")
        else:
            text, used_template = fill_template(template, template_id, extra_key="with raw_text")

    # Section: Disease and Variants. Subsection: Allergenic properties
    if subsection == "Allergenic properties":
        text = raw_text

    # Section: Disease and Variants. Subsection: Toxic dose
    if subsection == "Toxic dose":
        text = raw_text

    # Section: Disease and Variants. Subsection: Pharmaceutical use
    if subsection == "Pharmaceutical use":
        text = raw_text

    # Section: Disease and Variants. Subsection: Disruption phenotype
    if subsection == "Disruption phenotype":
        text = raw_text

    # Section: Disease and Variants. Subsection: Mutagenesis
    if subsection == "Mutagenesis":
        _, st, _, ed, _, ori, _, mut = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        # Lowercase the first letter of the raw_text
        raw_text = raw_text[0].lower() + raw_text[1:]
        text, used_template = fill_template(template, template_id)

    # Section: Subcellular location. Subsection: Subcellular location
    if subsection == "Subcellular location":
        if note == "note":
            text = raw_text
        else:
            text, used_template = fill_template(template, template_id)

    # Section: Subcellular location. Subsection: Transmembrane
    if subsection == "Transmembrane":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        if raw_text == "":
            text, used_template = fill_template(template, template_id, extra_key="without raw_text")
        else:
            text, used_template = fill_template(template, template_id, extra_key="with raw_text")

    # Section: Subcellular location. Subsection: Topological domain
    if subsection == "Topological domain":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Subcellular location. Subsection: Intramembrane
    if subsection == "Intramembrane":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        if raw_text == "":
            text, used_template = fill_template(template, template_id, extra_key="without raw_text")
        else:
            text, used_template = fill_template(template, template_id, extra_key="with raw_text")

    # Section: PTM/Processing. Subsection: Signal peptide
    if subsection == "Signal peptide":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Propeptide
    if subsection == "Propeptide":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        if raw_text == "":
            text, used_template = fill_template(template, template_id, extra_key="without raw_text")
        else:
            text, used_template = fill_template(template, template_id, extra_key="with raw_text")

    # Section: PTM/Processing. Subsection: Transit peptide
    if subsection == "Transit peptide":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Chain
    if subsection == "Chain":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Peptide
    if subsection == "Peptide":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Modified residue
    if subsection == "Modified residue":
        _, st, _, ed, _, _ = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Lipidation
    if subsection == "Lipidation":
        _, st, _, ed, _, _ = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Glycosylation
    if subsection == "Glycosylation":
        _, st, _, ed, _, _ = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Disulfide bond
    if subsection == "Disulfide bond":
        _, st, _, ed, _, _, _, _ = note.split(sep)
        text, used_template = fill_template(template, template_id)

    # Section: PTM/Processing. Subsection: Cross-link
    if subsection == "Cross-link":
        _, st, _, ed, _, st_aa, _, ed_aa = note.split(sep)
        if st == ed:
            text, used_template = fill_template(template, template_id, "interchain")
        else:
            text, used_template = fill_template(template, template_id, "intrachain")

    # Section: PTM/Processing. Subsection: Post-translational modification
    if subsection == "Post-translational modification":
        text = raw_text

    # Section: Expression. Subsection: Tissue specificity
    if subsection == "Tissue specificity":
        text = raw_text

    # Section: Expression. Subsection: Developmental stage
    if subsection == "Developmental stage":
        text = raw_text

    # Section: Expression. Subsection: Induction
    if subsection == "Induction":
        text = raw_text

    # Section: Interaction. Subsection: Subunit
    if subsection == "Subunit":
        text = raw_text

    # Section: Family and Domains. Subsection: Domain
    if subsection == "Domain":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Family and Domains. Subsection: Repeat
    if subsection == "Repeat":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Family and Domains. Subsection: Compositional bias
    if subsection == "Compositional bias":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Family and Domains. Subsection: Region
    if subsection == "Region":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"
        text, used_template = fill_template(template, template_id)

    # Section: Family and Domains. Subsection: Coiled coil
    if subsection == "Coiled coil":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Family and Domains. Subsection: Motif
    if subsection == "Motif":
        _, st, _, ed = note.split(sep)
        if st == ed:
            pos = st
        else:
            pos = f"{st} to {ed}"

        text, used_template = fill_template(template, template_id)

    # Section: Family and Domains. Subsection: Domain (non-positional annotation)
    if subsection == "Domain (non-positional annotation)":
        text = raw_text

    # Section: Family and Domains. Subsection: Sequence similarities
    if subsection == "Sequence similarities":
        text = raw_text

    # Section: Sequence. Subsection: RNA Editing
    if subsection == "RNA Editing":
        if note == "note":
            text = raw_text
        else:
            pos = note.split(sep)[-1]
            text, used_template = fill_template(template, template_id)

    # Section: Sequence. Subsection: Polymorphism
    if subsection == "Polymorphism":
        text = raw_text

    assert text is not None
    
    if return_info:
        info_dict = {"used_template": used_template}
        if used_template is not None:
            # Extract all variables in the used template
            for k in re.findall(r"{(\w+)}", used_template):
                info_dict[k] = eval(k)
        return text, info_dict
    
    else:
        return text
