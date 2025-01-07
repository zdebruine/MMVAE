import argparse as ap
import glob
import os
import pandas as pd

import cellxgene_ontology_guide.ontology_parser as op

HUMAN_EMBRYO = "HsapDv:0000002"
HUMAN_FETAL = "HsapDv:0000037"
HUMAN_IMMATURE = "HsapDv:0000264"
HUMAN_YOUNG_ADULT = "HsapDv:0000266"
HUMAN_MIDDLE_ADULT = "HsapDv:0000267"
HUMAN_LATE_ADULT = "HsapDv:0000227"

MOUSE_EMBRYO = "MmusDv:0000002"
MOUSE_FETAL = "MmusDv:0000031"
MOUSE_IMMATURE = "MmusDv:0000043"
MOUSE_YOUNG_ADULT = "MmusDv:0000153"
MOUSE_MIDDLE_ADULT = "MmusDv:0000135"
MOUSE_LATE_ADULT = "MmusDv:0000134"

def tag_human(term_id):
    parser = op.OntologyParser()
    if term_id == "unknown":
        return "unknown"
    elif term_id in parser.get_term_descendants(HUMAN_EMBRYO):
        return "embryonic"
    elif term_id in parser.get_term_descendants(HUMAN_FETAL):
        return "fetal"
    elif term_id in parser.get_term_descendants(HUMAN_IMMATURE):
        return "immature"
    elif term_id in parser.get_term_descendants(HUMAN_YOUNG_ADULT):
        return "young_adult"
    elif term_id in parser.get_term_descendants(HUMAN_MIDDLE_ADULT):
        return "middle_adult"
    elif term_id in parser.get_term_descendants(HUMAN_LATE_ADULT):
        return "late_adult"
    else:
        if parser.is_term_deprecated(term_id):
            replacement = parser.get_term_replacement(term_id)
            if replacement is not None:
                return tag_human(replacement)
            else:
                consider = parser.get_term_metadata(term_id)["consider"]
                if consider is not None:
                    return tag_human(consider[0])
                else:
                    return "unknown"
        return "unknown"
    
def tag_mouse(term_id):
    parser = op.OntologyParser()
    if term_id == "unknown":
        return "unknown"
    elif term_id in parser.get_term_descendants(MOUSE_EMBRYO):
        return "embryonic"
    elif term_id in parser.get_term_descendants(MOUSE_FETAL):
        return "fetal"
    elif term_id in parser.get_term_descendants(MOUSE_IMMATURE):
        return "immature"
    elif term_id in parser.get_term_descendants(MOUSE_YOUNG_ADULT):
        return "young_adult"
    elif term_id in parser.get_term_descendants(MOUSE_MIDDLE_ADULT):
        return "middle_adult"
    elif term_id in parser.get_term_descendants(MOUSE_LATE_ADULT):
        return "late_adult"
    else:
        if parser.is_term_deprecated(term_id):
            replacement = parser.get_term_replacement(term_id)
            if replacement is not None:
                return tag_mouse(replacement)
            else:
                consider = parser.get_term_metadata(term_id)["consider"]
                if consider is not None:
                    return tag_mouse(consider[0])
                else:
                    return "unknown"
        return "unknown"
                
def main(directory):

    human_metadata = glob.glob(
        os.path.join(
            directory, "human*.pkl"
        )
    )
    mouse_metadata = glob.glob(
        os.path.join(
            directory, "mouse*.pkl"
        )
    )
    for file in human_metadata:
        df = pd.read_pickle(file)
        df["dev_stage"] = df["development_stage_ontology_term_id"].apply(tag_human)
        # assert len(df[(df["dev_stage"] == "unknown") & (df["development_stage"] != "unknown")]) == 0
        df.to_pickle(file, protocol=4)

    for file in mouse_metadata:
        df = pd.read_pickle(file)
        df["dev_stage"] = df["development_stage_ontology_term_id"].apply(tag_mouse)
        # assert len(df[(df["dev_stage"] == "unknown") & (df["development_stage"] != "unknown")]) == 0
        df.to_pickle(file, protocol=4)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory to load data from."
    )
    args = parser.parse_args()
    main(args.directory)