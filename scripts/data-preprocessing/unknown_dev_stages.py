import cellxgene_ontology_guide.ontology_parser as op
import cellxgene_census as cxg

PARSER = op.OntologyParser()

HUMAN_EMBRYO = "HsapDv:0000002"
HUMAN_FETAL = "HsapDv:0000037"
HUMAN_IMMATURE = "HsapDv:0000264"
HUMAN_YOUNG_ADULT = "HsapDv:0000266"
HUMAN_MIDDLE_ADULT = "HsapDv:0000267"
HUMAN_LATE_ADULT = "HsapDv:0000227"

HUMAN_EMBRYO_DESCENDANTS = PARSER.get_term_descendants(HUMAN_EMBRYO, include_self=True)
HUMAN_FETAL_DESCENDANTS = PARSER.get_term_descendants(HUMAN_FETAL, include_self=True)
HUMAN_IMMATURE_DESCENDANTS = PARSER.get_term_descendants(HUMAN_IMMATURE, include_self=True)
HUMAN_YOUNG_ADULT_DESCENDANTS = PARSER.get_term_descendants(HUMAN_YOUNG_ADULT, include_self=True)
HUMAN_MIDDLE_ADULT_DESCENDANTS = PARSER.get_term_descendants(HUMAN_MIDDLE_ADULT, include_self=True)
HUMAN_LATE_ADULT_DESCENDANTS = PARSER.get_term_descendants(HUMAN_LATE_ADULT, include_self=True)

MOUSE_EMBRYO = "MmusDv:0000002"
MOUSE_FETAL = "MmusDv:0000031"
MOUSE_IMMATURE = "MmusDv:0000043"
MOUSE_YOUNG_ADULT = "MmusDv:0000153"
MOUSE_MIDDLE_ADULT = "MmusDv:0000135"
MOUSE_LATE_ADULT = "MmusDv:0000134"

MOUSE_EMBRYO_DESCENDANTS = PARSER.get_term_descendants(MOUSE_EMBRYO, include_self=True)
MOUSE_FETAL_DESCENDANTS = PARSER.get_term_descendants(MOUSE_FETAL, include_self=True)
MOUSE_IMMATURE_DESCENDANTS = PARSER.get_term_descendants(MOUSE_IMMATURE, include_self=True)
MOUSE_YOUNG_ADULT_DESCENDANTS = PARSER.get_term_descendants(MOUSE_YOUNG_ADULT, include_self=True)
MOUSE_MIDDLE_ADULT_DESCENDANTS = PARSER.get_term_descendants(MOUSE_MIDDLE_ADULT, include_self=True)
MOUSE_LATE_ADULT_DESCENDANTS = PARSER.get_term_descendants(MOUSE_LATE_ADULT, include_self=True)

def tag_human(term_id):
    
    if term_id == "unknown":
        return "unknown"
    elif term_id in HUMAN_EMBRYO_DESCENDANTS:
        return "embryonic"
    elif term_id in HUMAN_FETAL_DESCENDANTS:
        return "fetal"
    elif term_id in HUMAN_IMMATURE_DESCENDANTS:
        return "immature"
    elif term_id in HUMAN_YOUNG_ADULT_DESCENDANTS:
        return "young_adult"
    elif term_id in HUMAN_MIDDLE_ADULT_DESCENDANTS:
        return "middle_adult"
    elif term_id in HUMAN_LATE_ADULT_DESCENDANTS:
        return "late_adult"
    else:
        if PARSER.is_term_deprecated(term_id):
            replacement = PARSER.get_term_replacement(term_id)
            if replacement is not None:
                return tag_human(replacement)
            else:
                consider = PARSER.get_term_metadata(term_id)["consider"]
                if consider is not None:
                    return tag_human(consider[0])
                else:
                    return "unknown"
        return "unknown"
    
def tag_mouse(term_id):
    
    if term_id == "unknown":
        return "unknown"
    elif term_id in MOUSE_EMBRYO_DESCENDANTS:
        return "embryonic"
    elif term_id in MOUSE_FETAL_DESCENDANTS:
        return "fetal"
    elif term_id in MOUSE_IMMATURE_DESCENDANTS:
        return "immature"
    elif term_id in MOUSE_YOUNG_ADULT_DESCENDANTS:
        return "young_adult"
    elif term_id in MOUSE_MIDDLE_ADULT_DESCENDANTS:
        return "middle_adult"
    elif term_id in MOUSE_LATE_ADULT_DESCENDANTS:
        return "late_adult"
    else:
        if PARSER.is_term_deprecated(term_id):
            replacement = PARSER.get_term_replacement(term_id)
            if replacement is not None:
                return tag_mouse(replacement)
            else:
                consider = PARSER.get_term_metadata(term_id)["consider"]
                if consider is not None:
                    return tag_mouse(consider[0])
                else:
                    return "unknown"
        return "unknown"
    
CENSUS_VERSION = "2025-01-30"

with cxg.open_soma(census_version=CENSUS_VERSION) as census:

    human_obs = cxg.get_obs(
        census=census,
        organism="Homo sapiens",
        value_filter=f"is_primary_data == True",
        column_names=["development_stage_ontology_term_id"]
    )

    human_obs["dev_stage"] = human_obs["development_stage_ontology_term_id"].apply(tag_human)

    human_unknowns = human_obs[human_obs["dev_stage"] == "unknown"]
    human_unknowns = human_unknowns.drop_duplicates()
    human_unknowns = human_unknowns["development_stage_ontology_term_id"]
    print(human_unknowns.tolist())
    human_unknowns.to_csv("/mnt/projects/debruinz_project/tony_boos/human_unknown_devs.csv", index=False)
    human_obs = human_obs.drop_duplicates().sort_values(by="development_stage_ontology_term_id")
    print(human_obs)
    human_obs.to_csv("/mnt/projects/debruinz_project/tony_boos/human_dev_tags.csv", index=False)

    mouse_obs = cxg.get_obs(
        census=census,
        organism="Mus musculus",
        value_filter=f"is_primary_data == True",
        column_names=["development_stage_ontology_term_id"]
    )

    mouse_obs["dev_stage"] = mouse_obs["development_stage_ontology_term_id"].apply(tag_mouse)

    mouse_unknowns = mouse_obs[mouse_obs["dev_stage"] == "unknown"]
    mouse_unknowns = mouse_unknowns.drop_duplicates()
    mouse_unknowns = mouse_unknowns["development_stage_ontology_term_id"]
    print(mouse_unknowns.tolist())
    mouse_unknowns.to_csv("/mnt/projects/debruinz_project/tony_boos/mouse_unknown_devs.csv", index=False)
    mouse_obs = mouse_obs.drop_duplicates().sort_values(by="development_stage_ontology_term_id")
    print(mouse_obs)
    mouse_obs.to_csv("/mnt/projects/debruinz_project/tony_boos/mouse_dev_tags.csv", index=False)
