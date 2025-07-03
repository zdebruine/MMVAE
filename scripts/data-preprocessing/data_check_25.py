import numpy as np
import cellxgene_census as cxg

SEED = 42
CENSUS_VERSION = "2025-01-30"
np.random.seed(SEED)

VALUE_FILTER = (
    "is_primary_data == True and "
    "assay != 'unknown' and "
    "cell_type != 'unknown' and "
    "dataset_id != 'unknown' and "
    "disease != 'unknown' and "
    "donor_id != 'unknown' and "
    "sex != 'unknown' and "
    "suspension_type != 'unknown' and "
    "tissue_general != 'unknown'"
)
# VALUE_FILTER = (
#     "is_primary_data == True and "
#     "assay != 'unknown' and "
#     "dataset_id != 'unknown' and "
#     "disease != 'unknown' and "
#     "donor_id != 'unknown' and "
#     "suspension_type != 'unknown' and "
#     "tissue_general != 'unknown'"
# )

# CELL_FILTER = "and cell_type != 'unknown'"
# SEX_FILTER = "and sex != 'unknown'"

SOMA_ID = 'soma_joinid'

HUMAN = "homo_sapiens"
HUMAN_DEV_FILTER = (
    " and development_stage_ontology_term_id not in "
    "['unknown', 'HsapDv:0000226', 'HsapDv:0000258', 'HsapDv:0010000']"
)

MOUSE = "mus_musculus"
MOUSE_DEV_FILTER = (
    " and development_stage_ontology_term_id not in "
    "['unknown', 'MmusDv:0000092', 'MmusDv:0000110', 'MmusDv:0000136']"
)

with cxg.open_soma(census_version=CENSUS_VERSION) as census:
    human_obs = cxg.get_obs(census, HUMAN, value_filter=VALUE_FILTER, column_names=[SOMA_ID])
    mouse_obs = cxg.get_obs(census, MOUSE, value_filter=VALUE_FILTER, column_names=[SOMA_ID])

human_soma_ids = set(human_obs[SOMA_ID])
mouse_soma_ids = set(mouse_obs[SOMA_ID])

print(f"Without dev filter (H): {len(human_soma_ids)}")
print(f"Without dev filter (M): {len(mouse_soma_ids)}")

# with cxg.open_soma(census_version=CENSUS_VERSION) as census:
#     human_obs = cxg.get_obs(census, HUMAN, value_filter=VALUE_FILTER + CELL_FILTER, column_names=[SOMA_ID])
#     mouse_obs = cxg.get_obs(census, MOUSE, value_filter=VALUE_FILTER + CELL_FILTER, column_names=[SOMA_ID])

# human_soma_ids = set(human_obs[SOMA_ID])
# mouse_soma_ids = set(mouse_obs[SOMA_ID])

# print(f"With cell filter (H): {len(human_soma_ids)}")
# print(f"With cell filter (M): {len(mouse_soma_ids)}")

# with cxg.open_soma(census_version=CENSUS_VERSION) as census:
#     human_obs = cxg.get_obs(census, HUMAN, value_filter=VALUE_FILTER + SEX_FILTER, column_names=[SOMA_ID])
#     mouse_obs = cxg.get_obs(census, MOUSE, value_filter=VALUE_FILTER + SEX_FILTER, column_names=[SOMA_ID])

# human_soma_ids = set(human_obs[SOMA_ID])
# mouse_soma_ids = set(mouse_obs[SOMA_ID])

# print(f"With sex filter (H): {len(human_soma_ids)}")
# print(f"With sex filter (M): {len(mouse_soma_ids)}")

with cxg.open_soma(census_version=CENSUS_VERSION) as census:
    human_obs = cxg.get_obs(census, HUMAN, value_filter=VALUE_FILTER + HUMAN_DEV_FILTER, column_names=[SOMA_ID])
    mouse_obs = cxg.get_obs(census, MOUSE, value_filter=VALUE_FILTER + MOUSE_DEV_FILTER, column_names=[SOMA_ID])

human_soma_ids = set(human_obs[SOMA_ID])
mouse_soma_ids = set(mouse_obs[SOMA_ID])

print(f"With dev filter (H): {len(human_soma_ids)}")
print(f"With dev filter (M): {len(mouse_soma_ids)}")

# with cxg.open_soma(census_version=CENSUS_VERSION) as census:
#     human_obs = cxg.get_obs(census, HUMAN, value_filter=VALUE_FILTER + CELL_FILTER + SEX_FILTER + HUMAN_DEV_FILTER, column_names=[SOMA_ID])
#     mouse_obs = cxg.get_obs(census, MOUSE, value_filter=VALUE_FILTER + CELL_FILTER + SEX_FILTER + MOUSE_DEV_FILTER, column_names=[SOMA_ID])

# human_soma_ids = set(human_obs[SOMA_ID])
# mouse_soma_ids = set(mouse_obs[SOMA_ID])

# print(f"With all filters (H): {len(human_soma_ids)}")
# print(f"With all filters (M): {len(mouse_soma_ids)}")

