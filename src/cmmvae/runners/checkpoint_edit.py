import torch
import fnmatch

PATH = "/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/filtered/checkpoints/best_model.ckpt"

ckpt = torch.load(PATH, map_location="cpu")
state_dict = ckpt["state_dict"]

orig_keys = set(state_dict.keys())
new_state_dict = {}

pattern = "module.vae.conditionals*.human.*"
for k, v in state_dict.items():
    if fnmatch.fnmatch(k, pattern):
        # drop the "human" node and rewrite the key
        # e.g. "model.x.human.y.z" → "model.x.y.z"
        print(k)
        new_key = k.replace(".human", "")
        print(new_key)
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

# sanity check
print("Original keys:", len(state_dict))
print("New keys     :", len(new_state_dict))

ckpt["state_dict"] = new_state_dict
torch.save(ckpt, PATH)
