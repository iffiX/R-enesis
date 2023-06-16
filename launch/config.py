import os
import dill
from typing import Dict, Any
from . import PROJECT_ROOT


def merge_dict(target, source, path=None, changes=None):
    path = path if path is not None else []
    changes = changes if changes is not None else []
    for key in source:
        if key in target:
            if isinstance(target[key], dict) and isinstance(source[key], dict):
                merge_dict(target[key], source[key], path + [str(key)], changes=changes)
            elif target[key] == source[key]:
                pass  # same leaf value
            else:
                target[key] = source[key]
                changes.append(f"Overwrite -> {'.'.join(path + [str(key)])}")
        else:
            target[key] = source[key]
            changes.append(f"Create -> {'.'.join(path + [str(key)])}")
    return target, changes


def modify_config(global_dict, config_file_path):
    modify_file = os.path.join(os.path.dirname(config_file_path), "modify.data")
    if os.path.exists(modify_file):
        with open(modify_file, "rb") as file:
            modify = dill.load(file)
            require_global_present = modify["!require_global_present"]
            for key, value in modify.items():
                if not key.startswith("!"):
                    if require_global_present and key not in global_dict:
                        raise ValueError(
                            f"Key {key} is not present in global config context"
                        )
                    print(f"Update global: {key}")
                    if isinstance(value, dict) and isinstance(global_dict[key], dict):
                        merged_dict, changes = merge_dict(global_dict[key], value)
                        global_dict[key] = merged_dict
                        for change in changes:
                            print(change)
                    else:
                        global_dict[key] = value
    else:
        print("No modification file detected, skipping")


def create_config_modifier(
    modifier_dict: Dict[str, Any], experiment_dir: Any, require_global_present=True
):
    with open(os.path.join(experiment_dir, "modify.data"), "wb") as file:
        modifier_dict["!require_global_present"] = require_global_present
        dill.dump(modifier_dict, file)
