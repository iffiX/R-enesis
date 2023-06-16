import os
import json
import shutil
import fnmatch
import tempfile
from typing import Tuple
from shutil import which
from . import PROJECT_ROOT

CONFIG = {}
INCLUDE = [
    "experiments/*",
    "launch/*",
    "renesis/*",
    "COMMENT.txt",
    "init.sh",
    "install.sh",
    "launch_config.json",
    "LAUNCH_COMMAND.sh",
    "main.py",
    "README.md",
    "requirements.txt",
    "modify.*",
]
EXCLUDE = ["*__pycache__*"]
INCLUDE_CODE_ONLY = INCLUDE
EXCLUDE_CODE_ONLY = [
    "renesis/sim/build*",
    "*__pycache__*",
]


def init_config():
    global CONFIG
    config_path = os.path.join(PROJECT_ROOT, "launch_config.json")

    config = {}
    loaded = False
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as file:
                config = json.load(file)
                loaded = True
        except:
            config = {}

    if not loaded:
        # create configurations
        config["editor"] = get_user_preference_for_editor()

    with open(config_path, "w") as file:
        json.dump(config, file)
    CONFIG = config


def get_user_preference_for_editor() -> str:
    """
    Prompt user to select an editor, return the path to the editor.
    """
    editors = list(
        {which("nano"), which("vim"), which("gedit"), which("emacs")} - {None}
    )
    if editors:
        print("These text editors are found:")
        for idx, editor in enumerate(editors):
            print(f"({idx+1}) {editor}")
        selected = None
        while selected is None:
            if len(editors) == 1:
                prompt = (
                    "Select editor with number 1, "
                    "or input custom editor path with number 0: "
                )
            else:
                prompt = (
                    f"Select editor with number 1 to {len(editors)}, "
                    f"or input custom editor path with number 0: "
                )
            selected_id = input(prompt)
            try:
                selected_id = int(selected_id)
                if not 0 <= selected_id <= len(editors):
                    raise ValueError()
            except:
                continue

            if selected_id == 0:
                path = input("Path: ")
                if os.path.exists(path):
                    selected = path
                else:
                    continue
            else:
                selected = editors[selected_id - 1]
        return selected

    else:
        selected = None
        print("No text editors are found, please input a custom one: ")
        while selected is None:
            path = input("Path: ")
            if os.path.exists(path):
                selected = path
            else:
                continue
        return selected


def get_snapshot_comment_file() -> Tuple[str, str]:
    """
    Prompt user to create a comment file in a temporary directory
    and return the path to the created comment file. The temporary
    directory is returned as the second argument for you to delete.
    """
    comment_dir = tempfile.mkdtemp()
    comment_file = os.path.join(comment_dir, "COMMENT.txt")
    with open(comment_file, "w") as file:
        file.write("# Write any necessary comments, and close the editor")
    code = os.system(f"{CONFIG['editor']} {comment_file}")
    if code != 0:
        raise ValueError(f"Editor exited with code {code}")
    return comment_file, comment_dir


def get_snapshot(code_only=True) -> str:
    """
    Create a snapshot of the project root in a temporary directory
    and return the path to that temporary directory.

    Args:
        code_only: If set to true, only copy running code, otherwise
            also copy pre-built c++ modules.
    """

    def failed(exc):
        raise exc

    snapshot_dir = tempfile.mkdtemp()
    for dir_path, _dirs, files in os.walk(PROJECT_ROOT, topdown=True, onerror=failed):
        rel_path = (
            dir_path.replace(PROJECT_ROOT + "/", "") if dir_path != PROJECT_ROOT else ""
        )
        include_set = set()
        exclude_set = set()
        for patterns, file_set in (
            (INCLUDE if not code_only else INCLUDE_CODE_ONLY, include_set),
            (EXCLUDE if not code_only else EXCLUDE_CODE_ONLY, exclude_set),
        ):
            for pat in patterns:
                file_set.update(
                    fnmatch.filter(
                        [os.path.join(rel_path, file) for file in files], pat
                    )
                )
        include_set = include_set.difference(exclude_set)

        if include_set:
            target_dir = os.path.join(snapshot_dir, rel_path)

            os.makedirs(target_dir, exist_ok=True)
            for file in include_set:
                shutil.copy2(os.path.join(PROJECT_ROOT, file), target_dir)

    return snapshot_dir
