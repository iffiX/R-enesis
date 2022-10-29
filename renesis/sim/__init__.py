import os
import sys
import cmake
import shutil
import subprocess
import importlib.util
from checksumdir import dirhash

# Set this if gcc is not the default compiler
# os.environ["CC"] = "/usr/bin/gcc-9"
# os.environ["CXX"] = "/usr/bin/g++-9"


def build_module(src_path, build_path, ignore_build_error=False):
    """
    Args:
        src_path: The C/C++ source path where CMakeLists.txt is located.
        build_path: The temporary build directory to use
    """
    md5hash = dirhash(
        src_path,
        "md5",
        # If you ever changed the CMakeLists.txt, you need to delete the build directory
        excluded_extensions=["txt", "so"],
        excluded_files=["cmake-build-debug", ".idea"],
    )
    build = True
    if os.path.exists(os.path.join(build_path, "hash.txt")):
        with open(os.path.join(build_path, "hash.txt"), "r") as file:
            build = file.read() != md5hash
    if build:
        shutil.rmtree(build_path, ignore_errors=True)
        os.makedirs(build_path)
        subprocess.call(
            [
                os.path.join(cmake.CMAKE_BIN_DIR, "cmake"),
                "-S",
                src_path,
                "-B",
                build_path,
            ]
        )
        subprocess.call(["make", "-C", build_path, "clean"])
        if subprocess.call(["make", "-C", build_path, "-j4"]) != 0:
            if not ignore_build_error:
                raise RuntimeError("Make failed")
            else:
                print("Make failed, error ignored")
        else:
            subprocess.call(["make", "-C", build_path, "install"])
            with open(os.path.join(build_path, "hash.txt"), "w") as file:
                file.write(md5hash)

    # In case there are additional dynamic libraries
    old = os.environ.get("LD_LIBRARY_PATH")
    if old:
        os.environ["LD_LIBRARY_PATH"] = old + ":" + src_path
    else:
        os.environ["LD_LIBRARY_PATH"] = src_path


_dir_path = str(os.path.dirname(os.path.abspath(__file__)))
_voxcraft_src_path = str(os.path.join(_dir_path, "voxcraft_src"))
_voxcraft_build_path = str(os.path.join(_dir_path, "build"))
voxcraft_bin_path = os.path.join(_voxcraft_build_path, "voxcraft-sim")

_voxcraft_viz_src_path = str(os.path.join(_dir_path, "voxcraft_viz_src"))
_voxcraft_viz_build_path = str(os.path.join(_dir_path, "build_viz"))

sys.path.append(_voxcraft_src_path)
sys.path.append(_voxcraft_viz_src_path)

build_module(_voxcraft_src_path, _voxcraft_build_path, True)
build_module(_voxcraft_viz_src_path, _voxcraft_viz_build_path)
_vxz = importlib.import_module("voxcraft_viz")
VXHistoryRenderer = _vxz.VXHistoryRenderer
