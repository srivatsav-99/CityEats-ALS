#launcher.py
import os, sys, runpy

wd = os.getcwd()

#common archive layouts to support
candidates = [
    os.path.join(wd, "src"),
    os.path.join(wd, "src", "src"),   #if the zip itself contains a top-level "src/" folder
]
for p in candidates:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

#keeping it easy to locate
conf_dir = os.path.join(wd, "conf")
if os.path.isdir(conf_dir) and conf_dir not in sys.path:
    sys.path.insert(0, conf_dir)

#running the actual job script that was uploaded
runpy.run_path("train_als_local.py", run_name="__main__")
