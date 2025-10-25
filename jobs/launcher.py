# launcher.py
import os, sys, runpy

wd = os.getcwd()

# Make the src package importable in BOTH cases:
# 1) You supplied src.zip as an archive with "#src"  -> extracted folder ./src
# 2) You supplied src.zip via --py-files             -> already on PYTHONPATH (these inserts are harmless)

# Common archive layouts to support
candidates = [
    os.path.join(wd, "src"),          # when you used: .../src.zip#src
    os.path.join(wd, "src", "src"),   # if the zip itself contains a top-level "src/" folder
]
for p in candidates:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# If you shipped config as conf.zip#conf, keep it easy to locate
conf_dir = os.path.join(wd, "conf")
if os.path.isdir(conf_dir) and conf_dir not in sys.path:
    sys.path.insert(0, conf_dir)

# Run the actual job script you uploaded
runpy.run_path("train_als_local.py", run_name="__main__")
