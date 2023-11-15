import os
from modules import paths # only modules without dependencies can be imported here as its executed before server start


def preload(parser):
    pass
    # parser.add_argument("--nudenet-dir", type=str, help="Path to directory with NudeNet model", default=os.path.join(paths.models_path, 'Lora'))
