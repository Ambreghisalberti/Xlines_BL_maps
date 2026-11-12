"""Where this repo reads/writes its files. Override with the environment variables BLXLINE_OUT / BLXLINE_MODEL_LINES.

Defaults: the folders of kaa if they exist, else the folder `data/` of the repo (put the files downloaded from
Zenodo there, see the README)."""
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default(kaa_path, local_path):
    return kaa_path if os.path.isdir(kaa_path) else local_path


# Everything produced by the notebooks of this repo goes under OUT (never over the original files).
OUT = os.environ.get('BLXLINE_OUT', _default('/DATA/ghisalberti/BL_Xline_minimal_outputs', os.path.join(_REPO, 'data')))
# X-lines of the models (DXL, MRL, MSL) used in the comparison figures (read only).
MODEL_LINES = os.environ.get('BLXLINE_MODEL_LINES', _default('/home/ghisalberti/komar_separators',
                                                              os.path.join(_REPO, 'data', 'model_xlines')))
for d in ['all_pred', 'xlines', 'figures']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
