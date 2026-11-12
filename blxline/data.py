"""Download of the published datasets (Zenodo record 10.5281/zenodo.20312678)."""
import hashlib
import os
import urllib.request

ZENODO_RECORD = 'https://zenodo.org/api/records/20312678'
ZENODO_FILES = {
    # Map dataset of the paper (output of step 4, 1.1 GB): the input of 05_maps_and_xline and 07_interactive_xlines.
    # Same file as BL_data_for_maps_without_cusps_withoutNp<3_all_sat_1min_2025-05-06.pkl used for the paper.
    'BL_dataset_multisat_limited.pkl': 'md5:27f6513540dd78ac8dbd2fa342053790',
    # Full BL dataset (output of step 3, 19 GB).
    'BL_dataset_multisat.pkl': 'md5:99e2a3892146ec89f7c2e9de14664cf8',
    # Cold ions dataset (1 GB).
    'coldions_multisat.pkl': 'md5:f222c6c3eb0d46319124ebf6c4e7b735',
}


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return 'md5:' + h.hexdigest()


def download(name, folder):
    """Download a file of the Zenodo record into folder (if it is not already there) and check its checksum.
    Returns its path."""
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return path
    os.makedirs(folder, exist_ok=True)
    url = f'{ZENODO_RECORD}/files/{name}/content'
    print(f'Downloading {url}\n  to {path} ...', flush=True)
    urllib.request.urlretrieve(url, path + '.part')
    if _md5(path + '.part') != ZENODO_FILES[name]:
        raise IOError(f'{path}.part: wrong checksum, the download is incomplete or corrupted. Delete it and retry.')
    os.replace(path + '.part', path)
    return path
