"""Download official UCI 350 data with TLS and a checked file hash.

The hash pins the file retrieved during this project; it is not a publisher
signature. If the official release changes, inspect it rather than auto-repinning.
"""
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
import zipfile

URL = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
EXPECTED = '30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933'


def ensure_workbook(path):
    """Return the path to the verified workbook, downloading it if needed."""
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError('Existing data hash mismatch; inspect manually')
        return path
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        matches = [n for n in archive.namelist() if n.endswith('.xls')]
        if len(matches) != 1:
            raise ValueError('Unexpected archive layout')
        data = archive.read(matches[0])
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError('Downloaded release differs; investigate before use')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging_dir:
        temporary_path = Path(staging_dir) / path.name
        temporary_path.write_bytes(data)
        temporary_path.replace(path)
    return path


if __name__ == '__main__':
    ensure_workbook(Path(__file__).resolve().parent / 'data/default_of_credit_card_clients.xls')
    print('Official UCI data match the recorded SHA256.')
