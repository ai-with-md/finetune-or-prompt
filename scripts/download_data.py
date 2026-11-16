"""Download Banking77 into data/ and check the files are exactly the ones this study used.

Banking77 (Casanueva et al., 2020) is published by PolyAI under CC BY 4.0:
https://huggingface.co/datasets/PolyAI/banking77
The revision is pinned and every file is checked against its SHA-256, so the
numbers in the notebooks can be reproduced byte for byte.
"""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPO = "PolyAI/banking77"
REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"

FILES = {
    "train.parquet": ("data/train-00000-of-00001.parquet",
                      "8dc89b87a188509c42cce5a086e8f7c1ff4a225bdd9ae3f23e6cdfa5806e363a"),
    "test.parquet": ("data/test-00000-of-00001.parquet",
                     "1f40f777c33f0edfa6c530ff84fe2878db6490838c12d22d036f626948905341"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_labels(target: Path) -> None:
    """Intent names come from the dataset card's metadata at the pinned revision."""
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"
    with urlopen(url) as response:
        card = json.load(response)["cardData"]
    names = card["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
    labels = [names[str(i)] for i in range(len(names))]
    if len(labels) != 77:
        raise RuntimeError(f"Expected 77 intents, found {len(labels)}")
    target.write_text(json.dumps(labels, indent=1) + "\n")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, (remote, expected) in FILES.items():
        target = DATA_DIR / name
        if not target.exists():
            print(f"Downloading {name}")
            urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}", target)
        if sha256(target) != expected:
            target.unlink()
            raise RuntimeError(f"{name} does not match the expected checksum; deleted it, please rerun")
        print(f"OK {name} ({target.stat().st_size / 1e3:.0f} KB)")
    labels = DATA_DIR / "labels.json"
    if not labels.exists():
        download_labels(labels)
    print(f"OK labels.json ({len(json.loads(labels.read_text()))} intents)")


if __name__ == "__main__":
    main()
