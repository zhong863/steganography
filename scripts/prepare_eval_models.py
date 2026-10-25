"""Download SD 1.5 public assets into this workspace, with resume and hashes.

The checkpoint already contains the CLIP text weights. The evaluation loader
initializes CLIP from its config/tokenizer and loads those checkpoint weights,
so a second copy of the CLIP weight file is unnecessary.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SMALL_FILES = {
    "text_encoder/config.json": "clip/config.json",
    "tokenizer/merges.txt": "clip/merges.txt",
    "tokenizer/vocab.json": "clip/vocab.json",
    "tokenizer/tokenizer_config.json": "clip/tokenizer_config.json",
    "tokenizer/special_tokens_map.json": "clip/special_tokens_map.json",
}
CHECKPOINT = "v1-5-pruned-emaonly.safetensors"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_size=None, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError("Existing file has the wrong size: %s" % path)
        digest = sha256_file(path)
        if expected_sha and digest != expected_sha:
            raise ValueError("Existing file has the wrong SHA256: %s" % path)
        print("EXISTS", path, flush=True)
        return digest
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": "bytes=%d-" % offset} if offset else {})
    with urllib.request.urlopen(request, timeout=45) as response:
        append = offset > 0 and response.status == 206
        received = offset if append else 0
        reported = time.monotonic()
        with partial.open("ab" if append else "wb") as stream:
            while True:
                chunk = response.read(1024 ** 2)
                if not chunk:
                    break
                stream.write(chunk)
                received += len(chunk)
                if time.monotonic() - reported > 10:
                    print("DOWNLOAD", path.name, "%0.1f MiB" % (received / 1024 ** 2), flush=True)
                    reported = time.monotonic()
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ValueError("Incomplete download; rerun to resume: %s" % partial)
    digest = sha256_file(partial)
    if expected_sha and digest != expected_sha:
        raise ValueError("SHA256 mismatch: %s" % partial)
    partial.replace(path)
    print("READY", path, path.stat().st_size, digest, flush=True)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/sd15"))
    parser.add_argument("--metadata-only", action="store_true", help="Download tokenizer/config but omit the large checkpoint")
    args = parser.parse_args()
    info_url = "https://huggingface.co/api/models/%s/revision/%s?blobs=true" % (REPOSITORY, REVISION)
    with urllib.request.urlopen(info_url, timeout=30) as response:
        info = json.load(response)
    remote = {item["rfilename"]: item for item in info["siblings"]}
    files = dict(SMALL_FILES)
    if not args.metadata_only:
        files[CHECKPOINT] = CHECKPOINT
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": {}}
    for source, destination in files.items():
        metadata = remote[source]
        lfs = metadata.get("lfs") or {}
        path = args.output / destination
        url = "https://huggingface.co/%s/resolve/%s/%s" % (REPOSITORY, REVISION, source)
        digest = download(url, path, lfs.get("size", metadata.get("size")), lfs.get("sha256"))
        manifest["files"][destination] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        (args.output / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
