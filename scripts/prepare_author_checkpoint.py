#!/usr/bin/env python3
"""Extract author-linked inference tensors without storing the full 7.7 GB checkpoint.

Downloads selected byte ranges from the author's publicly linked PyTorch ZIP,
validates every extracted tensor against its ZIP CRC, and writes safetensors.
The original EMA shadow tensors are excluded because ldm.yaml uses use_ema=False.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import pickle
import struct
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import requests
import torch
from safetensors import safe_open

SOURCE = 'https://drive.usercontent.google.com/download?id=1ISeumzN-JrhyAacOPlh1IjAv6MKffogU&export=download&confirm=t'
SIZE = 7703807346
CHUNK = 16 * 1024 * 1024


def get_range(start, end):
    for attempt in range(5):
        try:
            response = requests.get(SOURCE, headers={'Range': 'bytes=%d-%d' % (start, end)}, timeout=(15, 20))
            response.raise_for_status()
            expected = 'bytes %d-%d/%d' % (start, end, SIZE)
            if response.status_code != 206 or response.headers.get('Content-Range') != expected or len(response.content) != end-start+1:
                raise ValueError('Invalid range response: %s' % response.headers.get('Content-Range'))
            return response.content
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


class RemoteFile(io.RawIOBase):
    def __init__(self):
        self.position = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position
    def read(self, n=-1):
        if n < 0: n = SIZE - self.position
        if n == 0: return b''
        end = min(SIZE - 1, self.position + n - 1)
        data = get_range(self.position, end)
        self.position = end + 1
        return data


class PlaceholderCheckpoint:
    pass


def tensor_metadata(storage, offset, size, stride, *args):
    return {'storage': storage, 'offset': offset, 'size': size, 'stride': stride}


class MetadataUnpickler(pickle.Unpickler):
    """Only resolve globals actually present in this known public checkpoint."""
    def persistent_load(self, pid):
        return pid
    def find_class(self, module, name):
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return tensor_metadata
        if (module, name) == ('pytorch_lightning.callbacks.model_checkpoint', 'ModelCheckpoint'):
            return PlaceholderCheckpoint
        permitted = {('collections', 'OrderedDict'), ('_codecs', 'encode'),
                     ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
                     ('torch', 'FloatStorage'), ('torch', 'IntStorage'), ('torch', 'LongStorage')}
        if (module, name) not in permitted:
            raise pickle.UnpicklingError('Unexpected global %s.%s' % (module, name))
        return super().find_class(module, name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('models/sd15/author-nonema-inference.safetensors'))
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    out = args.output
    partial = out.with_name(out.name + '.part')
    state_path = out.with_name(out.name + '.part.json')
    if out.exists():
        raise FileExistsError('Refusing to overwrite %s' % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(RemoteFile())
    metadata = MetadataUnpickler(io.BytesIO(archive.read('archive/data.pkl'))).load()
    original = metadata['state_dict']
    chosen = {k: v for k, v in original.items() if not k.startswith('model_ema.')}
    records = []
    entries = {}
    current = 0
    for name, meta in sorted(chosen.items()):
        storage = meta['storage']
        if meta['offset'] != 0 or storage[1] not in (torch.FloatStorage, torch.LongStorage):
            raise ValueError('Unexpected storage for ' + name)
        dtype = 'F32' if storage[1] is torch.FloatStorage else 'I64'
        size = int(np.prod(meta['size'])) * (4 if dtype == 'F32' else 8)
        step = 1
        for dim, stride in zip(meta['size'][::-1], meta['stride'][::-1]):
            if dim != 1 and stride != step: raise ValueError('Non-contiguous tensor ' + name)
            step *= dim
        info = archive.getinfo('archive/data/' + storage[2])
        if info.compress_type != zipfile.ZIP_STORED or info.file_size != size:
            raise ValueError('Unexpected archive entry for ' + name)
        # PyTorch pads each local ZIP header's extra field to a 64-byte boundary.
        start = ((info.header_offset + 30 + len(info.filename.encode()) + 4 + 63) // 64) * 64
        entries[name] = {'dtype': dtype, 'shape': list(meta['size']), 'data_offsets': [current, current + size]}
        records.append((name, start, size, current, info.CRC))
        current += size
    header = json.dumps(entries, separators=(',', ':')).encode('utf-8')
    header += b' ' * ((8 - len(header) % 8) % 8)
    data_offset = 8 + len(header)
    jobs = {}
    for name, start, size, destination, crc in records:
        pos = start
        while pos < start + size:
            end = min(start + size, (pos // CHUNK + 1) * CHUNK)
            index = pos // CHUNK
            jobs.setdefault(index, []).append((pos, end, data_offset + destination + pos - start))
            pos = end
    identity = hashlib.sha256((hashlib.sha256(header).hexdigest() + str(SIZE) + str(len(jobs))).encode()).hexdigest()
    if partial.exists() or state_path.exists():
        if not (partial.exists() and state_path.exists()): raise RuntimeError('Incomplete resume files')
        state = json.loads(state_path.read_text())
        if state['identity'] != identity or partial.stat().st_size != data_offset + current:
            raise RuntimeError('Existing partial download differs')
        done = set(state['done'])
    else:
        with partial.open('wb') as f:
            f.write(struct.pack('<Q', len(header)))
            f.write(header)
            f.truncate(data_offset + current)
        done = set()
        state_path.write_text(json.dumps({'identity': identity, 'done': []}))
    print('Preparing %d tensors, %.2f GB, %d range groups; %d complete' % (len(records), current / 1e9, len(jobs), len(done)), flush=True)
    fd = os.open(partial, os.O_WRONLY)
    def download(item):
        index, ranges = item
        first = min(x[0] for x in ranges)
        last = max(x[1] for x in ranges)
        data = get_range(first, last - 1)
        for start, end, dest in ranges:
            block = memoryview(data)[start-first:end-first]
            written = 0
            while written < len(block):
                n = os.pwrite(fd, block[written:], dest + written)
                if n <= 0: raise IOError('Short write')
                written += n
        return index
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download, item) for item in jobs.items() if item[0] not in done]
            for future in concurrent.futures.as_completed(futures):
                done.add(future.result())
                tmp = state_path.with_suffix(state_path.suffix + '.tmp')
                tmp.write_text(json.dumps({'identity': identity, 'done': sorted(done)}))
                tmp.replace(state_path)
                if len(done) % 5 == 0 or len(done) == len(jobs) or len(done) > len(jobs) - 5:
                    print('Downloaded %d/%d ranges' % (len(done), len(jobs)), flush=True)
    finally:
        os.close(fd)
    print('Validating all source ZIP CRCs', flush=True)
    with partial.open('rb') as f:
        for name, start, size, dest, crc in records:
            f.seek(data_offset + dest)
            left = size
            actual = 0
            while left:
                chunk = f.read(min(left, 4 * 1024 * 1024))
                if not chunk: raise IOError('Incomplete tensor ' + name)
                actual = zlib.crc32(chunk, actual)
                left -= len(chunk)
            if actual != crc: raise ValueError('Source CRC mismatch: ' + name)
    with safe_open(str(partial), framework='pt', device='cpu') as f:
        if set(f.keys()) != set(chosen): raise ValueError('Safetensors key mismatch')
    digest = hashlib.sha256()
    with partial.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): digest.update(chunk)
    partial.replace(out)
    state_path.unlink(missing_ok=True)
    provenance = {'source': 'https://drive.google.com/file/d/1ISeumzN-JrhyAacOPlh1IjAv6MKffogU/view',
                  'source_bytes': SIZE, 'output_bytes': out.stat().st_size, 'sha256': digest.hexdigest(),
                  'tensor_count': len(records), 'validation': 'all tensor bytes match source ZIP CRC-32',
                  'excluded': 'model_ema.*; author config uses use_ema=False',
                  'global_step': metadata.get('global_step')}
    out.with_suffix('.sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
