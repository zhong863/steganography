"""Audit paired final results and estimate prompt-sampling uncertainty."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from grdh_eval.runner import summarize, write_json


def score_means(means):
    """Vectorized slide formula on [bpp, jpeg, awgn, median, NIQE] means."""
    means = np.asarray(means)
    return (30 * np.minimum(1, np.log1p(1000 * means[..., 0]) / np.log(1001))
            + 50 / 3 * np.maximum(0, 2 * means[..., 1:4] - 1).sum(axis=-1)
            + 20 * np.clip((8 - means[..., 4]) / 5, 0, 1))


def paired_bootstrap(reference, candidate, draws=10000, seed=7361):
    keys = ('bpp', 'acc_jpeg', 'acc_awgn', 'acc_mblur', 'niqe')
    if len(reference) != len(candidate) or not reference:
        raise ValueError('Paired samples must have equal positive length')
    if [(r['index'], r['source_line']) for r in reference] != [(r['index'], r['source_line']) for r in candidate]:
        raise ValueError('Paired prompt identities or order differ')
    left = np.array([[r[k] for k in keys] for r in reference])
    right = np.array([[r[k] for k in keys] for r in candidate])
    sampled = np.random.RandomState(seed).randint(0, len(left), (draws, len(left)))
    differences = score_means(right[sampled].mean(axis=1)) - score_means(left[sampled].mean(axis=1))
    interval = np.percentile(differences, [2.5, 97.5]).tolist()
    return {'difference': float(score_means(right.mean(axis=0)) - score_means(left.mean(axis=0))),
            'paired_bootstrap_95_interval': interval, 'draws': draws, 'seed': seed,
            'note': 'Resamples paired prompts; fixed messages and keys. Does not cover unknown TA protocol or dataset shift.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/optimization'))
    args = parser.parse_args()
    root = args.root
    final = root / 'final'
    manifests, rows = {}, {}
    for name in ('reference', 'optimized'):
        manifests[name] = json.loads((final / name / 'manifest.json').read_text())
        rows[name] = [json.loads(x) for x in (final / name / 'per_image.jsonl').read_text().splitlines()]
        if len(rows[name]) != 100 or manifests[name]['split'] != 'final':
            raise ValueError('Final confirmation must contain all 100 prompts')
    for key in ('prompts', 'prompts_sha256', 'seed', 'attacks', 'niqe', 'count'):
        if manifests['reference'][key] != manifests['optimized'][key]:
            raise ValueError('Paired final protocol differs: ' + key)
    for left, right in zip(rows['reference'], rows['optimized']):
        if left['payload_bits'] == right['payload_bits'] and left['message_sha256'] != right['message_sha256']:
            raise ValueError('Equal-length paired payloads should use identical secret bits')
    selection = json.loads((root / 'selection.json').read_text())
    frozen = selection['frozen_final_candidate']
    for section in ('model', 'codec'):
        if frozen[section] != manifests['optimized'][section]:
            raise ValueError('Final configuration differs from frozen validation winner')
    report = {'reference': summarize(rows['reference'], 100, 'final'),
              'optimized': summarize(rows['optimized'], 100, 'final'),
              'comparison': paired_bootstrap(rows['reference'], rows['optimized']),
              'selected_candidate': frozen, 'validation_scores': selection['validation_scores']}
    for name in rows:
        accuracy = np.array([[r['acc_' + a] for a in ('jpeg', 'awgn', 'mblur')] for r in rows[name]])
        report[name]['per_prompt_mean_attack_acc_min'] = float(accuracy.mean(axis=1).min())
        report[name]['prompts_below_90_percent_mean_acc'] = int((accuracy.mean(axis=1) < .9).sum())
    table = []
    for stage in ('screen', 'ecc', 'solver', 'sdpa', 'blocks', 'denoise', 'ema', 'validation', 'final'):
        for path in sorted((root / stage).glob('*/summary.json')):
            value = json.loads(path.read_text())
            if 'score' in value:
                table.append(dict(stage=stage, candidate=path.parent.name, images=value['images'],
                                  bpp=value['bpp'], niqe=value['mean_niqe'], **value['accuracies'],
                                  **value['score'], seconds=value['compute_seconds']))
    report['completed_image_evaluations'] = sum(r['images'] for r in table)
    report['timing_note'] = 'Evaluation count includes reused carriers and multiple settings on the same development prompts.'
    write_json(root / 'comparison.json', report)
    with (root / 'all_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
