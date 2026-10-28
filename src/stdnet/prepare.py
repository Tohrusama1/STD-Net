"""Create disjoint stratified flow splits, optionally for one attack scenario."""
import argparse
import hashlib
import json
from pathlib import Path


def split_flows(frame, train_fraction=0.7, test_fraction=0.3, seed=42):
    from sklearn.model_selection import train_test_split
    from .preprocessing import binary_labels
    if not 0 < train_fraction < 1 or not 0 < test_fraction < 1 or train_fraction + test_fraction > 1 + 1e-9:
        raise ValueError('Fractions must be positive and sum to at most one')
    frame = frame.drop_duplicates().reset_index(drop=True)
    label_column = next((c for c in ['Label', 'flag', 'label'] if c in frame), None)
    if label_column is None:
        raise ValueError('Split preparation requires Label, label or flag')
    labels = binary_labels(frame, label_column)
    pool, test = train_test_split(frame, test_size=test_fraction, random_state=seed, stratify=labels)
    if train_fraction < 1 - test_fraction - 1e-9:
        train, _ = train_test_split(pool, train_size=train_fraction / (1 - test_fraction),
                                    random_state=seed, stratify=binary_labels(pool, label_column))
    else:
        train = pool
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--attack', help='Exact text attack label to combine with normal flows, e.g. DoS')
    parser.add_argument('--train-fraction', type=float, default=0.7,
                        help='Fraction of the whole deduplicated scenario, not of the training pool')
    parser.add_argument('--test-fraction', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    from .preprocessing import read_flows, binary_labels
    frame = read_flows(args.data)
    if args.attack:
        column = next((c for c in ['Label', 'flag', 'label'] if c in frame), None)
        if column is None or not frame[column].astype(str).str.strip().eq(args.attack).any():
            raise ValueError('Requested attack label was not found')
        frame = frame[(binary_labels(frame, column) == 0) | frame[column].astype(str).str.strip().eq(args.attack)]
    train, test = split_flows(frame, args.train_fraction, args.test_fraction, args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Split output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output / 'train.csv', index=False)
    test.to_csv(args.output / 'test.csv', index=False)
    report = {'seed': args.seed, 'attack': args.attack, 'train_fraction': args.train_fraction,
              'test_fraction': args.test_fraction, 'input_rows': len(frame),
              'unique_rows': len(frame.drop_duplicates()), 'train_rows': len(train), 'test_rows': len(test),
              'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'split_unit': 'deduplicated flow row; users can overlap'}
    (args.output / 'split.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
