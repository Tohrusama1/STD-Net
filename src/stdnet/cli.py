"""Command-line training and held-out evaluation."""
import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='STD-Net network anomaly detection')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='Fit preprocessing and train a model')
    train.add_argument('--data', required=True, type=Path)
    train.add_argument('--config', required=True, type=Path)
    train.add_argument('--output', required=True, type=Path)
    evaluate = sub.add_parser('evaluate', help='Evaluate a trusted locally generated run')
    evaluate.add_argument('--data', required=True, type=Path)
    evaluate.add_argument('--run', required=True, type=Path)
    evaluate.add_argument('--output', type=Path)
    evaluate.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    # Keep help usable even before the numerical stack is installed.
    if args.command == 'train':
        run_train(args)
    else:
        run_evaluate(args)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def run_train(args):
    import platform
    import numpy as np
    import torch
    import torch_geometric
    from .preprocessing import FlowPreprocessor, read_flows
    from .graph import build_graphs
    from .model import STDNet
    from .losses import HeteroscedasticFocalLoss

    config = json.loads(args.config.read_text(encoding='utf-8'))
    settings = config['training']
    if settings['epochs'] < 1:
        raise ValueError('epochs must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output directory must be empty; choose a new run directory')
    seed_everything(settings['seed'])
    device = device_for(settings['device'])
    flows = read_flows(args.data)
    preprocessor = FlowPreprocessor(**config['preprocessing']).fit(flows)
    graphs = build_graphs(*preprocessor.transform(flows), label_rule=config['label_rule'],
                          anomaly_ratio=preprocessor.anomaly_ratio_)
    labels = torch.cat([g.y for g in graphs])
    pos_weight = float((labels == 0).sum() / (labels == 1).sum().clamp(min=1))
    if labels.unique().numel() != 2:
        raise ValueError('Aggregated training labels must contain both classes')
    model = STDNet(len(preprocessor.feature_columns_), config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'])
    loss_fn = HeteroscedasticFocalLoss(pos_weight=pos_weight, **config['loss']).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'config.json', config)
    with (args.output / 'preprocessor.pkl').open('wb') as stream:
        pickle.dump(preprocessor, stream)
    write_json(args.output / 'features.json', {'columns': preprocessor.feature_columns_, 'iv': preprocessor.iv_,
                                             'grouping': preprocessor.grouping_, 'anomaly_ratio': preprocessor.anomaly_ratio_})
    history, best = [], float('inf')
    for epoch in range(settings['epochs']):
        model.train()
        losses = []
        for graph in graphs:
            data = graph.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, variance = model(data)
            loss = loss_fn(mean, variance, data.y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect data and hyperparameters')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.clamp_(-settings['grad_clip_value'], settings['grad_clip_value'])
            optimizer.step()
            losses.append(loss.item())
        average = float(np.mean(losses))
        history.append({'epoch': epoch + 1, 'train_loss': average})
        print(f'Epoch {epoch + 1}/{settings["epochs"]}: train_loss={average:.6f}', flush=True)
        if average < best:
            best = average
            torch.save(model.state_dict(), args.output / 'model.pt')
        write_json(args.output / 'history.json', history)
    write_json(args.output / 'run.json', {
        'python': platform.python_version(), 'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__,
        'seed': settings['seed'], 'device': str(device), 'num_users': len(graphs),
        'num_timestamps': int(labels.numel()), 'pos_weight': pos_weight,
        'selection': 'lowest mean per-user training loss', 'best_train_loss': best,
        'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'format_version': 1,
    })


def run_evaluate(args):
    import numpy as np
    import pandas as pd
    import torch
    from .preprocessing import read_flows
    from .graph import build_graphs
    from .model import STDNet
    from .losses import logistic_normal_probability
    from .metrics import classification_metrics

    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    seed_everything(config['training']['seed'])
    # This pickle is created by train. Only evaluate runs from trusted sources.
    with (args.run / 'preprocessor.pkl').open('rb') as stream:
        preprocessor = pickle.load(stream)
    graphs = build_graphs(*preprocessor.transform(read_flows(args.data)), label_rule=config['label_rule'],
                          anomaly_ratio=preprocessor.anomaly_ratio_)
    device = device_for(args.device)
    model = STDNet(len(preprocessor.feature_columns_), config['model']).to(device)
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location=device, weights_only=True))
    model.eval()
    rows = []
    with torch.no_grad():
        for graph in graphs:
            mean, variance = model(graph.to(device))
            probs = logistic_normal_probability(mean, variance).cpu().numpy().ravel()
            labels = graph.y.cpu().numpy().ravel()
            std = variance.sqrt().cpu().numpy().ravel()
            rows.extend({'user_id': graph.user_id, 'node_index': i, 'label': int(y),
                         'probability': float(p), 'latent_std': float(s)}
                        for i, (y, p, s) in enumerate(zip(labels, probs, std)))
    frame = pd.DataFrame(rows)
    metrics = classification_metrics(frame.label, frame.probability, config['threshold'])
    metrics['mean_latent_std'] = float(frame.latent_std.mean())
    metrics['input_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    output = args.output or args.run / 'evaluation'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Evaluation output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
