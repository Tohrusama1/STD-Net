"""Fit preprocessing on training flows, then reuse it unchanged on held-out flows."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


NORMAL_LABELS = {'normal', 'benign', '0', '0.0'}
IDENTIFIERS = {'Src IP', 'Dst IP', 'Flow ID', 'sc_user_id'}
LABEL_COLUMNS = {'Label', 'label', 'flag', 'Class', 'Anomaly', 'Result'}


def read_flows(path):
    for encoding in ('utf-8-sig', 'gbk', 'latin1'):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            frame.columns = frame.columns.str.strip()
            if frame.columns.duplicated().any():
                raise ValueError('Duplicate column names after stripping whitespace')
            if frame.empty:
                raise ValueError('Input CSV is empty')
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError('Unable to decode input CSV')


def binary_labels(frame, column):
    if column not in frame:
        raise ValueError(f'Missing label column: {column}')
    labels = frame[column].astype('string').str.strip().str.lower()
    if labels.isna().any() or labels.eq('').any():
        raise ValueError('Labels cannot be missing or empty')
    numeric = pd.to_numeric(labels, errors='coerce')
    if numeric.notna().any():
        if not numeric.notna().all() or not numeric.isin([0, 1]).all():
            raise ValueError('Numeric labels must be binary 0/1; do not mix text and numeric labels')
        return numeric.astype(int)
    # Dataset attack names (including unseen attack names) map to one.
    return (~labels.isin(NORMAL_LABELS)).astype(int)


def information_value(values, labels, bins=10):
    """Smoothed WoE/IV, finite even when a bin contains only one class."""
    if labels.nunique() < 2 or values.nunique() < 2:
        return 0.0
    groups = values if values.nunique() <= 5 else pd.qcut(values, bins, duplicates='drop')
    table = pd.DataFrame({'bin': groups, 'y': labels}).groupby('bin', observed=True).y.agg(['sum', 'count'])
    attack = table['sum'] + 0.5
    normal = table['count'] - table['sum'] + 0.5
    attack, normal = attack / attack.sum(), normal / normal.sum()
    return float(((normal - attack) * np.log(normal / attack)).sum())


class FlowPreprocessor:
    """Numeric flow features, chronological engineering, IV and standardization.

    Adaptive grouping is chosen with training labels only. User identifiers never
    contain labels; grouping, features and the scaler are frozen during transform.
    """

    def __init__(self, iv_min=0.02, iv_max=None, grouping='adaptive', grouping_threshold=0.1):
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.grouping = grouping
        self.grouping_threshold = grouping_threshold

    def _users(self, frame, strategy):
        if strategy == 'provided':
            columns = ['sc_user_id']
        elif strategy == 'pair':
            columns = ['Src IP', 'Dst IP']
        elif strategy == 'flow':
            columns = ['Flow ID']
        else:
            raise ValueError(f'Unknown grouping strategy: {strategy}')
        if any(c not in frame for c in columns):
            raise ValueError(f'Grouping {strategy!r} requires {columns}')
        if frame[columns].isna().any().any():
            raise ValueError('User identifier columns contain missing values')
        return frame[columns].astype(str).agg('|'.join, axis=1)

    def _base(self, frame):
        frame = frame.copy()
        frame['_label'] = binary_labels(frame, self.label_column_)
        frame['_user'] = self._users(frame, self.grouping_)
        if self.time_column_ not in frame:
            raise ValueError(f'Missing timestamp column: {self.time_column_}')
        series = frame[self.time_column_]
        parsed = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns]')
        for fmt in ('%d/%m/%Y %I:%M:%S %p', '%d/%m/%Y %H:%M', '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'):
            parsed = parsed.fillna(pd.to_datetime(series, format=fmt, errors='coerce'))
        if parsed.isna().any():
            raise ValueError(f'{parsed.isna().sum()} invalid timestamps; normalize to ISO datetime first')
        frame['_time'] = parsed
        return frame.sort_values(['_user', '_time'], kind='stable').reset_index(drop=True)

    def _engineer(self, frame):
        missing = set(self.numeric_columns_) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing trained numeric features: {sorted(missing)}')
        numeric = frame[self.numeric_columns_].apply(pd.to_numeric, errors='coerce')
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        engineered_names = [f'{c}_{suffix}' for c in self.numeric_columns_ for suffix in ['rolling_mean', 'diff']]
        engineered_names += ['time_diff', 'hour', 'day_of_week', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos']
        if set(self.numeric_columns_) & set(engineered_names):
            raise ValueError('Raw feature names collide with engineered feature names')
        result = {col: numeric[col] for col in self.numeric_columns_}
        users = frame['_user']
        for col in self.numeric_columns_:
            result[f'{col}_rolling_mean'] = numeric[col].groupby(users).transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            result[f'{col}_diff'] = numeric[col].groupby(users).diff().fillna(0)
        result['time_diff'] = frame.groupby('_user')['_time'].diff().dt.total_seconds().fillna(0)
        result['hour'] = frame['_time'].dt.hour
        result['day_of_week'] = frame['_time'].dt.dayofweek
        for unit, period in [('minute', 60), ('hour', 24)]:
            phase = getattr(frame['_time'].dt, unit) * 2 * np.pi / period
            result[f'{unit}_sin'] = np.sin(phase)
            result[f'{unit}_cos'] = np.cos(phase)
        return pd.DataFrame(result, index=frame.index)

    def fit(self, frame):
        self.label_column_ = next((c for c in ['Label', 'flag', 'label', 'Class', 'Anomaly', 'Result'] if c in frame), None)
        self.time_column_ = next((c for c in ['sc_time', 'Timestamp'] if c in frame), None)
        if self.label_column_ is None or self.time_column_ is None:
            raise ValueError('CSV requires a label column and sc_time or Timestamp')
        labels = binary_labels(frame, self.label_column_)
        if labels.nunique() != 2:
            raise ValueError('Training data must contain normal and attack flows')
        self.anomaly_ratio_ = float(labels.mean())
        self.grouping_ = self.grouping
        if self.grouping_ == 'adaptive':
            self.grouping_ = 'provided' if 'sc_user_id' in frame else 'pair'
            if self.grouping_ == 'pair':
                if {'Src IP', 'Dst IP'}.issubset(frame.columns):
                    ratio = labels.groupby(self._users(frame, 'pair')).max().mean()
                    if (ratio < self.grouping_threshold or ratio > 1 - self.grouping_threshold) and 'Flow ID' in frame:
                        self.grouping_ = 'flow'
                else:
                    self.grouping_ = 'flow'
        excluded = IDENTIFIERS | LABEL_COLUMNS | {self.time_column_, 'binary_label'}
        self.numeric_columns_ = [c for c in frame.columns if c not in excluded
                                 and pd.to_numeric(frame[c], errors='coerce').notna().mean() > 0.8]
        base = self._base(frame)
        engineered = self._engineer(base)
        self.iv_ = {c: information_value(engineered[c], base['_label']) for c in engineered}
        self.feature_columns_ = [c for c, iv in self.iv_.items() if iv >= self.iv_min
                                 and (self.iv_max is None or iv <= self.iv_max)]
        if not self.feature_columns_:
            raise ValueError('IV filtering removed every feature; inspect data or lower iv_min')
        self.scaler_ = StandardScaler().fit(engineered[self.feature_columns_])
        return self

    def transform(self, frame):
        if not hasattr(self, 'scaler_'):
            raise ValueError('Fit the preprocessor before transform')
        base = self._base(frame)
        values = self._engineer(base)[self.feature_columns_]
        values = self.scaler_.transform(values).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite standardized features')
        return base, values
