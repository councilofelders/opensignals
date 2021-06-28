# Open Signals

## Install

```
pip install -U opensignals
```

## Usage

```python
from pathlib import Path

from opensignals.data import yahoo
from opensignals.features import RSI, SMA

db_dir = Path('db')

yahoo.download_data(db_dir)

features_generators = [
    RSI(num_days=5, interval=14, variable='adj_close'),
    RSI(num_days=5, interval=21, variable='adj_close'),
    SMA(num_days=5, interval=14, variable='adj_close'),
    SMA(num_days=5, interval=21, variable='adj_close'),
]

train, test, live, feature_names = yahoo.get_data(db_dir, features_generators=features_generators)
```
