# ysignals

## Install

```
pip install -U git+http://github.com/jrdi/ysignals.git@master#egg=ysignals
```

## Usage

```python
from pathlib import Path

from ysignals.data import yahoo
from ysignals.features import RSI

db_dir = Path('db')

yahoo.download_data(db_dir)

features_generators = [
    RSI(num_days=5, interval=10)
]

train, test, live, feature_names = yahoo.get_data(db_dir, features_generators=features_generators)
```
