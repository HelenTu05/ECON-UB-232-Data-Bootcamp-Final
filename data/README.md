# Data

This project uses the **2015 Flight Delays and Cancellations** dataset published by
the U.S. Department of Transportation's Bureau of Transportation Statistics.

| File | Size | Tracked in git? |
| --- | --- | --- |
| `flights.csv` | ~592 MB, 5,819,079 rows | No — download it |
| `airlines.csv` | 374 B, 14 rows | Yes |
| `airports.csv` | 24 KB, 322 rows | Yes |

`flights.csv` is too large for git and is excluded via `.gitignore`. The two
lookup tables are small and are committed so the notebooks' join logic can be
read without downloading anything.

## Download

From [kaggle.com/datasets/usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays),
or with the Kaggle CLI:

```bash
kaggle datasets download -d usdot/flight-delays -p data/ --unzip
```

Place `flights.csv` in this directory so the layout is:

```
data
├── flights.csv     # downloaded
├── airlines.csv
└── airports.csv
```

The notebooks in `notebooks/` read these as `../data/flights.csv`, so run them
from the `notebooks/` directory (which is what Jupyter does by default when you
open a notebook from there).
