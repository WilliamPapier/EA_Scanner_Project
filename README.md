# EA_Scanner_Project
Fully automated EA + Python Scanner for High-probability trading setups

## Project Structure

```
EA_Scanner_Project/
├── src/
│   ├── __init__.py
│   ├── scanner.py          # Main scanner for detecting trading setups
│   ├── infer_and_write_params.py  # Parameter inference and adjustment
│   ├── news_updater.py     # News event handling
│   └── scheduler.py        # Main scheduling script
├── samples/
│   └── model_params_sample.csv  # Sample CSV output format
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Configure MetaTrader 5 connection
3. Run the scheduler: `python src/scheduler.py`

## Components

- **scanner.py**: Detects high-probability trading setups using technical analysis
- **infer_and_write_params.py**: Processes and adjusts scanner output
- **news_updater.py**: Manages news event filtering
- **scheduler.py**: Orchestrates the entire scanning process
