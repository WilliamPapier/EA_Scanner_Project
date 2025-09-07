# EA_Scanner_Project
Fully automated EA + Python Scanner for High-probability trading setups

## Project Structure

This project consists of several Python components that work together to scan for trading opportunities and provide data to a MetaTrader 5 Expert Advisor (EA):

### Python Files
- `scanner.py` - Main scanner that analyzes market data and identifies trading setups
- `infer_and_write_params.py` - Post-processes scanner output and adjusts risk parameters
- `news_updater.py` - Manages news-based trading restrictions
- `scheduler.py` - Orchestrates the execution of all components on a schedule

### Dependencies
Install required packages:
```bash
pip install MetaTrader5 pandas numpy requests schedule
```

### Usage
1. Ensure MetaTrader 5 is installed and running
2. Run individual components:
   ```bash
   python scanner.py
   python infer_and_write_params.py
   python news_updater.py
   ```
3. Or run the scheduler for automated execution:
   ```bash
   python scheduler.py
   ```

### Output Files
- `model_params.csv` - Trading setups for the EA to read
- `news_block.csv` - News-based trading restrictions
