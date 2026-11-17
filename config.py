"""Pre-registered constants for the pairs trading walk-forward study.

Every parameter lives here. Nothing is changed after Phase 0.
"""

START_DATE = "2015-01-01"
END_DATE_EXCLUSIVE = "2026-11-01"      # last trading day used is 2026-10-30
FORMATION_DAYS = 504                   # about 2 trading years
TRADING_DAYS = 126                     # about 6 trading months
STEP_DAYS = 126                        # windows roll forward by one trading window
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 4.0
COINT_ALPHA = 0.05                     # raw Engle-Granger p-value threshold
BH_FDR = 0.10                          # Benjamini-Hochberg false discovery rate
MIN_HALF_LIFE = 1.0                    # trading days
MAX_HALF_LIFE = 63.0                   # trading days (half of a trading window)
COST_BPS = 5.0                         # illustrative one-way cost per unit of gross notional traded
BORROW_ANNUAL = 0.005                  # illustrative 0.50% annual borrow fee on the short leg
TRADING_DAYS_PER_YEAR = 252
BOOTSTRAP_REPS = 10000
BLOCK_LENGTH = 21
RANDOM_SEED = 42
COST_GRID_BPS = [0.0, 2.0, 5.0, 10.0]  # sensitivity only
ENTRY_GRID = [1.5, 2.0, 2.5]           # sensitivity only
BENCHMARK = "SPY"

# Data files and the latest acceptable first-valid date for any ticker.
PRICES_PATH = "data/prices.csv"
METADATA_PATH = "data/metadata.json"
LATEST_FIRST_VALID_DATE = "2015-01-10"
MAX_PAIRS_DROPPED = 5
MAX_CALENDAR_LOSS_DAYS = 10

# Universe: (Y, X, sector label). The first ticker is the dependent variable.
PAIRS = [
    ("V", "MA", "payments networks"),
    ("KO", "PEP", "beverages"),
    ("XOM", "CVX", "integrated oil"),
    ("COP", "EOG", "exploration and production"),
    ("JPM", "BAC", "money-center banks"),
    ("GS", "MS", "investment banks"),
    ("USB", "PNC", "regional banks"),
    ("AXP", "COF", "consumer credit"),
    ("HD", "LOW", "home improvement retail"),
    ("WMT", "TGT", "big-box retail"),
    ("PG", "CL", "household products"),
    ("MCD", "YUM", "quick-service restaurants"),
    ("MO", "PM", "tobacco"),
    ("UPS", "FDX", "parcel delivery"),
    ("UNP", "CSX", "railroads"),
    ("DAL", "UAL", "airlines"),
    ("GM", "F", "automakers"),
    ("CAT", "DE", "heavy machinery"),
    ("LMT", "NOC", "defense"),
    ("MRK", "PFE", "pharmaceuticals"),
    ("ABT", "MDT", "medical devices"),
    ("DUK", "SO", "regulated utilities"),
    ("T", "VZ", "telecom"),
    ("SPGI", "MCO", "credit rating agencies"),
    ("CME", "ICE", "exchanges"),
    ("AMAT", "LRCX", "semiconductor equipment"),
    ("AMT", "CCI", "cell towers"),
    ("PSA", "EXR", "self-storage REITs"),
    ("SHW", "PPG", "paints and coatings"),
    ("ADP", "PAYX", "payroll services"),
]
