"""
CoinSense - configuration values used by the backend.

Everything that is a "setting" lives here, so the rules and thresholds
can be changed in one place without touching the logic.
"""

# Value of each coin in cents
COIN_VALUES = {
    "penny": 1,
    "nickel": 5,
    "dime": 10,
    "quarter": 25,
}

# How many coins of each type make one full bank roll
ROLL_SIZES = {
    "penny": 50,
    "nickel": 40,
    "dime": 50,
    "quarter": 40,
}

# Different Roboflow datasets use different label names for the same coin.
# Every label is mapped to one of our 4 standard names.
CLASS_ALIASES = {
    "penny": "penny", "pennies": "penny", "1c": "penny", "1 cent": "penny",
    "one cent": "penny", "cent": "penny", "1": "penny",
    "nickel": "nickel", "nickels": "nickel", "5c": "nickel", "5 cents": "nickel",
    "five cents": "nickel", "5": "nickel",
    "dime": "dime", "dimes": "dime", "10c": "dime", "10 cents": "dime",
    "ten cents": "dime", "10": "dime",
    "quarter": "quarter", "quarters": "quarter", "25c": "quarter",
    "25 cents": "quarter", "twenty five cents": "quarter", "25": "quarter",
}

# Confidence thresholds (proposal section 4)
MIN_CONFIDENCE = 0.50      # below this -> detection is ignored
SURE_CONFIDENCE = 0.70     # 0.50-0.70 -> "unsure" (grey box), above -> "sure"

# Two coin boxes overlapping more than this (IoU) -> "Spread coins out"
OVERLAP_IOU = 0.15

# The total must stay the same for this many seconds before it is shown as final
STABLE_SECONDS = 1.0

# Model / inference settings
DEFAULT_MODEL_PATH = "models/coinsense_best.pt"
IMAGE_SIZE = 640
DEVICE = 0                 # 0 = Jetson GPU, "cpu" = processor
