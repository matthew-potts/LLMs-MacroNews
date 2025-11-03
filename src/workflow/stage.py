from enum import IntEnum

class Stage(IntEnum):
    get_data = 0
    get_market_data = 1
    generate_ratings = 2
    regress = 3