# PASSKEY: rushit2712
import importlib

# Load 243A strategy dynamically since 243A starts with a digit
strategy_243a_module = importlib.import_module("243A.strategy_243a")
Strategy243A = strategy_243a_module.Strategy243A

from longpine.strategy_longpine import StrategyLongpineZFTF

STRATEGIES = {
    "243A": Strategy243A(),
    "LONGPINE_ZFTF": StrategyLongpineZFTF()
}

def get_strategy(name: str):
    return STRATEGIES.get(name)
