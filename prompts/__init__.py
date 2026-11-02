"""The single source of prompt templates; loaded without whitespace changes."""
from importlib.resources import files

def load_prompt(name: str) -> str:
    return files(__package__).joinpath(name).read_text(encoding="utf-8")
