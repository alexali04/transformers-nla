from datetime import datetime
from pathlib import Path
from random import randint

def print_time_taken(seconds, text="Time taken: "):
    """Print time taken in a human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60

    if hours > 0:
        print(f"{text}{hours}h {minutes}m {seconds:.2f}s")
    elif minutes > 0:
        print(f"{text}{minutes}m {seconds:.2f}s")
    else:
        print(f"{text}{seconds:.2f}s")

def append_timestamp(filepath, add_random_suffix=False):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    if add_random_suffix:
        timestamp += f"_r{randint(1, 1_000)}"
    filepath = Path(filepath)
    aux = f"{filepath.stem}_{timestamp}{filepath.suffix}"
    new_filepath = filepath.with_name(aux)
    return str(new_filepath)
