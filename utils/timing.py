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
