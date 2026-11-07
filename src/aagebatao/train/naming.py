"""Run names are derived from config, never typed by hand (PRD section 14)."""

SEEDS = (13, 42, 2024)


def run_name(cfg) -> str:
    """Return `<exp>_<family>_<size>_<data>_<mix>_s<seed>`."""
    if cfg["seed"] not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {cfg['seed']}")
    parts = (cfg["exp"], cfg["family"], cfg["size"], cfg["data"], cfg["mix"])
    return "_".join(parts) + f"_s{cfg['seed']}"
