import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curves(model) -> None:
    """
    Plots RMSE for learn/validation across iterations from CatBoost.
    """
    evals = model.get_evals_result()
    # Keys typically: 'learn' and 'validation' (or 'validation_0' if multiple)
    learn_rmse = evals.get('learn', {}).get('RMSE', None)
    # CatBoost may store validation under 'validation' or 'validation_0'
    valid_rmse = None
    if 'validation' in evals:
        valid_rmse = evals['validation'].get('RMSE', None)
    elif 'validation_0' in evals:
        valid_rmse = evals['validation_0'].get('RMSE', None)

    if learn_rmse is None or valid_rmse is None:
        print("No eval results found to plot. Ensure you passed eval_set in fit().")
        return

    iters = range(1, len(learn_rmse) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(iters, learn_rmse, label="Train RMSE")
    plt.plot(iters, valid_rmse, label="Valid RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("CatBoost Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def summarize_round_splits(
    df: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    season_col: str = "season_x",
    week_col: str = "round",
) -> pd.DataFrame:
    """
    For each season, list which rounds belong to TRAIN / VALID / TEST.
    Prints a compact view and returns a summary DataFrame.
    """
    df2 = df[[season_col, week_col]].copy()
    df2["_subset"] = "UNASSIGNED"
    df2.loc[train_idx.intersection(df2.index), "_subset"] = "TRAIN"
    df2.loc[valid_idx.intersection(df2.index), "_subset"] = "VALID"
    df2.loc[test_idx.intersection(df2.index),  "_subset"] = "TEST"

    # Keep only assigned rows
    df2 = df2[df2["_subset"] != "UNASSIGNED"]

    # Build per-season summary
    rows = []
    for season, g in df2.groupby(season_col, sort=True):
        out = {"season": season}
        for subset in ["TRAIN", "VALID", "TEST"]:
            rounds = sorted(g.loc[g["_subset"] == subset, week_col].unique().tolist())
            if rounds:
                rng = f"{min(rounds)}–{max(rounds)}"
                # If you also want the exact list uncomment next line:
                # out[f"{subset}_rounds_list"] = rounds
                out[f"{subset}_rounds"] = rng
                out[f"{subset}_count"] = len(rounds)
            else:
                out[f"{subset}_rounds"] = "—"
                out[f"{subset}_count"] = 0

        # Simple temporal sanity flags
        def _minmax(sub):
            r = g.loc[g["_subset"] == sub, week_col]
            return (r.min(), r.max()) if len(r) else (None, None)

        tr_min, tr_max = _minmax("TRAIN")
        va_min, va_max = _minmax("VALID")
        te_min, te_max = _minmax("TEST")

        # Check monotonic order: TRAIN ≤ VALID ≤ TEST (where present)
        ok_train_valid = (tr_max is None or va_min is None) or (tr_max <= va_min)
        ok_valid_test  = (va_max is None or te_min is None) or (va_max <= te_min)
        out["order_ok"] = bool(ok_train_valid and ok_valid_test)

        rows.append(out)

    summary = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)

    # Pretty print
    print("\nRound allocation by season (min–max rounds per subset):")
    for _, r in summary.iterrows():
        print(
            f"  {r['season']}: "
            f"TRAIN {r['TRAIN_rounds']} ({r['TRAIN_count']}) | "
            f"VALID {r['VALID_rounds']} ({r['VALID_count']}) | "
            f"TEST {r['TEST_rounds']} ({r['TEST_count']}) "
            f"| order_ok={r['order_ok']}"
        )

    return summary
