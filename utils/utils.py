import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import optimize as opt

# load the data
def load_data(path="../data"):
    raw = dict(np.load(Path(path) / "visual_coding_data.npz", allow_pickle=True))
    data = {
        "matched_cell_ids": raw["matched_cell_ids"],
        "templates": {},
        "sessions": {},
    }

    def session(L):
        return data["sessions"].setdefault(L, {"stim_tables": {}})

    for key, val in raw.items():
        if key == "matched_cell_ids":
            continue
        parts = key.split("__")
        if parts[0] == "tmpl":
            data["templates"][parts[1]] = val
            continue
        L = parts[0]
        s = session(L)
        if parts[1] == "stim" and parts[3] == "values":
            stim = parts[2]
            cols = list(raw[f"{L}__stim__{stim}__cols"])
            s["stim_tables"][stim] = pd.DataFrame(val, columns=cols)
        elif parts[1] == "epoch" and parts[2] == "values":
            cols = list(raw[f"{L}__epoch__cols"])
            s["stim_epoch_table"] = pd.DataFrame(val, columns=cols)
        elif parts[1] in ("session_type",):
            s["session_type"] = val.item() if hasattr(val, "item") else val
        elif parts[1] in ("t", "dff", "roi_masks", "max_projection", "running_speed"):
            s[parts[1]] = val
    return data


def print_info(data):
    print(f"matched cells: {len(data['matched_cell_ids'])}")
    print(f"templates: {list(data['templates'])}")
    for L, s in sorted(data["sessions"].items()):
        print(f"\nsession {L} ({s.get('session_type')})")
        print(
            f"  t: {s['t'].shape}, dff: {s['dff'].shape}, roi_masks: {s['roi_masks'].shape}"
        )
        for name, df in s["stim_tables"].items():
            print(f"  stim '{name}': {df.shape} cols={list(df.columns)}")

def save_data(data, filename):
    with open("../data/" + filename, "wb") as f:
        pickle.dump(data, f)

def load_saved_data(filename):
    with open("../data/" + filename, "rb") as f:
        return pickle.load(f)
    
def vonMises(θ: np.ndarray, α: float, κ: float, ν: float, ϕ: float) -> np.ndarray:
    """Evaluate the parametric von Mises tuning curve with parameters p at locations theta.
    """
    # transform into degree
    θ = np.deg2rad(θ)
    return np.exp(α + κ * (np.cos(2 * (θ - ϕ)) - 1) + ν * (np.cos(θ - ϕ) - 1))

def tuningCurve(
    counts: np.ndarray, 
    stim: np.ndarray
) -> np.ndarray:
    """Fit a von Mises tuning curve to the spike counts of a neuron in response to different stimulus orientations.
    """

    # Average spike count per unique stimulus orientation.
    grouped = (
        pd.DataFrame({"Stim": stim, "Counts": counts})
        .groupby("Stim", sort=True)["Counts"]
        .mean()
    )
    stim_values = grouped.index.to_numpy(dtype=float)
    mean_counts = grouped.to_numpy(dtype=float)
    x0 = (1, 1, 1, 1)

    def residuals(p, theta, counts_hat):
        # unpack free parameters
        α, κ, ν, ϕ = p
        # compute count misestimation as residual (loglikelihood estimation)
        return vonMises(theta, α, κ, ν, ϕ) - counts_hat
    
    # optimize after p
    res_1 = opt.least_squares(fun=residuals, x0=x0, args=(stim_values, mean_counts))

    return res_1.x

def testTuning(
    counts: np.ndarray,
    dirs: np.ndarray,
    psi: int = 1,
    niters: int = 1000,
    random_seed: int = 2046,
):
    """Permutation test for tuning significance based on the magnitude of a
    given Fourier component of the direction/orientation tuning curve.
    """
    df = pd.DataFrame({"Dirs": dirs, "Counts": counts}).dropna()

    grouped = [g.to_numpy(dtype=float) for _, g in df.groupby("Dirs")["Counts"]]
    dirs_unique = np.array(sorted(df["Dirs"].unique()), dtype=float)

    # Mean per direction; works even with unequal trial counts per direction.
    mean_counts = np.array([g.mean() for g in grouped], dtype=float)
    # Compute fourier component
    nu = np.exp(psi * 1j * np.deg2rad(dirs_unique))
    q_abs = float(np.abs(np.dot(mean_counts, nu)))

    pooled_counts = np.concatenate(grouped)
    group_sizes = np.array([len(g) for g in grouped], dtype=int)
    split_idx = np.cumsum(group_sizes)[:-1]

    rng = np.random.default_rng(random_seed)
    
    # Compute null distribution of q 
    qs_abs_shuffled = np.empty(niters, dtype=float)
    for i in range(niters):
        shuffled = rng.permutation(pooled_counts)
        shuffled_groups = np.split(shuffled, split_idx)
        shuffled_means = np.array([g.mean() for g in shuffled_groups], dtype=float)
        qs_abs_shuffled[i] = np.abs(np.dot(shuffled_means, nu))

    p_value = float(np.mean(qs_abs_shuffled >= q_abs))

    return p_value, q_abs, qs_abs_shuffled

def cov_anal(fit):
    """Analytical mean and covariance of the log-normal Cox process approximation to the fitted Poisson-GPFA model (Krumin & Shoham, 2009).
    """
    C = fit.optimParams["C"] # shape (n_neurons, latent_dim)
    d = fit.optimParams["d"] # shape (n_neurons,)
    
    # mu of log-normal Cox process approximation
    mu = np.exp(0.5 * np.sum(C**2, axis=1) + d) # shape (n_neurons,)
    
    # covariance of log-normal Cox process approximation
    cov = np.outer(mu, mu) * (np.exp(C @ C.T) -  1) + np.diag(mu) # shape (n_neurons, n_neurons)
    
    return cov, mu