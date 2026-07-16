import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import optimize as opt
import matplotlib.pyplot as plt

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


# ------------------ GPFA utils -----------------------------------
class GPFADataset:
    """
    Create a GPFA-compatible dataset restricted to trials matching one or
    more stimulus conditions (parameter/value pairs), further restricted to
    time bins matching a given running phase.

    Every trial is truncated to exactly `minimum_nr_of_bins` qualifying bins,
    so all trials in the dataset have the same fixed length (self.T).
    Trials with fewer than `minimum_nr_of_bins` matching bins are dropped.
    """

    def __init__(
        self,
        data=None,
        session_id="B",
        stimulus="natural_scenes",
        parameters=["orientation", "temporal_frequency"],
        parameter_values=[0, 1],
        minimum_nr_of_bins=10,
        running_phase_value=None,
    ):
        session = data["sessions"][session_id]
        spikes = np.asarray(session["spikes"], dtype=float)
        t = np.asarray(session["t"], dtype=float)
        stim_table = session["stim_tables"][stimulus].copy()
        running_speed = np.asarray(session["running_speed_filtered"][0], dtype=float)
        running_phase = np.asarray(session["running_speed_phase"], dtype=int)

        if "blank_sweep" not in parameters:
            for parameter, parameter_value in zip(parameters, parameter_values):
                stim_table = stim_table[stim_table[parameter] == parameter_value].reset_index(drop=True)
        elif "blank_sweep" in parameters:
            stim_table = stim_table[stim_table["blank_sweep"] == 1].reset_index(drop=True)

        dt_sec = float(np.median(np.diff(t)))
        dt_ms = dt_sec * 1000.0

        T = int(minimum_nr_of_bins)
        T -= T % 2  # keep even
        if T < 1:
            raise ValueError("minimum_nr_of_bins must be >= 2 (kept even).")

        self.binSize = dt_ms
        self.ydim = spikes.shape[0]
        self.session_id = session_id
        self.stimulus = stimulus
        self.parameters = parameters
        self.parameter_values = parameter_values
        self.running_phase_value = running_phase_value
        self.T = T

        if stim_table.empty:
            self.data, self.trial_durs, self.trialDur, self.numTrials = [], [], 0, 0
            return

        max_start = min(spikes.shape[1], running_speed.shape[0])

        trials = []
        for _, row in stim_table.iterrows():
            start = int(row["start"])
            stop = min(int(row["end"]), max_start)
            if stop <= start:
                continue

            Y = spikes[:, start:stop].clip(min=0.0)
            speed = running_speed[start:stop]
            phase = running_phase[start:stop]

            if running_phase_value is not None:
                mask = phase == int(running_phase_value)
                if mask.sum() < T:
                    continue
                Y = Y[:, mask]
                speed = speed[mask]
                phase = phase[mask]

            if Y.shape[1] < T:
                continue

            trials.append({
                "Y": Y[:, :T].copy(),
                "start_idx": start,
                "end_idx": stop,
                "running_speed": speed[:T].copy(), 
            })

        if not trials:
            raise ValueError(
                f"No trials had at least {T} bins matching running_phase_value={running_phase_value}."
            )

        self.data = trials
        self.trial_durs = [T * dt_ms] * len(trials)
        self.trialDur = T * dt_ms
        self.numTrials = len(trials)

def cov_anal(fit):
    """
    Analytical mean and covariance of the log-normal Cox process 
    approximation to the fitted Poisson-GPFA model (Krumin & Shoham, 2009).
    """
    C = fit.optimParams["C"] # shape (n_neurons, latent_dim)
    d = fit.optimParams["d"] # shape (n_neurons,)
    log_mu = 0.5 * np.sum(C**2, axis=1) + d
    mu = np.exp(log_mu) # shape (n_neurons,)
    cc = C @ C.T

    # covariance of log-normal Cox process approximation
    cov = np.outer(mu, mu) * (np.exp(cc) - 1.0) + np.diag(mu) 
    
    return cov, mu

def var_explained(raw_cov, approx_cov, approx_mu):
    """
    Fraction of variance explained by the shared latent covariance
    of the Poisson-GPFA model.
    """
    total_variance = float(np.trace(raw_cov))
    shared_cov = approx_cov - np.diag(approx_mu)
    shared_variance = float(np.trace(shared_cov))

    return shared_variance / total_variance


# GPFA Plotting functions 
def plot_explained_variance(explained_variances, figure_size=(3.5, 3)):
    """
    Violin plot of explained variance split by running phase.
    """
    phase_colors = {0: "steelblue", 1: "crimson"}
    phase_labels = {0: "Still", 1: "Running"}

    groups = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for (_, _, phase, blank), ve in explained_variances.items():
        groups[(phase, blank)].append(ve)

    fig, ax = plt.subplots(figsize=figure_size)

    violins = ax.violinplot(
        [groups[(0, 0)], groups[(1, 0)]],
        positions=[1, 2],
        widths=0.6,
        showmedians=True,
        showextrema=False,
    )

    for body, color in zip(violins["bodies"], phase_colors.values()):
        body.set_facecolor(color)
        body.set_alpha(0.6)

    for phase, pos in enumerate([1, 2]):
        y = groups[(phase, 0)]
        ax.scatter(
            np.random.normal(pos, 0.04, len(y)),
            y,
            color=phase_colors[phase],
            edgecolor="black",
            s=20,
            alpha=0.7,
        )

        y_blank = groups[(phase, 1)]
        if y_blank:
            ax.scatter(
                np.random.normal(pos, 0.04, len(y_blank)),
                y_blank,
                color="gold",
                edgecolor="black",
                s=40,
                marker="D",
                zorder=4,
                label="Blank sweep" if phase == 0 else None,
            )

    ax.set_xticks([1, 2])
    ax.set_xticklabels([phase_labels[0], phase_labels[1]])
    ax.set_ylabel("FVE")
    ax.legend(fontsize=7)
    fig.tight_layout()

    return fig, ax

def plot_covariances(
    raw_cov,
    approx_cov,
    figure_size=(6, 3),
    cmap="RdBu_r",
):
    """
    Plot raw covariance and GPFA covariance approximation.
    """

    fig, axes = plt.subplots(1, 2, figsize=figure_size, constrained_layout=True)
    vmax = max(np.abs(raw_cov).max(), np.abs(approx_cov).max())
    im = axes[0].imshow(raw_cov, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[0].set_title("Raw")
    axes[0].set_xlabel("Neuron")
    axes[0].set_ylabel("Neuron")

    axes[1].imshow(approx_cov, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[1].set_title("GPFA approximation")
    axes[1].set_xlabel("Neuron")

    fig.colorbar(im, ax=axes, label="Covariance", shrink=0.8, aspect=20)
    return fig, axes

def plot_param_histograms(
    fits,
    trial_keys,
    param_name="C",
    bins=30,
    running_phase=0,
    neuron_idx=None,
    figsize_per_cell=(3, 2),
):
    """
    Plot histograms of a GPFA parameter for selected conditions.
    """

    color = "steelblue" if running_phase == 0 else "crimson"
    
    # get the trial keys for the specified running phase
    keys = [k for k in trial_keys if k[2] == running_phase]
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(figsize_per_cell[0] * len(keys), figsize_per_cell[1]),
        squeeze=False,
    )

    for idx, key in enumerate(keys):
        ax = axes[0, idx]
        orientation, temporal_frequency, _, blank_sweep = key
        _, _, xval = fits[key]
        params = np.asarray(xval.fits[0].optimParams[param_name])
        values = params.ravel()
        ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.5,
            edgecolor="black",
        )
        if neuron_idx is not None and neuron_idx < params.shape[0]:
            ax.axvline(
                params[neuron_idx],
                color="black",
                linestyle="--",
                linewidth=2,
            )
        ax.tick_params(labelsize=6)
        if not blank_sweep:
            ax.set_title(
                f"Ori {orientation:g}°, TF {temporal_frequency:g} Hz", 
                fontsize=8,
            )
        elif blank_sweep:
            ax.set_title(
                f"Blank Sweep", 
                fontsize=8,
            )
    for ax in axes[0]:
        ax.set_xlabel(param_name)
    axes[0, 0].set_ylabel("Frequency")
    fig.tight_layout()
    return fig, axes

def plot_tau_histogram(fits, bins=30, figure_size=(4, 2.5)):
    """
    Plot fitted GP timescales pooled across conditions, split by running
    phase.
    """
    colors = {0: "steelblue", 1: "crimson"}
    labels = {0: "Still", 1: "Running"}
    taus = {0: [], 1: []}
    blank_taus = {0: [], 1: []}

    for (_, _, phase, blank_sweep), (_, _, xval) in fits.items():
        for fit in xval.fits:
            tau = float(fit.optimParams["tau"][0])
            if blank_sweep:
                blank_taus[phase].append(tau)
            else:
                taus[phase].append(tau)

    fig, ax = plt.subplots(figsize=figure_size)
    for phase in (0, 1):
        ax.hist(taus[phase], bins=bins, color=colors[phase], alpha=0.5,
                edgecolor="black", label=labels[phase])

        for tau in blank_taus[phase]:
            ax.axvline(tau, color="gold", linestyle="--", linewidth=2, zorder=3,
                       label="Blank sweep" if tau == blank_taus[phase][0] and phase == 0 else None)

    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax

def plot_gpfa_latent_and_speed_grid(
    fits,
    trial_keys,
    running_phase,
    trial_idx=0,
    figsize_per_cell=(3.5, 2),
):
    """
    Plot running speed and first GPFA latent trajectory for a selected trial
    across given conditions.
    """

    color = "steelblue" if running_phase == 0 else "crimson"
    keys = [k for k in trial_keys if k[2] == running_phase]
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=(figsize_per_cell[0] * len(keys), figsize_per_cell[1]),
        squeeze=False,
    )

    for idx, key in enumerate(keys):
        ax = axes[0, idx]
        orientation, temporal_frequency, _, blank_sweep = key

        _, _, xval = fits[key]
        fit = xval.fits[0]
        data = fit.experiment

        if trial_idx >= data.numTrials:
            ax.set_title("Trial unavailable", fontsize=9)
            ax.axis("off")
            continue

        time = np.arange(data.T) * data.binSize / 1000
        speed = np.asarray(data.data[trial_idx]["running_speed"])
        latent = np.asarray(fit.infRes["post_mean"][trial_idx][0])

        T = min(len(time), len(speed), len(latent))
        ax.plot(time[:T], speed[:T], color=color)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        if not blank_sweep:
            ax.set_title(f"Ori {orientation:g}°, TF {temporal_frequency:g} Hz", fontsize=9)
        else:
            ax.set_title("Blank Sweep", fontsize=9)

        ax2 = ax.twinx()
        ax2.plot(time[:T], latent[:T], color="orange", linestyle="--")
        ax2.set_ylabel("Latent", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

    fig.tight_layout()
    return fig, axes