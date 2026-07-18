import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, linregress, pearsonr, spearmanr

# ----------------------------- data loading and saving utils -----------------------------
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

#----------------------------- Running speed - neural activity analysis utils #-----------------------------

# Select only phases without stimulus to avoid mixing up stimulus and running speed influence
def spontaneous_mask(session):
    """Boolean mask selecting spontaneous imaging frames."""
    n_frames = session["spikes"].shape[1]
    mask = np.zeros(n_frames, dtype=bool)
    spontaneous = session["stim_epoch_table"]
    spontaneous = spontaneous[spontaneous["stimulus"] == "spontaneous"]
    for _, row in spontaneous.iterrows():
        mask[int(row.start):int(row.end)] = True
    return mask

# create (equal-sized) temporal bins for speed and spikes
def temporal_bin(speed, spikes, fps=30, seconds=1):
    """
    Average running speed and dF/F into fixed-duration bins.

    Parameters
    ----------
    speed : (T,)
    spikes : (N,T)
    """
    bin_size = int(round(fps * seconds))
    n_bins = speed.size // bin_size
    speed = speed[:n_bins * bin_size]
    spikes = spikes[:, :n_bins * bin_size]
    speed_binned = speed.reshape(n_bins, bin_size).mean(axis=1)
    spikes_binned = (
        spikes
        .reshape(spikes.shape[0], n_bins, bin_size)
        .mean(axis=2)
    )
    return speed_binned, spikes_binned

# Analyze spike activity modulation via running speed by correlating these two with pearson and
# spearman correlation and perform a linear regression to obtain explained variance. Returns a
# dataframe with results of correlation and linear regression. Requires binned speed and spike
# data as input
def analyze_running_modulation(speed, spikes):
    results = []
    for neuron in range(spikes.shape[0]):
        y = spikes[neuron]
        valid = np.isfinite(speed) & np.isfinite(y)
        x = speed[valid]
        y = y[valid]
        if len(x) < 20:
            continue
        pearson_r, _ = pearsonr(x, y)
        spearman_rho, _ = spearmanr(x, y)
        fit = linregress(x, y)
        results.append({
            "neuron": neuron,
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r2": fit.rvalue ** 2,
        })
    return pd.DataFrame(results)

# Applies the above methods to apply correlational and linear regression analysis on the binned
# data of spontaneous phases.
def perform_running_modulation_analysis(data):
    results = {}
    binned_data = {}
    for name, session in data["sessions"].items():
        mask = spontaneous_mask(session)
        speed = session["running_speed_filtered"][0, mask]
        spikes = session["spikes"][:, mask]
        binned_data[name] = temporal_bin(speed, spikes, seconds=0.1)
        results[name] = analyze_running_modulation(binned_data[name][0], binned_data[name][1])
    return results, binned_data

#----------------------------- Running speed - neural activity analysis plotting functions -----------------------------

# shows the linear regression line and correlation between running speed and neural activity as well as
# the mean spike rate for each speed bin for one neuron
def plot_running_modulation_neuron(binned_data, all_results, session, neuron):
    speed, spikes = binned_data[session]
    row = all_results[session].query("neuron == @neuron").iloc[0]
    y = spikes[neuron]
    m = np.isfinite(speed) & np.isfinite(y)
    x, y = speed[m], y[m]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=5, alpha=0.15, color="gray", label="Samples")
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, row.intercept + row.slope * xx,
            color="crimson", lw=2, label="Linear fit")
    mean, edges, _ = binned_statistic(x, y, statistic="mean", bins=15)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, mean, "-o",
            color="royalblue", ms=4, lw=2, label="Binned mean")
    ax.set_xlabel("Running speed")
    ax.set_ylabel("Mean spikes")
    ax.set_title(f"{session} N{neuron}  r={row.pearson_r:.2f}")
    ax.legend(frameon=False)

# shows the linear regression line and correlation between running speed and neural activity as well as
# the mean spike rate for each speed bin for all neurons
def plot_running_modulation_all(binned_data, all_results, session, cols=6):
    speed, spikes = binned_data[session][0], binned_data[session][1]
    df = all_results[session]
    n = len(df)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, df.itertuples()):
        y = spikes[row.neuron]
        m = np.isfinite(speed) & np.isfinite(y)
        x, y = speed[m], y[m]

        mean, edges, _ = binned_statistic(x, y, statistic="mean", bins=15)
        centers = (edges[:-1] + edges[1:]) / 2

        ax.plot(centers, mean, "-o", c="royalblue", ms=2)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, row.intercept + row.slope * xx, c="crimson", lw=1)
        ax.set_title(f"N{row.neuron} r={row.pearson_r:.2f}", fontsize=8)
    for ax in axes[n:]:
        fig.delaxes(ax)
    fig.supxlabel("Running speed")
    fig.supylabel("Mean spikes")

# plots the distribution of correlations as a histogram
def plot_correlation_histograms(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(3 * len(all_results), 2.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (session, df) in zip(axes, all_results.items()):
        ax.hist(df.pearson_r, bins=15, color="steelblue", edgecolor="white")
        ax.axvline(0, c="k", ls="--", lw=1)
        ax.set_title(session)
        ax.set_xlabel("Pearson r")

    axes[0].set_ylabel("Neurons")
    plt.tight_layout(pad=.3)

# plots the neurons correlations sorted by strength (alternative to histogram)
def plot_running_correlations(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(3 * len(all_results), 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (session, df) in zip(axes, all_results.items()):
        ax.plot(np.sort(df.pearson_r), "o-", ms=3)
        ax.axhline(0, c="k", ls="--", lw=1)
        ax.set_title(session)
        ax.set_xlabel("Neuron")
    axes[0].set_ylabel("Pearson r")

# plots the neurons correlation consistence across sessions (note: data for session c is very sparse)
def plot_running_correlation_sessions(all_results):
    sessions = list(all_results)
    fig, ax = plt.subplots(figsize=(6, 3))
    for n in range(len(all_results[sessions[0]])):
        ax.plot(sessions, [all_results[s].iloc[n].pearson_r for s in sessions], c="gray", alpha=.25)
    for s in sessions:
        ax.scatter([s] * len(all_results[s]), all_results[s].pearson_r, label=s)
    ax.axhline(0, c="k", ls="--", lw=1)
    ax.set_ylabel("Pearson r")
    ax.legend()
    plt.tight_layout(pad=.3)


#----------------------------- Running onset - neural activity analysis utils -----------------------------

# detects running onsets
def detect_running_onsets(session, pre=30, post=30):
    phase = session["running_speed_phase"]
    spont = spontaneous_mask(session)

    onsets = []

    for i in range(0, len(phase) - post):
        if phase[i - 1] != 0 or phase[i] != 1:
            continue
        if not spont[i - pre:i + post].all():
            continue
        onsets.append(i)

    return np.asarray(onsets)

# extracts onset windows
def extract_onset_windows(spikes, onsets, pre=30, post=30):
    windows = np.stack([
        spikes[:, i - pre:i + post]
        for i in onsets
    ])
    return windows

# takes onset windows as input, computes the diff as mean after - mean before
# and tests the onset modulation significance using a sign-flip permutation test
def running_onset_analysis(windows, pre=30, n_perm=10000):
    baseline = windows[:, :, :pre].mean(axis=2)
    response = windows[:, :, pre:].mean(axis=2)
    results = []
    for neuron in range(windows.shape[1]):
        diff = response[:, neuron] - baseline[:, neuron]
        observed = diff.mean()
        null = np.empty(n_perm)
        for i in range(n_perm):
            signs = np.random.choice([-1, 1], size=len(diff))
            null[i] = np.mean(diff * signs)
        p = (np.sum(null >= observed) + 1) / (n_perm + 1)
        results.append({
            "neuron": neuron,
            "baseline": baseline[:, neuron].mean(),
            "response": response[:, neuron].mean(),
            "delta": observed,
            "p": p,
            "n_events": len(diff),
        })
    return pd.DataFrame(results)

# applies the methods above to test onset modulation significance
def perform_onset_modulation_analysis(data):
    onset_results = {}
    onset_data = {}
    for name, session in data["sessions"].items():
        onsets = detect_running_onsets(session)
        windows = extract_onset_windows(session["spikes"],onsets)
        onset_data[name] = {"onsets": onsets,"windows": windows,}
        onset_results[name] = running_onset_analysis(windows)
    return onset_results, onset_data

# uses a score to rank consistency of onset responses
def onset_consistency(onset_results):
    out = pd.DataFrame(index=onset_results[next(iter(onset_results))].neuron)
    for session, df in onset_results.items():
        out[session] = df.set_index("neuron").delta
    out["mean_delta"] = out.mean(axis=1)
    return out.sort_values("mean_delta", ascending=False)

#----------------------------- Running onset - neural activity analysis plotting functions ---------------------------

# plots the distribution of onset deltas (mean after minus mean before) over the neurons
def plot_onset_delta(onset_results):
    fig, axes = plt.subplots(1, len(onset_results), figsize=(3 * len(onset_results), 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, df) in zip(axes, onset_results.items()):
        ax.hist(df.delta, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(0, c="k", ls="--")
        ax.set_title(name)
        ax.set_xlabel("Running onset Δ firing")
    axes[0].set_ylabel("Neurons")

# plots the mean onset response and standard deviation of one neuron in a particular session
def plot_onset_neuron(onset_data, session, neuron):
    w = onset_data[session]["windows"][:, neuron]
    mean = w.mean(axis=0)
    sem = w.std(axis=0) / np.sqrt(len(w))
    t = np.arange(len(mean)) - 30
    plt.figure(figsize=(5, 3))
    plt.plot(t, mean, c="k")
    plt.fill_between(t, mean - sem, mean + sem, alpha=.3)
    plt.axvline(0, c="r", ls="--")
    plt.xlabel("Frames from running onset")
    plt.ylabel("Spike rate")

# plots the mean onset response and standard deviation over all neurons in a particular session
def plot_population_onset(onset_data, session, pre=30):
    w = onset_data[session]["windows"]
    trace = w.mean(axis=0)
    mean = trace.mean(axis=0)
    sem = trace.std(axis=0) / np.sqrt(trace.shape[0])
    t = np.arange(len(mean)) - pre
    plt.figure(figsize=(5, 3))
    plt.plot(t, mean)
    plt.fill_between(t, mean - sem, mean + sem, alpha=.3)
    plt.axvline(0, c="r", ls="--")
    plt.xlabel("Frames from onset")
    plt.ylabel("Population firing")

# Alternative heatmap visualization of population tuning onsets with neurons stacked
def plot_onset_population(onset_data, session, pre=30):
    w = onset_data[session]["windows"]  # events x neurons x time
    activity = w.mean(axis=0)
    order = np.argsort(activity[:, pre:].mean(axis=1))[::-1]
    activity = activity[order]
    t = np.arange(activity.shape[1]) - pre
    fig, ax = plt.subplots(
        2, 1,
        figsize=(7, 5),
        gridspec_kw={"height_ratios":[3,1]},
        sharex=True
    )
    # neurons x time heatmap
    im = ax[0].imshow(
        activity,
        aspect="auto",
        cmap="viridis",
        extent=[t[0], t[-1], 0, activity.shape[0]]
    )
    ax[0].axvline(0, c="r", ls="--")
    ax[0].set_ylabel("Neuron (sorted by strength)")
    # population response
    mean = activity.mean(axis=0)
    sem = activity.std(axis=0)/np.sqrt(activity.shape[0])
    ax[1].plot(t, mean, c="k")
    ax[1].fill_between(t, mean-sem, mean+sem, alpha=.3)
    ax[1].axvline(0, c="r", ls="--")
    ax[1].set_xlabel("Time from running onset (frames)")
    ax[1].set_ylabel("Spike rate")

# ----------------------------- Tuning Analysis utils -----------------------------

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
        T -= T % 2  # keep even, GPFA somehow requires this

        self.binSize = dt_ms
        self.ydim = spikes.shape[0]
        self.session_id = session_id
        self.stimulus = stimulus
        self.parameters = parameters
        self.parameter_values = parameter_values
        self.running_phase_value = running_phase_value
        self.T = T
        
        # empty dataset
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
            
            # Constrain spikes to running phase bins
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

def plot_tau_histogram(
    fits, 
    bins=30, 
    figure_size=(4, 2.5)
):
    """
    Plot fitted GP timescales pooled across conditions, 
    split by running phase.
    """
    colors = {0: "steelblue", 1: "crimson"}
    labels = {0: "Still", 1: "Running"}
    taus = {0: [], 1: []}

    for (_, _, phase, blank_sweep), (_, _, xval) in fits.items():
        if blank_sweep:
            continue
        for fit in xval.fits:
            tau = float(fit.optimParams["tau"][0])
            taus[phase].append(tau)

    fig, ax = plt.subplots(figsize=figure_size)
    for phase in (0, 1):
        ax.hist(taus[phase], bins=bins, color=colors[phase], alpha=0.5,
                edgecolor="black", label=labels[phase])

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
    Plot running speed and GPFA latent trajectory for a selected trial
    across given conditions (keys).
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