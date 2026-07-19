import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, linregress, pearsonr, spearmanr
from oasis.functions import deconvolve
from scipy import signal
import seaborn as sns

STIMULUS_SESSION_MAP = {
    "drifting_gratings": "A",
    "static_gratings": "B",
    "natural_scenes": "B",
    "locally_sparse_noise": "C",
}

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

# ----------------------------- data preprocessing (oasis and speed) -----------------------------

# spike inference for one session
def infer_spikes_oasis(session, fs):
    dff = session["dff"]
    n_cells, n_time = dff.shape
    spikes = np.zeros_like(dff)
    calcium = np.zeros_like(dff)
    for i in range(n_cells):
        c, s, b, g, lam = deconvolve(dff[i], penalty = 1) # tau_d=0.1, framerate=fs)
        spikes[i] = s
        calcium[i] = c
    return spikes, calcium

# spike inference for all sessions
def spike_inference(data):
    for name, session in data["sessions"].items():
        fs = 1.0 / np.median(np.diff(session["t"]))
        spikes, calcium = infer_spikes_oasis(session, fs)
        session["spikes"] = spikes
        session["calcium_fit"] = calcium
    return data

# plot spike inference result for one session
def plot_oasis_result(data, session):
    plt.figure(figsize=(15, 4))
    plt.plot(data["sessions"][session]["t"], data["sessions"][session]["dff"][0], label="ΔF/F")
    plt.plot(data["sessions"][session]["t"], data["sessions"][session]["spikes"][0], label="OASIS output")
    plt.legend()

# Running speed filtering and running phase extraction
def preprocess_running_speed(
    data: dict,
    high: float = 3.0,
    order: int = 3,
    clip_speed: bool = True,
    running_threshold: float = 2.0,
    ) -> dict:

    for name, session in data["sessions"].items():
        speed = np.asarray(session["running_speed"][0], dtype=float)
        t = np.asarray(session["running_speed"][1], dtype=float)
        dt = np.median(np.diff(t))

        fs = 1.0 / dt
        nyquist = 0.5 * fs
        if not (0 < high < nyquist):
            raise ValueError(f"{name}: high must be between 0 and {nyquist:.3f} Hz, got {high}")

        valid = np.isfinite(speed)

        # We have nans at beginning and end of the speed trace,
        # so we need to interpolate to fill them in before filtering.
        speed_interp = np.interp(
            np.arange(speed.size),
            np.flatnonzero(valid),
            speed[valid],
        )

        b, a = signal.butter(order, high, btype="low", fs=fs)
        speed_filtered = signal.filtfilt(b, a, speed_interp)

        # clip running speed at 0
        if clip_speed:
            speed_filtered = np.clip(speed_filtered, 0, None)

        # Separate filtered speed into two kinematic phases: still (0) vs running (1).
        phases = np.full(speed_filtered.shape, -1, dtype=int)
        phases[speed_filtered <= running_threshold] = 0
        phases[speed_filtered > running_threshold] = 1

        # Restore missing values in filtered signal while keeping phase=-1 for invalid points.
        speed_filtered[~valid] = np.nan
        phases[~valid] = -1
        session["running_speed_filtered"] = np.vstack([speed_filtered, t])
        session["running_speed_phase"] = phases

    return data

# plot filtered/unfiltered running speed comparison
def plot_running_speed(data, session="B", t=(0, 100)):
    """Plot raw and filtered running speed with phase background blocks."""
    session_data = data["sessions"][session]

    speed_raw, time_raw = (np.asarray(a, dtype=float) for a in session_data["running_speed"])
    speed_filt, time_filt = (np.asarray(a, dtype=float) for a in session_data["running_speed_filtered"])

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(time_raw, speed_raw, color="0.2", lw=0.8)
    axs[0].set(xlim=t, title="Original Running Speed", ylabel="Speed (cm/s)")

    axs[1].plot(time_filt, speed_filt, color="0.2", lw=0.8)
    axs[1].set(xlim=t, title="Filtered Running Speed", ylabel="Speed (cm/s)", xlabel="Time (s)")

    # Shade contiguous phase blocks on both subplots.

    phase = np.asarray(session_data["running_speed_phase"], dtype=int)

    phase_names = {-1: "Invalid", 0: "Still", 1: "Running"}
    phase_colors = {-1: "0.8", 0: "tab:blue", 1: "tab:red"}

    change_points = np.flatnonzero(np.diff(phase)) + 1  # get the indices where the phase changes
    block_starts = np.concatenate(([0], change_points))
    block_ends = np.concatenate((change_points, [phase.size]))

    seen_labels = set()
    for start, end in zip(block_starts, block_ends):
        p = phase[start]
        label = phase_names.get(p, f"Phase {p}") if p not in seen_labels else None
        seen_labels.add(p)
        for ax in axs:
            ax.axvspan(
                time_filt[start], time_filt[end - 1],
                color=phase_colors.get(p, "0.5"), alpha=0.2, label=label,
            )

    axs[1].legend(loc="upper right")
    fig.tight_layout()
    return fig, axs

# ----------------------------Exploratory analysis utils --------------------------------
def plot_spike_raster(data, session, ax=None, threshold=0.05, xlim=None):
    """
    Plot a raster plot of inferred spike activity across all neurons in a session.
    Each row is a neuron; each vertical tick marks a time point where spike
    activity exceeds the threshold (since OASIS output is continuous, not binary).

    Parameters
    ----------
    threshold : float
        Minimum spike value to count as a "spike event" for raster display.
        Adjust based on your data's typical spike amplitude range.
    xlim : tuple, optional
        (start_time, end_time) in seconds to zoom into a specific window.
    """
    s = data["sessions"][session]
    spikes = s["spikes"]  # shape (n_cells, n_timepoints)
    t = s["t"]
    n_cells = spikes.shape[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    for cell_idx in range(n_cells):
        spike_times = t[spikes[cell_idx] > threshold]
        ax.vlines(spike_times, cell_idx + 0.5, cell_idx + 1.5, color="black", linewidth=0.5)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron #")
    ax.set_ylim(0.5, n_cells + 0.5)
    ax.set_title(f"session {session}: spike raster ({n_cells} neurons)")

    if xlim is not None:
        ax.set_xlim(xlim)

    return ax

def plot_spike_raster_with_running_all_sessions(data, sessions=["A", "B", "C"],
                                                    threshold=0.02, xlim=None,
                                                    use_filtered_speed=True):
    """
    Plot spike raster + running speed for multiple sessions, stacked vertically,
    for easy side-by-side comparison.
    """
    n_sessions = len(sessions)
    fig, axes = plt.subplots(
        n_sessions * 2, 1, figsize=(12, 4 * n_sessions),
        gridspec_kw={"height_ratios": [3, 1] * n_sessions}
    )

    for i, session in enumerate(sessions):
        ax_raster = axes[i * 2]
        ax_speed = axes[i * 2 + 1]

        s = data["sessions"][session]
        spikes = s["spikes"]
        t = s["t"]
        n_cells = spikes.shape[0]

        speed_key = "running_speed_filtered" if use_filtered_speed else "running_speed"
        speed = s[speed_key][0]
        t_speed = s[speed_key][1]

        # Raster
        for cell_idx in range(n_cells):
            spike_times = t[spikes[cell_idx] > threshold]
            ax_raster.vlines(spike_times, cell_idx + 0.5, cell_idx + 1.5,
                              color="black", linewidth=0.5)
        ax_raster.set_ylabel("neuron #")
        ax_raster.set_ylim(0.5, n_cells + 0.5)
        ax_raster.set_title(f"session {session}: spike raster ({n_cells} neurons)")

        # Running speed
        ax_speed.plot(t_speed, speed, lw=0.8, color="darkorange")
        ax_speed.set_ylabel("speed\n(cm/s)")

        if xlim is not None:
            ax_raster.set_xlim(xlim)
            ax_speed.set_xlim(xlim)

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    return fig, axes


def add_stimulus_shading(ax, data, session, t, xlim=None):
    """
    Shade the time windows during which any stimulus was being presented,
    using each stimulus's own stim_table (start/end indices into t).
    """
    s = data["sessions"][session]
    session_stims = [name for name, sess in STIMULUS_SESSION_MAP.items() if sess == session]

    for stim_name in session_stims:
        stim_table = s["stim_tables"][stim_name]
        starts = stim_table["start"].astype(int).values
        ends = stim_table["end"].astype(int).values

        for start_idx, end_idx in zip(starts, ends):
            start_t, end_t = t[start_idx], t[end_idx]
            if xlim is not None and (end_t < xlim[0] or start_t > xlim[1]):
                continue
            ax.axvspan(start_t, end_t, color="steelblue", alpha=0.15, zorder=0)


def plot_spike_raster_with_running(data, session, threshold=0.05, xlim=None,
                                   use_filtered_speed=True, figsize=(12, 8)):
    """
    Plot a raster plot of inferred spike activity (top) alongside the
    running speed trace (bottom), with blue shading marking stimulus
    presentation windows.
    """
    STIMULUS_SESSION_MAP = {
        "drifting_gratings": "A",
        "static_gratings": "B",
        "natural_scenes": "B",
        "locally_sparse_noise": "C",
    }

    s = data["sessions"][session]
    spikes = s["spikes"]
    t = s["t"]
    n_cells = spikes.shape[0]

    speed_key = "running_speed_filtered" if use_filtered_speed else "running_speed"
    speed = s[speed_key][0]
    t_speed = s[speed_key][1]

    fig, (ax_raster, ax_speed) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Stimulus shading (draw first, so it sits behind the raster/speed lines)
    add_stimulus_shading(ax_raster, data, session, t, xlim=xlim)
    add_stimulus_shading(ax_speed, data, session, t, xlim=xlim)

    # --- Top: spike raster ---
    for cell_idx in range(n_cells):
        spike_times = t[spikes[cell_idx] > threshold]
        ax_raster.vlines(spike_times, cell_idx + 0.5, cell_idx + 1.5,
                         color="black", linewidth=0.5, zorder=2)

    ax_raster.set_ylabel("neuron #")
    ax_raster.set_ylim(0.5, n_cells + 0.5)
    ax_raster.set_title(f"session {session}: spike raster ({n_cells} neurons)")

    # --- Bottom: running speed ---
    ax_speed.plot(t_speed, speed, lw=0.8, color="darkorange", zorder=2)
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("running speed\n(cm/s)")

    if xlim is not None:
        ax_raster.set_xlim(xlim)
        ax_speed.set_xlim(xlim)

    plt.tight_layout()
    return fig, (ax_raster, ax_speed)

def plot_spike_raster_subset_with_running(data, session, cell_ids, threshold=0.05, xlim=None,
                                             use_filtered_speed=True, figsize=(12, 6)):
    """
    Plot a raster plot for a SUBSET of neurons, with running speed shown
    below, and blue shading marking stimulus presentation windows.

    Parameters
    ----------
    cell_ids : list of int
        Which neuron indices to plot (e.g. [8, 15, 25, 36]).
    threshold : float
        Minimum spike value to count as a "spike event" for raster display.
    xlim : tuple, optional
        (start_time, end_time) in seconds to zoom into a specific window.
    """
    s = data["sessions"][session]
    spikes = s["spikes"]
    t = s["t"]

    speed_key = "running_speed_filtered" if use_filtered_speed else "running_speed"
    speed = s[speed_key][0]
    t_speed = s[speed_key][1]

    fig, (ax_raster, ax_speed) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Stimulus shading (draw first, so it sits behind the raster/speed lines)
    add_stimulus_shading(ax_raster, data, session, t, xlim=xlim)
    add_stimulus_shading(ax_speed, data, session, t, xlim=xlim)

    # --- Top: raster for selected neurons only ---
    for row_idx, cell_idx in enumerate(cell_ids):
        spike_times = t[spikes[cell_idx] > threshold]
        ax_raster.vlines(spike_times, row_idx + 0.5, row_idx + 1.5,
                          color="black", linewidth=0.8, zorder=2)

    ax_raster.set_yticks(np.arange(1, len(cell_ids) + 1))
    ax_raster.set_yticklabels([f"neuron {c}" for c in cell_ids], fontsize=15)
    ax_raster.set_ylim(0.5, len(cell_ids) + 0.5)
    ax_raster.set_title(f"session {session}: spike raster (neurons {cell_ids})", fontsize=15)

    # --- Bottom: running speed ---
    ax_speed.plot(t_speed, speed, lw=0.8, color="darkorange", zorder=2)
    ax_speed.set_xlabel("time (s)", fontsize=15)
    ax_speed.set_ylabel("running speed\n(cm/s)", fontsize=15)
    ax_speed.tick_params(axis="both", labelsize=15)


    if xlim is not None:
        ax_raster.set_xlim(xlim)
        ax_speed.set_xlim(xlim)

    plt.tight_layout()
    return fig, (ax_raster, ax_speed)

def compare_neuron_activity_by_movement(data, session, stim_name, cell_ids,
                                          phase_key="running_speed_phase"):
    """
    For specific neurons, compare spike activity during stimulus presentation
    trials, split by whether the mouse was moving or stationary during
    each trial (majority vote based on running_speed_phase).

    Parameters
    ----------
    cell_ids : list of int
        Which neuron indices to analyze (e.g. [25, 36]).

    Returns
    -------
    pd.DataFrame with columns: cell, movement_state, sum_spikes, start, end
    """
    s = data["sessions"][session]
    spikes = s["spikes"]
    phase = s[phase_key]  # 0 = still, 1 = moving, -1 = invalid
    stim_table = s["stim_tables"][stim_name].copy()

    if "frame" in stim_table.columns:
        stim_table = stim_table[stim_table["frame"] != -1].reset_index(drop=True)

    rows = []
    for _, row in stim_table.iterrows():
        start, end = int(row["start"]), int(row["end"])
        trial_phase = phase[start:end + 1]
        trial_phase = trial_phase[trial_phase != -1]
        if len(trial_phase) == 0:
            continue

        # Majority vote: is the mouse moving for most of this trial?
        state = "moving" if np.mean(trial_phase) > 0.5 else "stationary"

        for cell_id in cell_ids:
            trial_spikes = spikes[cell_id, start:end + 1]
            rows.append({
                "cell": cell_id,
                "movement_state": state,
                "sum_spikes": np.nansum(trial_spikes),
                "start": start,
                "end": end,
            })

    return pd.DataFrame(rows)

def plot_neuron_activity_by_movement(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(data=df, x="cell", y="sum_spikes", hue="movement_state", ax=ax,
                showfliers=False, palette="Set2",
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 6})

    ax.set_xlabel("neuron #")
    ax.set_ylabel("total spike activity per trial")
    ax.set_title("Neural activity: moving vs. stationary")
    ax.legend(title="movement state")

    return ax

def compare_neuron_activity_by_movement(data, session, stim_name, cell_ids,
                                          phase_key="running_speed_phase"):
    """
    For specific neurons, compare spike activity during stimulus presentation
    trials, split by whether the mouse was moving or stationary during
    each trial (majority vote based on running_speed_phase).

    Parameters
    ----------
    cell_ids : list of int
        Which neuron indices to analyze (e.g. [25, 36]).

    Returns
    -------
    pd.DataFrame with columns: cell, movement_state, sum_spikes, start, end
    """
    s = data["sessions"][session]
    spikes = s["spikes"]
    phase = s[phase_key]  # 0 = still, 1 = moving, -1 = invalid
    stim_table = s["stim_tables"][stim_name].copy()

    if "frame" in stim_table.columns:
        stim_table = stim_table[stim_table["frame"] != -1].reset_index(drop=True)

    rows = []
    for _, row in stim_table.iterrows():
        start, end = int(row["start"]), int(row["end"])
        trial_phase = phase[start:end + 1]
        trial_phase = trial_phase[trial_phase != -1]
        if len(trial_phase) == 0:
            continue

        # Majority vote: is the mouse moving for most of this trial?
        state = "moving" if np.mean(trial_phase) > 0.5 else "stationary"

        for cell_id in cell_ids:
            trial_spikes = spikes[cell_id, start:end + 1]
            rows.append({
                "cell": cell_id,
                "movement_state": state,
                "sum_spikes": np.nansum(trial_spikes),
                "start": start,
                "end": end,
            })

    return pd.DataFrame(rows)

def plot_neuron_activity_by_movement(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(data=df, x="cell", y="sum_spikes", hue="movement_state", ax=ax,
                showfliers=False, palette="Set2",
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 6})

    ax.set_xlabel("neuron #")
    ax.set_ylabel("total spike activity per trial")
    ax.set_title("Neural activity: moving vs. stationary")
    ax.legend(title="movement state")

    return ax

def compare_activity_by_movement_all_stimuli(data, stimulus_session_map=STIMULUS_SESSION_MAP,
                                                 cell_ids=None, phase_key="running_speed_phase"):
    """
    For each stimulus type, compare spike activity between moving and
    stationary trials, separately for each specified neuron.

    Parameters
    ----------
    cell_ids : list of int, optional
        If specified, compute activity separately for each of these neurons.
        If None, use all cells combined (population-level sum, labeled as "all_cells").

    Returns
    -------
    pd.DataFrame with columns: stimulus, cell, movement_state, sum_spikes
    """
    rows = []

    for stim_name, session in stimulus_session_map.items():
        s = data["sessions"][session]
        spikes = s["spikes"]
        phase = s[phase_key]
        stim_table = s["stim_tables"][stim_name].copy()

        if "frame" in stim_table.columns:
            stim_table = stim_table[stim_table["frame"] != -1].reset_index(drop=True)

        cells_to_use = cell_ids if cell_ids is not None else [None]  # None = all cells combined

        for _, row in stim_table.iterrows():
            start, end = int(row["start"]), int(row["end"])
            trial_phase = phase[start:end + 1]
            trial_phase = trial_phase[trial_phase != -1]
            if len(trial_phase) == 0:
                continue

            state = "moving" if np.mean(trial_phase) > 0.5 else "stationary"

            for cell_id in cells_to_use:
                if cell_id is None:
                    trial_activity = np.nansum(spikes[:, start:end + 1])
                    cell_label = "all_cells"
                else:
                    trial_activity = np.nansum(spikes[cell_id, start:end + 1])
                    cell_label = f"neuron{cell_id}"

                rows.append({
                    "stimulus": stim_name,
                    "neuron": cell_label,
                    "movement_state": state,
                    "sum_spikes": trial_activity,
                })

    return pd.DataFrame(rows)

def plot_all_neurons_movement_grid(data, stimulus_session_map=STIMULUS_SESSION_MAP,
                                       phase_key="running_speed_phase", cols=6):
    df_all = compare_activity_by_movement_all_stimuli(data, cell_ids=list(range(47)))

    n_cells = 47
    rows = int(np.ceil(n_cells / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2), sharey=False)

    for i, ax in enumerate(axes.flat):
        if i >= n_cells:
            ax.axis("off")
            continue
        cell_df = df_all[df_all["neuron"] == f"neuron{i}"]
        sns.boxplot(data=cell_df, x="stimulus", y="sum_spikes", hue="movement_state",
                    ax=ax, showfliers=False, palette="Set2", legend=False)
        ax.set_title(f"neuron {i}", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)

    plt.tight_layout()
    return fig, axes

def plot_activity_by_movement_per_cell(df):
    g = sns.catplot(
        data=df, x="stimulus", y="sum_spikes", hue="movement_state",
        col="neuron", kind="box", showfliers=False, palette="Set2",
        height=4, aspect=1.1
    )
    g.set_xticklabels(rotation=15, ha="right",fontsize=18)
    g.set_axis_labels("stimulus", "total spike activity per trial", fontsize=20)
    g.fig.suptitle("Neural activity by movement state", y=1.05, fontsize=20)

    # Set title and tick label sizes for each subplot
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize=18)
        ax.tick_params(axis="y", labelsize=18)

    # Increase legend font size
    if g.legend is not None:
        g.legend.set_title("movement state", prop={"size": 18})
        for text in g.legend.get_texts():
            text.set_fontsize(18)
    return g

def get_running_speed_by_stimulus(data, stimulus_session_map=STIMULUS_SESSION_MAP, filtered=True):
    """
    Extract running speed values during each stimulus's presentation trials.
    Uses the filtered (smoothed, clipped) speed trace by default.
    """
    stimulus_speed = {}

    for stim_name, session in stimulus_session_map.items():
        s = data["sessions"][session]
        key = "running_speed_filtered" if filtered else "running_speed"
        speed = s[key][0]
        stim_table = s["stim_tables"][stim_name]

        starts = stim_table["start"].astype(int).values
        ends = stim_table["end"].astype(int).values

        speed_values = [speed[start:end + 1] for start, end in zip(starts, ends)]
        stimulus_speed[stim_name] = np.concatenate(speed_values) if speed_values else np.array([])

    return stimulus_speed


def plot_speed_by_stimulus(stimulus_speed, ax=None):
    """
    Plot running speed distributions across the four stimulus types as boxplots,
    with mean values shown as markers and text labels.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    rows = []
    for stim_name, speeds in stimulus_speed.items():
        for v in speeds:
            rows.append({"stimulus": stim_name, "speed": v})
    df_long = pd.DataFrame(rows)

    sns.boxplot(data=df_long, x="stimulus", y="speed", ax=ax,
                showfliers=False, width=0.5, palette="Set2",
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 7})

    # Annotate each box with its mean value as text
    stim_order = list(stimulus_speed.keys())
    for i, stim_name in enumerate(stim_order):
        mean_val = np.mean(stimulus_speed[stim_name])
        ax.text(i, mean_val, f"{mean_val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black")

    ax.set_xlabel("stimulus")
    ax.set_ylabel("running speed (cm/s)")
    ax.set_title("Running speed across stimulus types")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    return ax

def show_all_natural_scenes(data, ncols=12):
    '''this function could show all the natural stimulation
    118 in total'''
    tmpl = data["templates"]["natural_scenes"]
    n = tmpl.shape[0]                          # Total 118 frame
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 1.5, nrows * 1.5),
        constrained_layout=True,
    )
    axes = axes.flat

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(tmpl[i], cmap="gray")
            ax.set_title(str(i), fontsize=6)
        ax.axis("off")




#----------------------------- Running speed - neural activity analysis utils #-----------------------------

def get_speed_per_trial(data, session, stim_name, frame_to_category, filtered=True):
    """
    For each stimulus presentation trial, extract the mean running speed
    during that trial, and tag it with its frame number and category.

    Returns
    -------
    pd.DataFrame with columns: frame, category, mean_speed, start, end
    """
    # Categorized the pictures into aversive and neutral
    aversive_frames = [
        0, 1, 2,  # bears
        4,  # Bird
        6, 7,  # lions
        8, 9,  # elephant
        10, 11, 12,  # tigers close-up
        13, 14,  # coyote howling
        15, 16,  # cheetah
        17,  # leopard
        18,  # eagle
        19,  # birds fighting
        21,  # bird
        22,  # leopard/jaguar close-up
        23,  # monkey
        25,  # Otter
        27,  # tiger
        28,  # bird
        29,  # coyotes
        34,  # elephants
        35,  # bird
        39,  # leopard camouflaged
        47,  # bobcat
        49,  # bird
        50,  # wolf/coyote
        51, 52,  # bird
        55, 56,  # hawk flying
        58,  # owl
        102,  # bird
    ]

    frame_to_category = {i: "aversive" for i in aversive_frames}
    for i in range(118):
        if i not in frame_to_category:
            frame_to_category[i] = "neutral"
    s = data["sessions"][session]
    key = "running_speed_filtered" if filtered else "running_speed"
    speed = s[key][0]
    stim_table = s["stim_tables"][stim_name].copy()

    # Exclude blank sweeps (frame == -1, no image shown)
    stim_table = stim_table[stim_table["frame"] != -1].reset_index(drop=True)

    mean_speeds = []
    for _, row in stim_table.iterrows():
        start, end = int(row["start"]), int(row["end"])
        trial_speed = speed[start:end + 1]
        mean_speeds.append(np.nanmean(trial_speed))

    stim_table["mean_speed"] = mean_speeds
    stim_table["category"] = stim_table["frame"].map(frame_to_category)

    return stim_table

def plot_speed_by_category(trial_df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.boxplot(data=trial_df, x="category", y="mean_speed", ax=ax,
                showfliers=False, width=0.5, palette="Set2",
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 7})

    ax.set_xlabel("image category")
    ax.set_ylabel("mean running speed per trial (cm/s)")
    ax.set_title("Running speed by image category (aversive vs neutral, session B)")

    return ax

def get_neural_activity_per_trial(data, session, stim_name, frame_to_category):
    """
    For each stimulus presentation trial, extract the total (summed) spike activity
    during that trial, and tag it with its frame number and category.

    Returns
    -------
    pd.DataFrame with columns: frame, category, sum_spikes, start, end
    """
    s = data["sessions"][session]
    spikes = s["spikes"]  # shape (n_cells, n_timepoints)
    stim_table = s["stim_tables"][stim_name].copy()

    # Exclude blank sweeps (frame == -1, no image shown)
    stim_table = stim_table[stim_table["frame"] != -1].reset_index(drop=True)

    sum_spikes = []
    for _, row in stim_table.iterrows():
        start, end = int(row["start"]), int(row["end"])
        trial_spikes = spikes[:, start:end + 1]
        sum_spikes.append(np.nansum(trial_spikes))

    stim_table["sum_spikes"] = sum_spikes
    stim_table["category"] = stim_table["frame"].map(frame_to_category)

    return stim_table

def get_neural_activity_by_stimulus(data, stimulus_session_map=STIMULUS_SESSION_MAP):
    """
    Extract total (summed) spike activity during each stimulus's presentation trials,
    across all four stimulus types (spanning sessions A, B, C).

    Returns
    -------
    dict: {stimulus_name: np.ndarray of per-trial summed spike values}
    """
    stimulus_activity = {}

    for stim_name, session in stimulus_session_map.items():
        s = data["sessions"][session]
        spikes = s["spikes"]
        stim_table = s["stim_tables"][stim_name]

        starts = stim_table["start"].astype(int).values
        ends = stim_table["end"].astype(int).values

        trial_sums = [np.nansum(spikes[:, start:end + 1]) for start, end in zip(starts, ends)]
        stimulus_activity[stim_name] = np.array(trial_sums)

    return stimulus_activity

def plot_activity_by_stimulus(stimulus_activity, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    rows = []
    for stim_name, values in stimulus_activity.items():
        for v in values:
            rows.append({"stimulus": stim_name, "sum_spikes": v})
    df_long = pd.DataFrame(rows)

    sns.boxplot(data=df_long, x="stimulus", y="sum_spikes", ax=ax,
                showfliers=False, width=0.5, palette="Set2",
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 7})

    stim_order = list(stimulus_activity.keys())
    for i, stim_name in enumerate(stim_order):
        mean_val = np.mean(stimulus_activity[stim_name])
        ax.text(i, mean_val, f"{mean_val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black")

    ax.set_xlabel("stimulus")
    ax.set_ylabel("total population spike activity per trial")
    ax.set_title("Neural activity across stimulus types")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    return ax

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


def fit_tuning_curves(
        data,
        session="B",
        stimulus="drifting_gratings",
        parameter="orientation",
        speed_phase=None,
):
    """Compute per-trial spike counts and fit a von Mises tuning curve per cell,
    in one pass over the stimulus table.
    Also allows filtering trials based on running speed phase (e.g., still vs running).
    """
    session_data = data["sessions"][session]
    spikes = np.asarray(session_data["spikes"], dtype=float)
    stim_table = session_data["stim_tables"][stimulus]
    running_speed_phase = session_data["running_speed_phase"]

    n_cells = spikes.shape[0]
    # Hold spike counts and stimulus values for each cell
    counts_per_cell = [[] for _ in range(n_cells)]
    stim_per_cell = [[] for _ in range(n_cells)]

    for _, trial in stim_table.iterrows():
        start_idx = int(trial["start"])
        stop_idx = int(trial["end"])
        stim_value = trial[parameter]

        if speed_phase is not None:
            samples_in_phase = running_speed_phase[start_idx:stop_idx] == speed_phase
            if not np.any(samples_in_phase):
                continue
            # Count spikes only in the samples that match the specified speed phase
            spike_count = np.sum(spikes[:, start_idx:stop_idx][:, samples_in_phase], axis=1)
        else:
            spike_count = np.sum(spikes[:, start_idx:stop_idx], axis=1)

        # Store spike counts and stimulus values for each cell
        for cell_idx in range(n_cells):
            counts_per_cell[cell_idx].append(spike_count[cell_idx])
            stim_per_cell[cell_idx].append(stim_value)

    # Fit tuning curves for each cell
    fitted_params = []
    for cell_idx in range(n_cells):
        counts = np.asarray(counts_per_cell[cell_idx], dtype=float)  # shape (n_trials,)
        stim = np.asarray(stim_per_cell[cell_idx], dtype=float)  # shape (n_trials,)

        # Remove NaN or infinite values
        valid = np.isfinite(counts) & np.isfinite(stim)
        counts = counts[valid]
        stim = stim[valid]

        params = tuningCurve(counts, stim)
        fitted_params.append(
            {
                "cell": cell_idx,
                "stimulus_parameters": stim,
                "spike_counts": counts,
                "van-mises-params": params,
                "speed_phase": speed_phase
            }
        )

    return fitted_params

def compute_tuning_significance(fitted_params, psi=2, niters=1000, random_seed=2046):
    """Permutation-based tuning significance per cell.
    """
    p_values = {}
    for entry in fitted_params:
        cell = int(entry["cell"])
        stim = np.asarray(entry["stimulus_parameters"], dtype=float)
        counts = np.asarray(entry["spike_counts"], dtype=float)

        p_value, q_abs, _ = testTuning(
            counts, stim, psi=psi, niters=niters, random_seed=random_seed
        )
        p_values[cell] = p_value

    return p_values

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

def plot_tuning_curves(
    fitted_params,
    p_values=None,
    n_cols=4,
    alpha=0.05,
    psi=2,
    only_significant=False,
):
    """Plot per-cell trial rates and von Mises fit..
    """
    if only_significant:
        if p_values is None:
            raise ValueError("p_values must be provided when only_significant=True.")
        fitted_params = [e for e in fitted_params if p_values.get(int(e["cell"]), 1) < alpha]

    n_cells = len(fitted_params)

    n_rows = int(np.ceil(n_cells / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False)
    theta_range = np.linspace(0, np.max([entry["stimulus_parameters"] for entry in fitted_params]), 200)
    speed_phase = fitted_params[0]['speed_phase'] if fitted_params else 0
    curve_color = "steelblue" if speed_phase == 0 else "crimson"

    for i, entry in enumerate(fitted_params):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        cell = int(entry["cell"])
        stim = np.asarray(entry["stimulus_parameters"], dtype=float)
        counts = np.asarray(entry["spike_counts"], dtype=float)
        unique_stim = np.unique(stim)
        mean_counts = np.array([np.mean(counts[stim == us]) for us in unique_stim])
        alpha_vm, kappa_vm, nu_vm, phi_vm = entry["van-mises-params"]

        ax.scatter(unique_stim, mean_counts, color="black", s=12, label="Mean spike count")
        fitted_curve = vonMises(theta_range, alpha_vm, kappa_vm, nu_vm, phi_vm)
        ax.plot(theta_range, fitted_curve, color=curve_color, label="von Mises fit", linewidth=3)

        is_sig = p_values is not None and p_values.get(cell, 1) < alpha
        ax.set_title(f"Cell {cell}", fontsize=12, color="green" if is_sig and not only_significant else "black")
        if is_sig:
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if not only_significant else "black")
                spine.set_linewidth(2)
        if i == 0:
            ax.set_xlabel("Orientation (deg)", fontsize=12)
            ax.set_ylabel("Spike count", fontsize=12)

        ymax = max(fitted_curve.max(), mean_counts.max(), 1e-6)
        ax.set_ylim(bottom=0, top=ymax * 1.5)

    for j in range(n_cells, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].axis("off")

    fig.tight_layout()
    return fig, axes


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

        T = minimum_nr_of_bins
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