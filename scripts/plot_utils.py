from pathlib import Path
import numpy as np
import matplotlib.pyplot as pp
import arviz as az

# Import PNAS helpers (works both as package or as bare scripts/)
try:
    from .pnas_utils import fig_ax, save_pnas
except Exception:  # notebook-friendly fallback
    from scripts.pnas_utils import fig_ax, save_pnas


# ---- internal: handle wavelength coord name + dtype (str/int) ----
def _aph_key_val(idata, band):
    """
    Return (coord_key, coord_value) for selecting a given band in idata,
    handling both 'λ_aφ' and 'lambda_aph' and str/int labels.
    """
    od = idata.observed_data
    if "λ_aφ" in od.coords:
        key = "λ_aφ"
    elif "lambda_aph" in od.coords:
        key = "lambda_aph"
    else:
        raise KeyError("No wavelength coord 'λ_aφ' or 'lambda_aph' found in observed_data.")
    vals = od[key].values
    # Coerce band to the coord dtype
    if getattr(vals, "dtype", None) is not None and vals.dtype.kind in "OUS":
        return key, str(band)
    return key, int(band)


# ---- 1) HDI ribbons vs observed (single band) ----
def plot_regression_hdi(
    idata,
    var: str = "likelihood",
    band: int | str = 443,
    intervals=(0.95, 0.80, 0.50),
    color_obs="red", alpha_scatter=1.0,
    out: str | Path | None = None,
):
    key, val = _aph_key_val(idata, band)
    ppc = idata.posterior_predictive[var].sel({key: val})  # (chain, draw, obs)
    draws = ppc.stack(sample=("chain", "draw")).transpose("obs_idx", "sample").values  # (obs, samples)
    y = idata.observed_data[var].sel({key: val}).values  # (obs,)

    idx = np.argsort(y)
    x = np.arange(y.size)
    q = {0.95:(0.025,0.975), 0.90:(0.05,0.95), 0.80:(0.10,0.90), 0.63:(0.185,0.815), 0.50:(0.25,0.75)}

    fig, ax = fig_ax(width="one", aspect=0.62)
    for p in sorted(intervals):
        lo, hi = np.quantile(draws, q[p], axis=1)
        ax.fill_between(
            x, lo[idx], hi[idx],
            label=f"{int(p*100)}% interval",
            alpha={0.95:0.25, 0.90:0.28, 0.80:0.32, 0.63:0.35, 0.50:0.40}.get(p, 0.3),
        )
    ax.scatter(
        x, y[idx], s=12, color=color_obs, label="observed", zorder=3, 
        alpha=alpha_scatter, edgecolor='k')
    ax.set_xlabel("Observation # (sorted by observed)")
    ax.set_ylabel(rf"$\log_{{10}}\,a_{{\phi}}({str(band)}\,\mathrm{{nm}})$")
    ax.legend(frameon=False)
    fig.tight_layout()
    if out is not None:
        save_pnas(fig, out)
    return fig, ax


# ---- 2) 3-way PPC panels (per-band, optional 'All targets') ----
def plot_3_way_ppc(
    idata,
    var: str = "likelihood",
    bands: tuple[int | str, ...] = (443, 555, 670),
    colors=('0.6','0.2','C1'),
    kind='kde',
    include_all: bool = False,
    out: str | Path | None = None,
):
    if include_all:
        fig, axs = fig_ax(width="two", aspect=0.50, nrows=2, ncols=2, sharex=True, sharey=True)
        axs = np.atleast_1d(axs).ravel()
        az.plot_ppc(idata, var_names=[var], ax=axs[-1], colors=colors, kind=kind)
        axs[-1].set_title("All targets")
        axs[-1].set_xlabel(r"$\log_{10}\,a_{\phi}$")
        target_axes = axs[:len(bands)]
    else:
        fig, axs = fig_ax(width="two", aspect=0.55, ncols=3, sharex=True, sharey=True)
        target_axes = np.atleast_1d(axs)

    for ax, band in zip(target_axes, bands):
        key, val = _aph_key_val(idata, band)
        az.plot_ppc(
            idata, var_names=[var], coords={key: val},
            ax=ax, colors=colors, kind=kind, legend=False
        )
        ax.set_title(rf"$\lambda = {str(band)}\,\mathrm{{nm}}$")
        ax.set_xlabel(r"$\log_{10}\,a_{\phi}$")
    fig.tight_layout()
    if out is not None:
        save_pnas(fig, out)
    return fig, axs


# ---- 3) Prior vs Posterior PPC comparison (optional prior) ----
def compare_ppc(
    idata,
    var: str = "likelihood",
    band: int | str | None = None,    # None = all targets
    kind: str = "kde",
    colors = ('0.6','0.2','C1'),
    include_observed: bool = True,
    xlims: tuple[float,float] | None = None,
    ylims_left: tuple[float,float] | None = None,
    ylims_right: tuple[float,float] | None = None,
    out: str | Path | None = None,
):
    coords = None
    if band is not None:
        key, val = _aph_key_val(idata, band)
        coords = {key: val}

    has_prior = (getattr(idata, "prior_predictive", None) is not None) and (var in idata.prior_predictive)
    if not has_prior:
        fig, ax = fig_ax(width="one", aspect=0.62)
        az.plot_ppc(idata, var_names=[var], coords=coords, ax=ax, colors=colors, kind=kind)
        ax.set_title("Posterior predictive")
        if out is not None:
            save_pnas(fig, out)
        return fig, np.array([ax])

    fig, ax = fig_ax(width="two", aspect=0.62, ncols=2, sharex=True)

    az.plot_ppc(
        idata, var_names=[var], coords=coords, group="prior",
        observed=include_observed, ax=ax[0], colors=colors, kind=kind, legend=False
    )
    ax[0].set_title("Prior predictive")

    az.plot_ppc(
        idata, var_names=[var], coords=coords, group="posterior",
        observed=include_observed, ax=ax[1], colors=colors, kind=kind, legend=False
    )
    ax[1].set_title("Posterior predictive")

    # Auto x-limits from observed + posterior draws
    if xlims is None:
        obs = idata.observed_data[var]
        if coords is not None:
            obs = obs.sel(**coords)
        o = np.asarray(obs).ravel()
        ppc = idata.posterior_predictive[var]
        if coords is not None:
            ppc = ppc.sel(**coords)
        d = np.asarray(ppc).reshape(-1)
        lo = np.nanmin([o.min(), d.min()])
        hi = np.nanmax([o.max(), d.max()])
        pad = 0.03 * (hi - lo + 1e-12)
        xlims = (lo - pad, hi + pad)
    ax[0].set_xlim(*xlims); ax[1].set_xlim(*xlims)

    xlabel = (r"$\log_{10}\,a_{\phi}$"
              if band is None else rf"$\log_{{10}}\,a_{{\phi}}({str(band)}\,\mathrm{{nm}})$")
    ax[0].set_xlabel(xlabel); ax[1].set_xlabel(xlabel)

    if ylims_left is not None:
        ax[0].set_ylim(*ylims_left)
    if ylims_right is not None:
        ax[1].set_ylim(*ylims_right)

    fig.tight_layout()
    if out is not None:
        save_pnas(fig, out)
    return fig, ax


# ---- 4) PPC + LOO-PIT (two-panel) ----
def plot_ppc_pit(
    idata,
    var: str = "likelihood",
    band: int | str | None = None,    # None = all targets
    kind: str = "kde",
    colors = ('0.6','0.2','C1'),
    include_observed: bool = True,
    out: str | Path | None = None,
):
    fig, (ax_ppc, ax_pit) = fig_ax(width="two", aspect=0.62, ncols=2)

    # --- PPC (left) ---
    ppc_kwargs = dict(var_names=[var], kind=kind, colors=colors, legend=False, observed=include_observed)
    if band is None:
        az.plot_ppc(idata, ax=ax_ppc, **ppc_kwargs)
        xlabel = r"$\log_{10}\,a_{\phi}$"
        ax_ppc.set_title("Posterior predictive (all targets)")
    else:
        key, val = _aph_key_val(idata, band)
        az.plot_ppc(idata, ax=ax_ppc, coords={key: val}, **ppc_kwargs)
        xlabel = rf"$\log_{{10}}\,a_{{\phi}}({str(band)}\,\mathrm{{nm}})$"
        ax_ppc.set_title(rf"Posterior predictive ($\lambda={str(band)}\,\mathrm{{nm}}$)")
    ax_ppc.set_xlabel(xlabel)
    ax_ppc.set_ylabel("Estimated density")

    # --- LOO-PIT (right) ---
    if band is None:
        # Requires idata.log_likelihood to be present for the var
        az.plot_loo_pit(idata, y=var, ax=ax_pit, legend=True)
    else:
        key, val = _aph_key_val(idata, band)
        y_obs = idata.observed_data[var].sel({key: val}).values           # (obs,)
        y_ppc = idata.posterior_predictive[var].sel({key: val}).values    # (chain, draw, obs)
        y_hat = y_ppc.reshape((-1, y_ppc.shape[-1]))                      # (draws, obs)
        az.plot_loo_pit(y=y_obs, y_hat=y_hat, ax=ax_pit, legend=True)
    ax_pit.set_xlabel(r"$\mathrm{PIT}(y_i \mid y_{-i})$")
    ax_pit.set_ylabel("Estimated density")
    ax_pit.set_title("LOO-PIT")

    fig.tight_layout()
    if out is not None:
        save_pnas(fig, out)
    return {"fig_handle": fig, "ppc_ax_hdl": ax_ppc, "pit_ax_hdl": ax_pit}


# --- NEW: PPC + LOO-PIT grid over all bands ---
def plot_ppc_pit_grid(
    idata,
    var: str = "likelihood",
    bands: list | tuple | None = None,   # None => use all bands in idata
    kind: str = "kde",
    colors = ("0.6", "0.2", "C1"),
    include_observed: bool = True,
    out: str | Path | None = None,
):
    """
    Multi-panel figure: for each band, PPC (left col) and LOO-PIT (right col).
    Works with string wavelength coords (λ_aφ='443','555','670'). Requires
    idata.log_likelihood[var] for LOO-PIT.
    """
    # figure out band coord and labels
    od = idata.observed_data
    if "λ_aφ" in od.coords:
        key = "λ_aφ"
    elif "lambda_aph" in od.coords:
        key = "lambda_aph"
    else:
        raise KeyError("No wavelength coord 'λ_aφ' or 'lambda_aph' in observed_data.")

    labels = list(map(str, od[key].values.tolist()))
    if bands is None:
        bands = labels
    else:
        bands = list(map(str, bands))  # normalize to strings for selection

    # size: two-column width; scale height with rows
    nrows = len(bands)
    fig, axs = fig_ax(width="two", aspect=0.34 * nrows, nrows=nrows, ncols=2, sharex=False, sharey=False)
    axs = np.atleast_1d(axs).reshape(nrows, 2)

    # sanity: need log_likelihood to compute LOO-PIT
    has_ll = (getattr(idata, "log_likelihood", None) is not None) and (var in idata.log_likelihood)
    if not has_ll:
        raise ValueError(f"plot_ppc_pit_grid requires idata.log_likelihood['{var}'].")

    # helper: slice *every* group by band
    def _sel_band_idata(id_, key_, val_):
        new = id_.copy()
        for grp in ("observed_data", "posterior_predictive", "prior_predictive",
                    "log_likelihood", "posterior", "prior", "sample_stats", "constant_data"):
            ds = getattr(new, grp, None)
            if ds is not None and (key_ in ds.dims or key_ in ds.coords):
                setattr(new, grp, ds.sel({key_: val_}))
        return new

    # plot rows
    for r, b in enumerate(bands):
        # left: PPC
        az.plot_ppc(
            idata, var_names=[var], coords={key: b},
            ax=axs[r, 0], kind=kind, colors=colors, legend=False, observed=include_observed
        )
        axs[r, 0].set_title(rf"PPC  ($\lambda={b}\,\mathrm{{nm}}$)")
        axs[r, 0].set_xlabel(rf"$\log_{{10}}\,a_{{\phi}}({b}\,\mathrm{{nm}})$")
        axs[r, 0].set_ylabel("Estimated density")

        # right: LOO-PIT (use band-sliced idata so ArviZ can compute weights)
        idata_b = _sel_band_idata(idata, key, b)
        az.plot_loo_pit(idata=idata_b, y=var, ax=axs[r, 1], legend=(r == 0))
        axs[r, 1].set_title("LOO-PIT")
        axs[r, 1].set_xlabel(r"$\mathrm{PIT}(y_i \mid y_{-i})$")
        axs[r, 1].set_ylabel("Estimated density")

    fig.tight_layout()
    if out is not None:
        save_pnas(fig, out)
    return fig, axs
