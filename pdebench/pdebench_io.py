"""Reader for the two PDEBench file layouts used in this project.

Layout A ("flat"): advection, Burgers, 1D reaction-diffusion, Darcy.
    /tensor          (n_samples, n_t, n_x)      or (n_samples, 1, n_x, n_y) for Darcy
    /x-coordinate    (n_x,)      /t-coordinate  (n_t + 1,)
    file attributes hold the PDE parameters (beta, Nu, rho).

Layout B ("per-sample groups"): diffusion-sorption, 2D diffusion-reaction,
shallow water.
    /0000/data       (n_t, n_x, n_c)            or (n_t, n_x, n_y, n_c)
    /0000/grid/{t,x[,y]}
    group attribute "config" is a YAML string holding the simulation parameters.

Typical use:

    from pdebench_io import open_pde, load_sample
    info = open_pde(r"E:\\...\\1D_Advection_Sols_beta0.4.hdf5")
    print(info)                       # layout, samples, shapes, parameters
    s = load_sample(info, 0)          # dict with u, t, x, (y), params
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import numpy as np
import h5py


@dataclass
class PDEFile:
    path: Path
    layout: str                     # "flat" or "groups"
    n_samples: int
    field_shape: tuple              # shape of one sample, without the sample axis
    params: dict = field(default_factory=dict)
    sample_keys: list[str] | None = None

    def __repr__(self) -> str:
        return (f"PDEFile({self.path.name}, layout={self.layout}, "
                f"n_samples={self.n_samples}, sample shape={self.field_shape}, "
                f"params={self.params})")


def _parse_config(text: str) -> dict:
    """Extract the numeric entries of the embedded YAML configuration.

    pyyaml is used when available; otherwise a small regular-expression
    fallback picks up the `key: value` pairs, which is all this project needs.
    """
    try:
        import yaml
        cfg = yaml.safe_load(text)
        sim = cfg.get("sim", cfg) if isinstance(cfg, dict) else {}
        return {k: v for k, v in sim.items() if isinstance(v, (int, float))}
    except Exception:
        out = {}
        for k, v in re.findall(r"^\s{2,}([A-Za-z_][\w]*):\s*([-\d.eE+]+)\s*$", text, re.M):
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out


def open_pde(path: str | Path) -> PDEFile:
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "tensor" in f:                                    # layout A
            t = f["tensor"]
            params = {k: (float(v) if np.isscalar(v) else v) for k, v in f.attrs.items()}
            return PDEFile(path, "flat", t.shape[0], tuple(t.shape[1:]), params)
        keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))
        if not keys:
            raise ValueError(f"unrecognised layout: {path}")
        g = f[keys[0]]                                        # layout B
        params = _parse_config(g.attrs["config"]) if "config" in g.attrs else {}
        return PDEFile(path, "groups", len(keys), tuple(g["data"].shape), params, keys)


def load_sample(info: PDEFile, index: int = 0) -> dict[str, Any]:
    """Load one sample as {u, t, x, [y], params}.

    u has shape (n_t, n_x) in 1D and (n_t, n_x, n_y) in 2D; a trailing channel
    axis is kept only when the file stores more than one channel.
    """
    with h5py.File(info.path, "r") as f:
        if info.layout == "flat":
            u = np.asarray(f["tensor"][index])
            x = np.asarray(f["x-coordinate"])
            t = np.asarray(f["t-coordinate"]) if "t-coordinate" in f else np.arange(u.shape[0])
            y = np.asarray(f["y-coordinate"]) if "y-coordinate" in f else None
            out = dict(u=u, t=t[: u.shape[0]], x=x, params=dict(info.params))
            if y is not None:
                out["y"] = y
            if "nu" in f:                                     # Darcy: coefficient field
                out["nu"] = np.asarray(f["nu"][index])
            return out
        key = info.sample_keys[index]
        g = f[key]
        u = np.asarray(g["data"])
        if u.shape[-1] == 1:
            u = u[..., 0]
        grid = g["grid"]
        out = dict(u=u, t=np.asarray(grid["t"]), x=np.asarray(grid["x"]),
                   params=_parse_config(g.attrs["config"]) if "config" in g.attrs else {})
        if "y" in grid:
            out["y"] = np.asarray(grid["y"])
        return out


def summarise(path: str | Path, index: int = 0) -> str:
    info = open_pde(path)
    s = load_sample(info, index)
    u = s["u"]
    dx = float(s["x"][1] - s["x"][0])
    dt = float(s["t"][1] - s["t"][0])
    lines = [repr(info),
             f"  sample {index}: u{u.shape}  dx={dx:.5g}  dt={dt:.5g}  "
             f"range=[{u.min():.4g}, {u.max():.4g}]"]
    if s["params"]:
        lines.append(f"  parameters: {s['params']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(summarise(p)); print()
