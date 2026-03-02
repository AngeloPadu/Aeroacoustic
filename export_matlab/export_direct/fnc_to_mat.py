#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# ----------------- helpers -----------------

def _decode_null_terminated(var) -> list[str]:
    raw = var[:]
    b = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
    s = b.decode("ascii", errors="ignore")
    parts = s.split("\x00")
    return [p for p in parts if p]

def _as_numpy(arr):
    if hasattr(arr, "mask"):
        return np.array(arr.filled(np.nan))
    return np.array(arr)

def _ensure_coords_nx3(coords):
    c = np.array(coords, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"coords deve essere 2D, trovato {c.ndim}D")
    if c.shape[1] == 3:
        return c
    if c.shape[0] == 3:
        return c.T
    raise ValueError(f"coords shape inattesa: {c.shape}")

def _unique_sorted_snapped(v, decimals=12):
    v = np.round(np.asarray(v, dtype=float), decimals=decimals)
    return np.unique(v)

def _build_grid_axes(coords_m, decimals=12):
    X = _unique_sorted_snapped(coords_m[:, 0], decimals)
    Y = _unique_sorted_snapped(coords_m[:, 1], decimals)
    Zraw = _unique_sorted_snapped(coords_m[:, 2], decimals)
    Z = None if Zraw.size <= 1 else Zraw
    return X, Y, Z

def _point_indices_on_axes(coords_m, X, Y, Z, decimals=12):
    cx = np.round(coords_m[:, 0], decimals)
    cy = np.round(coords_m[:, 1], decimals)
    cz = np.round(coords_m[:, 2], decimals)

    xi = np.searchsorted(X, cx)
    yi = np.searchsorted(Y, cy)

    if Z is None:
        zi = np.zeros_like(xi)
        Nz = 1
    else:
        zi = np.searchsorted(Z, cz)
        Nz = Z.size

    # bounds check
    if (xi.min() < 0 or xi.max() >= X.size or
        yi.min() < 0 or yi.max() >= Y.size or
        zi.min() < 0 or zi.max() >= Nz):
        raise ValueError("Alcuni punti non ricadono sulla griglia: aumenta --round-decimals oppure dati non strutturati.")

    # warning duplicati
    Ny, Nx = Y.size, X.size
    idx_flat = (zi * Ny + yi) * Nx + xi if Z is not None else (yi * Nx + xi)
    if np.unique(idx_flat).size != idx_flat.size:
        print("[WARN] Ci sono punti duplicati sulla stessa cella di griglia: l’ultimo sovrascrive i precedenti.")

    return xi, yi, zi

def _grid_one_field(values_1d, xi, yi, zi, X, Y, Z):
    """
    values_1d: (npoints,)
    ritorna:
      2D -> (Ny, Nx)
      3D -> (Ny, Nx, Nz)  (ordine MATLAB-friendly: y,x,z)
    """
    values_1d = np.asarray(values_1d, dtype=float)
    npoints = values_1d.size
    Ny, Nx = Y.size, X.size

    if Z is None:
        nspatial = Ny * Nx
        flat = np.full((nspatial,), np.nan, dtype=float)
        idx_flat = yi * Nx + xi
        flat[idx_flat] = values_1d
        return flat.reshape((Ny, Nx))

    Nz = Z.size
    nspatial = Nz * Ny * Nx
    flat = np.full((nspatial,), np.nan, dtype=float)
    idx_flat = (zi * Ny + yi) * Nx + xi
    flat[idx_flat] = values_1d
    grid_zyx = flat.reshape((Nz, Ny, Nx))       # (z,y,x)
    grid_yxz = np.transpose(grid_zyx, (1, 2, 0))  # (y,x,z) -> comodo in MATLAB
    return grid_yxz


# ----------------- conversion -----------------

def _prepare_lattice_maps(ds: Dataset):
    var_short = _decode_null_terminated(ds.variables["variable_short_names"])
    var_lx    = _decode_null_terminated(ds.variables["variable_lattice_unit_names"])

    lx_names   = _decode_null_terminated(ds.variables["lx_names"])
    lx_scales  = _as_numpy(ds.variables["lx_scales"][:]).astype(float)
    lx_offsets = _as_numpy(ds.variables["lx_offsets"][:]).astype(float)
    lx_map = {n: (float(s), float(o)) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

    # scales/offsets per ciascuna variabile
    scales  = np.array([lx_map.get(u, (1.0, 0.0))[0] for u in var_lx], dtype=float)
    offsets = np.array([lx_map.get(u, (1.0, 0.0))[1] for u in var_lx], dtype=float)

    idx_map = {name: i for i, name in enumerate(var_short)}
    return var_short, var_lx, lx_map, scales, offsets, idx_map

def _convert_one_set_to_physical(meas_lu, var_short, var_lx, lx_map, scales, offsets, idx_map):
    """
    meas_lu: (nvars, npoints) lattice units
    ritorna: meas_phys (nvars, npoints) in unità fisiche

    - conversione diretta via scales/offsets
    - override LBM per pressione: p_scale = rho_scale * U_scale^2 (se scale diretto = 1)
    - override LBM per velocità: U_scale (se scale diretto = 1)
    """
    meas_lu = np.asarray(meas_lu, dtype=float)
    meas_phys = meas_lu * scales[:, None] + offsets[:, None]

    # scale base LBM
    rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
    U_scale, _   = lx_map.get("LatticeVelocity", (1.0, 0.0))
    p_scale = rho_scale * (U_scale ** 2)

    def _scale_direct_for(var_index: int) -> float:
        lx_name = var_lx[var_index] if var_index < len(var_lx) else ""
        return float(lx_map.get(lx_name, (1.0, 0.0))[0])

    # --- override velocità (se necessario) ---
    # aggiungi qui eventuali nomi alternativi che trovi nei tuoi file
    vel_names = [
        "x_velocity", "y_velocity", "z_velocity",
        "u", "v", "w",
        "velocity_x", "velocity_y", "velocity_z",
    ]
    for vn in vel_names:
        if vn in idx_map:
            i = idx_map[vn]
            scale_direct = _scale_direct_for(i)
            # se lo scale diretto è 1, probabilmente è rimasto in LU -> applica U_scale
            if abs(scale_direct - 1.0) < 1e-12:
                meas_phys[i, :] = meas_lu[i, :] * U_scale

    # --- override pressione (se necessario) ---
    if "static_pressure" in idx_map:
        i = idx_map["static_pressure"]
        scale_direct = _scale_direct_for(i)
        if abs(scale_direct - 1.0) < 1e-12:
            meas_phys[i, :] = meas_lu[i, :] * p_scale

    return meas_phys


def _coords_to_m(coords_lu, lx_map):
    L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
    return coords_lu * L_scale + L_off, (L_scale,)


# ----------------- plots (optional) -----------------

def quick_plots(out_dir, coords_m, meas_phys, idx_map):
    os.makedirs(out_dir, exist_ok=True)

    if "density" in idx_map:
        density = meas_phys[idx_map["density"], :]
        plt.figure()
        plt.hist(density, bins=60)
        plt.xlabel("density (physical units)")
        plt.ylabel("count")
        plt.title("Density distribution")
        p = os.path.join(out_dir, "density_hist.png")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        print("[PLOT]", p)

    v_names = ["x_velocity", "y_velocity", "z_velocity"]
    if all(n in idx_map for n in v_names):
        vx = meas_phys[idx_map["x_velocity"], :]
        vy = meas_phys[idx_map["y_velocity"], :]
        vz = meas_phys[idx_map["z_velocity"], :]
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        plt.figure()
        plt.hist(speed, bins=80)
        plt.xlabel("|v| (physical units)")
        plt.ylabel("count")
        plt.title("Speed magnitude distribution")
        p = os.path.join(out_dir, "speed_hist.png")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        print("[PLOT]", p)

    if "static_pressure" in idx_map:
        pfield = meas_phys[idx_map["static_pressure"], :]
        z = coords_m[:, 2]
        uniq = np.unique(np.round(z, decimals=12))
        z_plane = float(uniq[len(uniq)//2]) if len(uniq) else float(np.median(z))
        m = np.isclose(z, z_plane, rtol=0, atol=1e-12)

        if m.sum() < 10:
            z0 = np.median(z)
            order = np.argsort(np.abs(z - z0))
            m = np.zeros_like(z, dtype=bool)
            m[order[:max(1000, len(z)//20)]] = True
            z_plane = float(z0)

        xy = coords_m[m, :2]
        psl = pfield[m]

        plt.figure()
        plt.scatter(xy[:, 0], xy[:, 1], c=psl, s=3)
        plt.colorbar(label="static_pressure (physical units)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Static pressure slice near z={z_plane:.6g} m")
        p = os.path.join(out_dir, "pressure_slice.png")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        print("[PLOT]", p)


# ----------------- export .mat (time series + gridded) -----------------

def export_mat_timeseries(
    mat_path,
    coords_m,
    X_m, Y_m, Z_m,
    set_indices,
    var_names,
    gridded_data,   # array (Ny,Nx,Nt) OR (Ny,Nx,Nz,Nt)
    scales_dict,
    prefer_v73=True,
):
    out = {
        "coords_m": coords_m,
        "X_m": X_m,
        "Y_m": Y_m,
        "set_indices": np.array(set_indices, dtype=np.int32),
        "__var_names__": np.array(var_names, dtype=object),
        "__scales__": scales_dict,
    }
    if Z_m is not None:
        out["Z_m"] = Z_m

    # campi grigliati
    out.update(gridded_data)

    if prefer_v73:
        try:
            import hdf5storage
            hdf5storage.savemat(mat_path, out, format="7.3")
            print("[MAT] Scritto (v7.3):", mat_path)
            return
        except Exception as e:
            print("[WARN] Export v7.3 non riuscito. Ripiego su SciPy v5.")
            print("       Dettagli:", repr(e))

    from scipy.io import savemat
    savemat(mat_path, out, do_compression=True)
    print("[MAT] Scritto (v5):", mat_path)

# -------------------- finding the reference system -------------------

def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]

# --------------------Apply shift to coordinates ---------------------


def apply_T(points_xyz, T):
    P = np.c_[points_xyz, np.ones(points_xyz.shape[0])]
    Q = (P @ T.T)[:, :3]
    return Q
# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(
        description="PowerFLOW .fnc -> LU->fisico + grigliatura (x,y,z) + export time series in .mat"
    )
    ap.add_argument("input", help="Percorso file .fnc/.nc")
    ap.add_argument("--first", type=int, default=0, help="Primo snapshot/set (inclusivo). Default 0")
    ap.add_argument("--last", type=int, default=None, help="Ultimo snapshot/set (inclusivo). Default = first")
    ap.add_argument("--outdir", default="out_plots", help="Cartella output plot (usa solo sul primo set).")
    ap.add_argument("--mat", default="export_timeseries.mat", help="Nome file .mat da creare")
    ap.add_argument("--round-decimals", type=int, default=12, help="Snapping coordinate (default 12)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Tipo output per i campi grigliati")
    ap.add_argument("--no-plots", action="store_true", help="Disabilita i plot")
    ap.add_argument("--no-v73", action="store_true", help="Forza .mat v5 (non usare v7.3)")
    args = ap.parse_args()

    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    with Dataset(args.input, "r") as ds:
        var_short, var_lx, lx_map, scales, offsets, idx_map = _prepare_lattice_maps(ds)

        meas_var = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas_var.shape

        first = max(0, args.first)
        last = first if args.last is None else min(args.last,nsets-1)
        if last < first:
            raise ValueError(f"--last ({last}) deve essere >= --first ({first})")
        set_indices = list(range(first, last + 1))  # step = 1 come richiesto
        Nt = len(set_indices)

        print(f"[INFO] measurements shape = (nsets={nsets}, nvars={nvars}, npoints={npoints})")
        print(f"[INFO] Export set: {first}..{last} (Nt={Nt})")
        print("[INFO] Variabili:", var_short)

        # coord in metri + assi griglia
        coords_lu = _ensure_coords_nx3(ds.variables["vertex_coords"][:])
        names = decode_null_terminated(ds.variables["csys_names"])
        csys  = np.array(ds.variables["csys"][:], float)
        # scegli csys da nome
        idx = names.index("lattice_csys")     # esempio
        T = csys[idx]                      # (4,4) in lattice units (tipicamente)
        
        coords_lu = _ensure_coords_nx3(ds.variables["vertex_coords"][:])
        coords_lu2 = apply_T(coords_lu, T) # ancora in LU
        coords_m, (L_scale,) = _coords_to_m(coords_lu2, lx_map)

        X_m, Y_m, Z_m = _build_grid_axes(coords_m, decimals=args.round_decimals)
        xi, yi, zi = _point_indices_on_axes(coords_m, X_m, Y_m, Z_m, decimals=args.round_decimals)

        Ny, Nx = Y_m.size, X_m.size
        Nz = 1 if Z_m is None else Z_m.size
        print(f"[INFO] Grid dims: Ny={Ny}, Nx={Nx}, Nz={Nz}")

        # prealloc gridded output per variabile
        gridded = {}
        if Z_m is None:
            # 2D: (Ny, Nx, Nt)
            for name in var_short:
                gridded[name] = np.full((Ny, Nx, Nt), np.nan, dtype=out_dtype)
        else:
            # 3D: (Ny, Nx, Nz, Nt)  (ordine MATLAB-friendly)
            for name in var_short:
                gridded[name] = np.full((Ny, Nx, Nz, Nt), np.nan, dtype=out_dtype)

        # scale dict per matlab
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _   = lx_map.get("LatticeVelocity", (1.0, 0.0))
        scales_dict = {
            "LatticeLength_scale": float(L_scale),
            "LatticeDensity_scale": float(rho_scale),
            "LatticeVelocity_scale": float(U_scale),
            "pressure_scale_used_if_LBM": float(rho_scale * (U_scale**2)),
        }

        # loop su tutti i set richiesti
        for t, sidx in enumerate(set_indices):
            meas_lu = _as_numpy(meas_var[sidx, :, :]).astype(float)  # (nvars,npoints)
            meas_phys = _convert_one_set_to_physical(meas_lu, var_short, var_lx, lx_map, scales, offsets, idx_map)

            # griglia ogni variabile
            for name in var_short:
                vi = idx_map[name]
                grid = _grid_one_field(meas_phys[vi, :], xi, yi, zi, X_m, Y_m, Z_m)  # 2D o 3D(y,x,z)
                if Z_m is None:
                    gridded[name][:, :, t] = grid.astype(out_dtype)
                else:
                    gridded[name][:, :, :, t] = grid.astype(out_dtype)

            if t == 0 and not args.no_plots:
                # plot solo sul primo set esportato
                quick_plots(args.outdir, coords_m, meas_phys, idx_map)

            if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                print(f"[INFO] progress: {t+1}/{Nt}")

    # salva MAT
    export_mat_timeseries(
        mat_path=args.mat,
        coords_m=coords_m.astype(out_dtype),
        X_m=X_m.astype(out_dtype),
        Y_m=Y_m.astype(out_dtype),
        Z_m=None if Z_m is None else Z_m.astype(out_dtype),
        set_indices=set_indices,
        var_names=var_short,
        gridded_data=gridded,
        scales_dict=scales_dict,
        prefer_v73=(not args.no_v73),
    )


if __name__ == "__main__":
    import sys
    sys.argv = [
        sys.argv[0],
        "./fwh_prop.snc",
        "--first", "0",
        "--mat", "test.mat",
    ]
    main()
