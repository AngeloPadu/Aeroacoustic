#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from netCDF4 import Dataset
import h5py


def decode_null_terminated(var) -> list[str]:
    b = var[:].tobytes()
    return [p for p in b.decode("ascii", errors="ignore").split("\x00") if p]


def apply_csys(points_xyz, T4x4, invert=False):
    """Apply homogeneous 4x4 transform to points (N,3)."""
    T = np.array(T4x4, dtype=np.float64)
    if invert:
        T = np.linalg.inv(T)
    P = np.c_[points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)]
    Q = (P @ T.T)[:, :3]
    return Q


def surfel_centroids_from_vertices(V, first, vrefs):
    pts = V[vrefs]  # (nvertex_refs,3)
    sums = np.add.reduceat(pts, first, axis=0)
    counts = np.diff(np.r_[first, len(vrefs)]).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("first_vertex_refs not coherent: found surfels with <=0 vertices")
    return sums / counts[:, None], counts


def fan_triangulate_polygons(first, vrefs):
    """
    Triangulates polygonal surfels with fan triangulation.
    Returns:
      F (Ntri,3) 0-based
      tri_owner (Ntri,) surfel index for each triangle
    """
    first = np.asarray(first, dtype=np.int64).ravel()
    vrefs = np.asarray(vrefs, dtype=np.int64).ravel()

    nfaces = first.size
    tris = []
    owner = []

    for i in range(nfaces):
        a = int(first[i])
        b = int(first[i + 1]) if i + 1 < nfaces else len(vrefs)
        poly = vrefs[a:b]
        nv = poly.size
        if nv < 3:
            continue
        p0 = int(poly[0])
        for j in range(1, nv - 1):
            tris.append((p0, int(poly[j]), int(poly[j + 1])))
            owner.append(i)

    F = np.array(tris, dtype=np.int64)
    tri_owner = np.array(owner, dtype=np.int64)
    return F, tri_owner


def _write_string_array(h5, name, strings):
    """Store strings in an HDF5 dataset in a MATLAB-friendly way."""
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _parse_vars_list(vars_in):
    """
    Accept:
      --vars static_pressure x_velocity
      --vars static_pressure,x_velocity
      --vars all
    """
    if len(vars_in) == 1 and isinstance(vars_in[0], str) and "," in vars_in[0]:
        vars_in = [v.strip() for v in vars_in[0].split(",") if v.strip()]
    return vars_in


def convert_field(var_name: str, field_lu: np.ndarray, p_scale: float, U_scale: float) -> np.ndarray:
    """
    Keep the SAME conversion logic you had:
      - static_pressure -> * p_scale
      - x/y/z_velocity  -> * U_scale
      - else -> no scaling
    Extend here if needed for other variables.
    """
    if var_name == "static_pressure":
        return field_lu * p_scale
    if var_name in ("x_velocity", "y_velocity", "z_velocity"):
        return field_lu * U_scale
    return field_lu


def main():
    ap = argparse.ArgumentParser(
        description="Export .snc surface -> .mat (HDF5/v7.3-style) exporting MULTIPLE variables, streaming frames"
    )
    ap.add_argument("input", help="input .snc file")
    ap.add_argument("--mat", default="export_multi_v73.mat", help="output .mat (HDF5)")
    ap.add_argument("--csys", default="base_frame",
                    help="csys: base_frame | lattice_csys | default_csys | duct_csys")
    ap.add_argument("--invert-csys", action="store_true", help="invert csys matrix")

    # MULTI VARS
    ap.add_argument("--vars", nargs="+", default=["static_pressure"],
                    help="variables to export (short names). Example: --vars static_pressure x_velocity y_velocity  OR  --vars all")

    # sets / frames
    ap.add_argument("--set", type=int, default=0, help="single set (default 0) if not exporting all")
    ap.add_argument("--all-sets", action="store_true", help="export all frames/sets")
    ap.add_argument("--first", type=int, default=None, help="first set index (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last set index (inclusive)")

    # memory / output controls
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="dtype for time-varying fields (default float32)")
    ap.add_argument("--export-C", action="store_true",
                    help="ALSO export triangle colors C per variable (nTri,Nt). HUGE! Not recommended.")
    ap.add_argument("--chunk-frames", type=int, default=8,
                    help="HDF5 chunk size along time dimension (default 8; larger can speed writes).")
    ap.add_argument("--point-chunk", type=int, default=200_000,
                    help="HDF5 chunk size along points dimension (default 200k).")

    args = ap.parse_args()
    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    vars_req = _parse_vars_list(args.vars)

    with Dataset(args.input, "r") as ds:
        # ---- csys ----
        csys_names = decode_null_terminated(ds.variables["csys_names"])
        csys = np.array(ds.variables["csys"][:], dtype=np.float64)
        base_frame = int(np.array(ds.variables["base_frame"][:]).ravel()[0])

        if args.csys == "base_frame":
            T = csys[base_frame]
            csys_used = f"base_frame:{csys_names[base_frame] if base_frame < len(csys_names) else base_frame}"
        else:
            if args.csys not in csys_names:
                raise ValueError(
                    f"csys '{args.csys}' not found. Available: {csys_names} "
                    f"(base_frame={csys_names[base_frame]})"
                )
            T = csys[csys_names.index(args.csys)]
            csys_used = args.csys

        # ---- lattice scales ----
        lx_names = decode_null_terminated(ds.variables["lx_names"])
        lx_scales = np.array(ds.variables["lx_scales"][:], dtype=np.float64)
        lx_offsets = np.array(ds.variables["lx_offsets"][:], dtype=np.float64)
        lx_map = {n: (s, o) for n, s, o in zip(lx_names, lx_scales, lx_offsets)}

        L_scale, L_off = lx_map.get("LatticeLength", (1.0, 0.0))
        rho_scale, _ = lx_map.get("LatticeDensity", (1.0, 0.0))
        U_scale, _ = lx_map.get("LatticeVelocity", (1.0, 0.0))
        p_scale = rho_scale * (U_scale ** 2)

        # ---- mesh ----
        V_lu = np.array(ds.variables["vertex_coords"][:], dtype=np.float64)          # (nV,3)
        first = np.array(ds.variables["first_vertex_refs"][:], dtype=np.int64)      # (npoints,)
        vrefs = np.array(ds.variables["vertex_refs"][:], dtype=np.int64)            # (nvertex_refs,)

        # Apply csys to vertices in lattice units
        V_lu_t = apply_csys(V_lu, T, invert=args.invert_csys)

        # Convert to meters
        V_m = (V_lu_t * L_scale + L_off).astype(np.float32)  # static -> float32 ok

        # Surfel centroids (meters) for debug
        cent_lu, _counts = surfel_centroids_from_vertices(V_lu_t, first, vrefs)
        cent_m = (cent_lu * L_scale + L_off).astype(np.float32)

        # ---- variables ----
        var_short = decode_null_terminated(ds.variables["variable_short_names"])
        meas = ds.variables["measurements"]  # (nsets,nvars,npoints)
        nsets, nvars, npoints = meas.shape

        # Resolve requested variables
        if len(vars_req) == 1 and vars_req[0].lower() == "all":
            vars_out = list(var_short)
        else:
            missing = [v for v in vars_req if v not in var_short]
            if missing:
                raise ValueError(f"Variables not found: {missing}\nAvailable: {var_short}")
            vars_out = list(vars_req)

        vi_list = [var_short.index(v) for v in vars_out]
        nvars_out = len(vars_out)

        # Decide which sets to export
        if args.all_sets or args.first is not None or args.last is not None:
            first_set = 0 if args.first is None else max(0, args.first)
            last_set = (nsets - 1) if args.last is None else min(nsets - 1, args.last)
            if last_set < first_set:
                raise ValueError(f"Invalid range: first={first_set}, last={last_set}")
            set_indices = list(range(first_set, last_set + 1))
        else:
            if args.set < 0 or args.set >= nsets:
                raise ValueError(f"--set {args.set} out of range (0..{nsets-1})")
            set_indices = [args.set]

        Nt = len(set_indices)

        # ---- triangulation surfel (fan) ----
        F0, tri_owner = fan_triangulate_polygons(first, vrefs)    # 0-based
        F = (F0 + 1).astype(np.int32)                              # 1-based for MATLAB
        tri_owner = tri_owner.astype(np.int32)                     # 0-based surfel index
        ntri = F.shape[0]

        # ---- Create HDF5 (.mat v7.3-style) and stream-write ----
        with h5py.File(args.mat, "w") as h5:
            # Static datasets
            h5.create_dataset("V_m", data=V_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("F", data=F, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("tri_owner", data=tri_owner, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("centroids_m", data=cent_m, compression="gzip", compression_opts=4, shuffle=True)
            h5.create_dataset("set_indices", data=np.array(set_indices, dtype=np.int32))

            _write_string_array(h5, "__var_names__", vars_out)
            _write_string_array(h5, "csys_names", csys_names)
            _write_string_array(h5, "lx_names", lx_names)

            # Metadata group
            meta = h5.create_group("meta")
            meta.attrs["csys_used"] = csys_used
            meta.attrs["invert_csys"] = bool(args.invert_csys)
            meta.attrs["L_scale"] = float(L_scale)
            meta.attrs["L_off"] = float(L_off)
            meta.attrs["rho_scale"] = float(rho_scale)
            meta.attrs["U_scale"] = float(U_scale)
            meta.attrs["p_scale"] = float(p_scale)
            meta.attrs["npoints"] = int(npoints)
            meta.attrs["ntri"] = int(ntri)
            meta.attrs["nsets_total"] = int(nsets)

            # Groups for multi-variable outputs
            g_fields = h5.create_group("fields")
            g_C = h5.create_group("C") if args.export_C else None

            # HDF5 chunking
            chunk_t = max(1, int(args.chunk_frames))
            chunk_p = max(10_000, int(args.point_chunk))
            chunk_p = min(chunk_p, npoints)

            # Create datasets for each variable
            field_dsets = {}
            C_dsets = {}

            for vname in vars_out:
                field_dsets[vname] = g_fields.create_dataset(
                    vname,
                    shape=(npoints, Nt),
                    dtype=out_dtype,
                    chunks=(chunk_p, min(Nt, chunk_t)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                if args.export_C:
                    C_dsets[vname] = g_C.create_dataset(
                        vname,
                        shape=(ntri, Nt),
                        dtype=out_dtype,
                        chunks=(min(ntri, chunk_p), min(Nt, chunk_t)),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

            # --- streaming over time (sets) ---
            # Read all requested vars for each set in one shot: (nvars_out, npoints)
            for t, sidx in enumerate(set_indices):
                block = np.array(meas[sidx, vi_list, :], dtype=np.float64)  # (nvars_out, npoints)

                for k, vname in enumerate(vars_out):
                    field_lu = block[k, :]
                    field = convert_field(vname, field_lu, p_scale=p_scale, U_scale=U_scale)
                    field = field.astype(out_dtype, copy=False)

                    field_dsets[vname][:, t] = field
                    if args.export_C:
                        C_dsets[vname][:, t] = field[tri_owner]

                if (t + 1) % max(1, Nt // 10) == 0 or (t + 1) == Nt:
                    print(f"[INFO] progress {t+1}/{Nt}")

    print(f"✅ wrote {args.mat}")
    print(f"   exported vars: {vars_out}")
    print(f"   V_m: {V_m.shape}  F: {F.shape}  tri_owner: {tri_owner.shape}")
    for vname in vars_out:
        print(f"   /fields/{vname}: ({npoints}, {Nt}) dtype={args.dtype}")
    if args.export_C:
        for vname in vars_out:
            print(f"   /C/{vname}: ({ntri}, {Nt}) dtype={args.dtype}  (WARNING: huge)")


if __name__ == "__main__":
    main()
