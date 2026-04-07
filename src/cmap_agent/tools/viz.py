from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

def save_timeseries_png(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    out_png: str | None = None,
    out_path: str | None = None,
) -> str:
    """Save a time-series line plot.

    Improves readability by using an automatic date locator/formatter when the x
    axis is time-like, preventing overly-dense tick labels for long ranges.
    """

    # Backward/forward compatible output path:
    # - Newer callers pass out_png=...
    # - Older callers may pass out_path=...
    out = out_png or out_path
    if not out:
        raise ValueError('save_timeseries_png requires out_png or out_path')

    fig, ax = plt.subplots(figsize=(10, 4))

    x_vals = df[x]
    # If x looks like time, parse to datetime for nice tick formatting.
    x_dt = None
    try:
        if pd.api.types.is_datetime64_any_dtype(x_vals) or pd.api.types.is_datetime64tz_dtype(x_vals):
            x_dt = x_vals
        else:
            parsed = pd.to_datetime(x_vals, errors='coerce', utc=True)
            # Treat as datetime only if most values parsed successfully.
            if parsed.notna().mean() >= 0.8:
                x_dt = parsed
    except Exception:
        x_dt = None

    x_plot = x_dt if x_dt is not None else x_vals
    ax.plot(x_plot, df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)

    if x_dt is not None:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
        # Non-time x axis: avoid overcrowding by limiting tick count.
        try:
            ax.xaxis.set_major_locator(MaxNLocator(8))
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def save_scatter_map_png(df: pd.DataFrame, lat: str, lon: str, val: str, out_png: str, title: str | None = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(df[lon], df[lat], c=df[val], s=8)
    ax.set_xlabel(lon)
    ax.set_ylabel(lat)
    if title:
        ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=val)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_cartopy_map_png(
    df: pd.DataFrame,
    lat: str,
    lon: str,
    val: str,
    out_png: str,
    title: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    projection: str = "PlateCarree",
    central_longitude: float | None = None,
    central_latitude: float | None = None,
    method: str = "auto",
) -> bool:
    """Save a Cartopy map when possible.

    Returns True if Cartopy rendering succeeded; otherwise returns False and does not raise.

    Notes:
    - Many CMAP gridded datasets come back as (lat, lon, value) rows. If the points form a
      full rectilinear grid, we render with pcolormesh for a nicer cartographic plot.
    - If the requested lon range crosses the antimeridian (lon1 > lon2 in [-180, 180]), we
      unwrap longitudes to a continuous 0–360 range and use a Pacific-centered projection.
    """

    try:
        # Requires cartopy + proj stack
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        return False

    def _unwrap_lon_0_360(x: float) -> float:
        y = x % 360.0
        if y < 0:
            y += 360.0
        return y

    try:
        # Work on a clean copy of the columns we need
        d0 = df[[lat, lon, val]].copy()
        # Keep NaNs in the value column so gridded products with land/missing cells can still render as a raster.
        d0 = d0.dropna(subset=[lat, lon]).copy()
        d0[lon] = d0[lon].astype(float)
        d0[lat] = d0[lat].astype(float)
        try:
            d0[val] = pd.to_numeric(d0[val], errors='coerce')
        except Exception:
            pass

        def _norm_lon_180(x: float) -> float:
            # Normalize to (-180, 180]
            return ((x + 180.0) % 360.0) - 180.0

        def _make_projection(name: str, lon0: float, lat0: float):
            n = (name or "PlateCarree").strip().lower()
            if n == "platecarree":
                return ccrs.PlateCarree(central_longitude=lon0)
            if n == "robinson":
                return ccrs.Robinson(central_longitude=lon0)
            if n == "mollweide":
                return ccrs.Mollweide(central_longitude=lon0)
            if n == "mercator":
                return ccrs.Mercator(central_longitude=lon0)
            if n == "equalearth":
                return ccrs.EqualEarth(central_longitude=lon0)
            if n == "northpolarstereo":
                return ccrs.NorthPolarStereo(central_longitude=lon0)
            if n == "southpolarstereo":
                return ccrs.SouthPolarStereo(central_longitude=lon0)
            if n == "orthographic":
                return ccrs.Orthographic(central_longitude=lon0, central_latitude=lat0)
            # Fallback
            return ccrs.PlateCarree(central_longitude=lon0)

        # Decide whether we need antimeridian handling + pick an extent
        crosses_antimeridian = False
        extent = None
        lon_plot_col = lon

        # Auto centers derived from bbox (preferred) or left at defaults
        auto_lon0 = 0.0
        auto_lat0 = 0.0

        if bbox is not None:
            lat1, lat2, lon1, lon2 = map(float, bbox)
            crosses_antimeridian = (lon1 > lon2)

            auto_lat0 = (min(lat1, lat2) + max(lat1, lat2)) / 2.0
            if crosses_antimeridian:
                auto_lon0 = 180.0
                lon1_u = _unwrap_lon_0_360(lon1)
                lon2_u = _unwrap_lon_0_360(lon2)
                if lon2_u < lon1_u:
                    lon2_u += 360.0
                extent = [lon1_u, lon2_u, min(lat1, lat2), max(lat1, lat2)]
                d0["__lon_plot"] = d0[lon].astype(float)
                d0.loc[d0["__lon_plot"] < 0, "__lon_plot"] += 360.0
                lon_plot_col = "__lon_plot"
            else:
                extent = [min(lon1, lon2), max(lon1, lon2), min(lat1, lat2), max(lat1, lat2)]
                # For non-PlateCarree projections, centering on bbox is usually nicer.
                auto_lon0 = _norm_lon_180((min(lon1, lon2) + max(lon1, lon2)) / 2.0)

        # Determine final projection centers
        proj_name = (projection or "PlateCarree").strip()
        if central_longitude is not None:
            lon0 = float(central_longitude)
        else:
            if proj_name.strip().lower() == "platecarree":
                lon0 = 180.0 if crosses_antimeridian else 0.0
            else:
                lon0 = auto_lon0

        if central_latitude is not None:
            lat0 = float(central_latitude)
        else:
            lat0 = auto_lat0

        proj = _make_projection(proj_name, lon0, lat0)

        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=proj)

        # Cartography layers:
        # - OCEAN at the bottom
        # - LAND above ocean but **below data** so that datasets with valid values
        #   over land (e.g., atmospheric products) remain visible.
        # - Borders/coastlines above everything for context.
        try:
            ax.add_feature(cfeature.OCEAN, zorder=0)
        except Exception:
            log.exception("Cartopy: failed to add OCEAN feature; continuing without it.")
        try:
            # LAND below data so atmospheric fields over land remain visible.
            ax.add_feature(cfeature.LAND, zorder=0.6)
        except Exception:
            log.exception("Cartopy: failed to add LAND feature; continuing without it.")

        # Zoom to requested bbox (preferred) or data extents (fallback)
        if extent is None:
            lon_min, lon_max = float(d0[lon].min()), float(d0[lon].max())
            lat_min, lat_max = float(d0[lat].min()), float(d0[lat].max())
            if lon_min != lon_max and lat_min != lat_max:
                pad_lon = (lon_max - lon_min) * 0.03
                pad_lat = (lat_max - lat_min) * 0.03
                extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]
        if extent is not None:
            # Clamp extents to valid lon/lat ranges in the PlateCarree CRS to avoid projection-domain errors.
            try:
                if not crosses_antimeridian:
                    extent = [
                        max(-180.0, min(180.0, float(extent[0]))),
                        max(-180.0, min(180.0, float(extent[1]))),
                        max(-90.0, min(90.0, float(extent[2]))),
                        max(-90.0, min(90.0, float(extent[3]))),
                    ]
                    # Ensure min < max after clamping
                    if extent[0] >= extent[1] or extent[2] >= extent[3]:
                        extent = None
                if extent is not None:
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                else:
                    ax.set_global()
            except Exception:
                # If set_extent fails for a projection, fall back to global view.
                log.exception("Cartopy: set_extent failed; falling back to set_global().")
                try:
                    ax.set_global()
                except Exception:
                    pass


        # Detect rectilinear grid (preferred). Many CMAP gridded products include explicit rows for missing cells
        # (e.g., over land) with NaN values; we still want a raster-like plot in that case.
        def _edges_from_centers(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.size == 0:
                return x
            if x.size == 1:
                d = 1.0
                return np.array([x[0] - d / 2.0, x[0] + d / 2.0], dtype=float)
            dx = np.diff(x)
            mid = x[:-1] + dx / 2.0
            edges = np.empty(x.size + 1, dtype=float)
            edges[1:-1] = mid
            edges[0] = x[0] - dx[0] / 2.0
            edges[-1] = x[-1] + dx[-1] / 2.0
            return edges

        mappable = None
        method_norm = (method or "auto").strip().lower()
        try:
            pivot = (
                d0.pivot_table(index=lat, columns=lon_plot_col, values=val, aggfunc="mean")
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
                grid_lats = pivot.index.to_numpy(dtype=float)
                grid_lons = pivot.columns.to_numpy(dtype=float)
                z = pivot.to_numpy(dtype=float)

                # Ensure monotonic longitudes for pcolormesh (prevents wrap artifacts)
                order = np.argsort(grid_lons)
                grid_lons = grid_lons[order]
                z = z[:, order]

                # Choose rendering method.
                n_cells = int(z.shape[0] * z.shape[1])

                if method_norm in {"scatter", "points"}:
                    mappable = None
                else:
                    # Auto: prefer contourf for smaller grids (looks smoother than pixel blocks).
                    use_contour = (
                        method_norm in {"contour", "contourf"}
                        or (method_norm == "auto" and n_cells <= 40000)
                    )

                    z_masked = np.ma.masked_invalid(z)

                    if use_contour:
                        # Filled contours for a smooth-looking climatology-style map.
                        X, Y = np.meshgrid(grid_lons, grid_lats)
                        mappable = ax.contourf(
                            X,
                            Y,
                            z_masked,
                            levels=30,
                            transform=ccrs.PlateCarree(),
                            zorder=1,
                        )
                    else:
                        lon_edges = _edges_from_centers(grid_lons)
                        lat_edges = _edges_from_centers(grid_lats)
                        mappable = ax.pcolormesh(
                            lon_edges,
                            lat_edges,
                            z_masked,
                            transform=ccrs.PlateCarree(),
                            shading="auto",
                            zorder=1,
                            rasterized=True,
                            antialiased=False,
                        )
        except Exception:
            mappable = None

        if mappable is None:
            # Scatter fallback (still respects lon unwrapping when provided)
            d_sc = d0.dropna(subset=[val])
            x = d_sc[lon_plot_col] if lon_plot_col in d_sc.columns else d_sc[lon]
            mappable = ax.scatter(x, d_sc[lat], c=d_sc[val], s=8, transform=ccrs.PlateCarree(), zorder=1)

        # Outlines above the data for context.
        try:
            ax.coastlines(resolution="110m", linewidth=0.6, zorder=3)
        except Exception:
            log.exception("Cartopy: failed to draw coastlines; continuing without them.")
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=3)
        except Exception:
            log.exception("Cartopy: failed to draw borders; continuing without them.")

        if title:
            ax.set_title(title)
        # Draw labeled lat/lon ticks (when supported by the projection)
        try:
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.3,
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}
        except Exception:
            gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
        _ = gl  # silence lint
        plt.colorbar(mappable, ax=ax, label=val, shrink=0.8, pad=0.02)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False
