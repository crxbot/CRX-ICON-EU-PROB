import sys
import os
import cfgrib
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from zoneinfo import ZoneInfo
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------
# Eingabe/Ausgabe
# --------------------------
data_dir = sys.argv[1]        # z.B. "data/t2m"
output_dir = sys.argv[2]      # z.B. "output/temp0"
var_type = sys.argv[3]        # 'temp30','temp20','temp0','acc10mm'
gridfile = sys.argv[4] if len(sys.argv) > 4 else None

if gridfile is None or not os.path.exists(gridfile):
    raise FileNotFoundError("NetCDF Grid-Datei fehlt oder existiert nicht!")

os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Städte
# --------------------------
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden', 'Stuttgart', 'Düsseldorf',
             'Nürnberg', 'Erfurt', 'Leipzig', 'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

eu_cities = pd.DataFrame({
    'name': [
        'Berlin', 'Oslo', 'Warschau',
        'Lissabon', 'Madrid', 'Rom',
        'Ankara', 'Helsinki', 'Reykjavik',
        'London', 'Paris'
    ],
    'lat': [
        52.52, 59.91, 52.23,
        38.72, 40.42, 41.90,
        39.93, 60.17, 64.13,
        51.51, 48.85
    ],
    'lon': [
        13.40, 10.75, 21.01,
        -9.14, -3.70, 12.48,
        32.86, 24.94, -21.82,
        -0.13, 2.35
    ]
})

# --------------------------
# Farben / Colormap
# --------------------------
temp_bounds = [0,1,2,5,10,20,30,40,50,60,70,80,90,95,98,100]
temp_colors = ListedColormap([
    "#056B6F","#079833","#08B015","#40C50C","#7DD608",
    "#9BE105","#BBEA04","#DBF402","#FFF600","#FEDC00",
    "#FFAD00","#FF6300","#E50014","#BC0035","#930058","#660179"
])
temp_norm = BoundaryNorm(temp_bounds, temp_colors.N)


tp_bounds = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
tp_colors = ListedColormap([
    "#FE9226", "#FFC02B", "#FFEE32", "#DDE02D", "#BBD629",
    "#9AC925", "#79BC21", "#37A319", "#367C40",
    "#366754", "#4A3E7C", "#593192"
])

tp_norm = BoundaryNorm(tp_bounds, tp_colors.N)

wind_bounds = [0,1,2,5,10,20,30,40,50,60,70,80,90,95,98,100]
wind_colors = ListedColormap([
    "#056B6F","#079833","#08B015","#40C50C","#7DD608",
    "#9BE105","#BBEA04","#DBF402","#FFF600","#FEDC00",
    "#FFAD00","#FF6300","#E50014","#BC0035","#930058","#660179"
])
wind_norm = BoundaryNorm(wind_bounds, wind_colors.N)


# --------------------------
# Schwellenwerte
# --------------------------
thresholds = {"temp30":30,"temp20":20,"temp0":0,"temp30_eu":30,"temp20_eu":20,"temp0_eu":0,"tp10":10.0, "tp30":30.0, "tp100":100.0, "tp10_eu":10.0, "tp30_eu":30.0, "tp100_eu":100.0, "wind60":60.0, "wind90":90.0, "wind120":120.0, "wind60_eu":60.0, "wind90_eu":90.0, "wind120_eu":120.0}

# --------------------------
# Kartenparameter
# --------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX

extent = [5, 16, 47, 56]

extent_eu = [-23.5, 45.0, 29.5, 68.4]

# --------------------------
# NetCDF Grid laden (Zielraster)
# --------------------------
nc = netCDF4.Dataset(gridfile)
lon_grid = np.rad2deg(nc.variables["clon"][:])
lat_grid = np.rad2deg(nc.variables["clat"][:])
nc.close()

# Konvertiere masked arrays zu normalen arrays
if isinstance(lon_grid, np.ma.MaskedArray):
    lon_grid = lon_grid.filled(np.nan)
if isinstance(lat_grid, np.ma.MaskedArray):
    lat_grid = lat_grid.filled(np.nan)

# ICON-D2 ist ein unstrukturiertes Grid - wir erstellen ein reguläres Grid für Plotting
print(f"Grid hat {len(lon_grid)} Punkte")

# Erstelle reguläres Grid für Interpolation
if var_type in ["temp30_eu","temp20_eu","temp0_eu", "tp10_eu", "tp30_eu", "tp100_eu", "wind60_eu", "wind90_eu", "wind120_eu"]:
    grid_resolution = 0.13  # ~12km
    lon_min, lon_max, lat_min, lat_max = extent_eu
    buffer = grid_resolution * 20
    nx = int(round((lon_max - lon_min) / grid_resolution)) + 1
    ny = int(round((lat_max - lat_min) / grid_resolution)) + 1
    lon_reg = np.linspace(lon_min - buffer, lon_max + buffer, nx + 15)
    lat_reg = np.linspace(lat_min - buffer, lat_max + buffer, ny + 15)
    lon_grid2d, lat_grid2d = np.meshgrid(lon_reg, lat_reg)
else:
    grid_resolution = 0.025  # ~2,5km
    lon_min, lon_max, lat_min, lat_max = extent
    lon_reg = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
    lat_reg = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
    lon_grid2d, lat_grid2d = np.meshgrid(lon_reg, lat_reg)

acc_tp_prev = None  # für akkumulierten Niederschlag

# --------------------------
# GRIB Dateien durchgehen
# --------------------------
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".grib2"):
        continue

    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Variable auswählen
    # --------------------------
    if var_type in ["temp30","temp20","temp0","temp30_eu","temp20_eu","temp0_eu"]:
        if "t2m" not in ds:
            print(f"Keine t2m in {filename}")
            continue
        data = ds["t2m"].values - 273.15
    elif var_type in ["tp10", "tp30", "tp100", "tp10_eu", "tp30_eu", "tp100_eu"]:
    # Suche die Niederschlagsvariable
        if "tp" not in ds:
            print(f"Keine Niederschlagsvariable in {filename}")
            continue
        # Akkumulation direkt
        data = ds["tp"].values  # shape: (member, npoints)
    elif var_type in ["wind60", "wind90", "wind120", "wind60_eu", "wind90_eu", "wind120_eu"]:
        if "fg10" not in ds:
            print(f"Keine 10m Windkomponenten in {filename} ds.keys(): {ds.keys()}")
            continue
        data = ds["fg10"].values * 3.6  # m/s zu km/h

    else:
        print(f"Unbekannter var_type {var_type}")
        continue

    # --------------------------
    # Ensemble zu Wahrscheinlichkeit
    # --------------------------
    if var_type in ["temp0","temp0_eu"]:
    # Zähle, wie viele Member < 0°C sind, dann in Prozent
        data_prob = (data < thresholds[var_type]).sum(axis=0) / data.shape[0] * 100
    else:
        # Zähle, wie viele Member >= Schwelle sind, dann in Prozent
        data_prob = (data >= thresholds[var_type]).sum(axis=0) / data.shape[0] * 100

    # --------------------------
    # Interpolation auf reguläres Grid
    # --------------------------
    print("Interpoliere auf reguläres Grid...")
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    data_grid = griddata(points, data_prob.ravel(), (lon_grid2d, lat_grid2d), method='nearest')

    # --------------------------
    # Figure erstellen
    # --------------------------
    if var_type in ["temp30_eu","temp20_eu","temp0_eu", "tp10_eu", "tp30_eu", "tp100_eu", "wind60_eu", "wind90_eu", "wind120_eu"]:
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                        projection=ccrs.PlateCarree())
        ax.set_extent(extent_eu)
        ax.set_axis_off()
        ax.set_aspect('auto')
    else:
        scale = 0.9
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        shift_up = 0.02
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX/FIG_H_PX + shift_up, 1.0, TOP_AREA_PX/FIG_H_PX],
                        projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.set_axis_off()
        ax.set_aspect('auto')

    if var_type in ["temp30","temp20","temp0"]:
        data_smooth = gaussian_filter(data_grid, sigma=1.2)
        im = ax.pcolormesh(lon_grid2d, lat_grid2d, data_smooth,
                        cmap=temp_colors, norm=temp_norm, shading="auto")
    elif var_type in ["temp30_eu","temp20_eu","temp0_eu"]:
        im = ax.pcolormesh(lon_grid2d, lat_grid2d, data_grid,
                        cmap=temp_colors, norm=temp_norm, shading="auto")
    elif var_type in ["tp10", "tp30", "tp100"]:
        data_smooth = gaussian_filter(data_grid, sigma=1.2)
        im = ax.pcolormesh(lon_grid2d, lat_grid2d, data_smooth,
                        cmap=tp_colors, norm=tp_norm, shading="auto")
    elif var_type in ["tp10_eu", "tp30_eu", "tp100_eu"]:
        im = ax.pcolormesh(lon_grid2d, lat_grid2d, data_grid,
                        cmap=tp_colors, norm=tp_norm, shading="auto")
    elif var_type in ["wind60", "wind90", "wind120"]:
        data_smooth = gaussian_filter(data_grid, sigma=2.0)
        im = ax.pcolormesh(lon_grid2d, lat_grid2d, data_smooth,
                        cmap=wind_colors, norm=wind_norm, shading="auto")
    elif var_type in ["wind60_eu", "wind90_eu", "wind120_eu"]:
        im = ax.pcolormesh(lon_grid2d, lat_grid2d, data_grid,
                        cmap=wind_colors, norm=wind_norm, shading="auto")

    if var_type in ["temp30_eu","temp20_eu","temp0_eu", "tp10_eu", "tp30_eu", "tp100_eu", "wind60_eu", "wind90_eu", "wind120_eu"]:
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=0.7)

        for _, city in eu_cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6,
                    markerfacecolor="black", markeredgecolor="white", markeredgewidth=1.5)
            txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"], fontsize=9,
                        color="black", weight="bold")
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

        ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes,
                                        fill=False, color="black", linewidth=2))

    else:
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=0.7)

        for _, city in cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6,
                    markerfacecolor="black", markeredgecolor="white", markeredgewidth=1.5)
            txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"], fontsize=9,
                        color="black", weight="bold")
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

        ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes,
                                        fill=False, color="black", linewidth=2))

    # --------------------------
    # Colorbar
    # --------------------------
    legend_h_px = 50
    legend_bottom_px = 45
    if var_type in ["temp30","temp20","temp0", "temp30_eu", "temp20_eu", "temp0_eu", "tp10", "tp30", "tp100", "tp10_eu", "tp30_eu", "tp100_eu", "wind60", "wind90", "wind120", "wind60_eu", "wind90_eu", "wind120_eu"]:
        bounds = temp_bounds if var_type in ["temp30","temp20","temp0", "temp30_eu", "temp20_eu", "temp0_eu"] else tp_bounds if var_type in ["tp10", "tp30", "tp100", "tp10_eu", "tp30_eU", "tp100_eu"] else wind_bounds
        cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
        cbar.ax.tick_params(colors="black", labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

    # --------------------------
    # Footer
    # --------------------------
    # Laufzeit (Run) in UTC
    # Laufzeit (Run) in UTC
    if "time" in ds:
        time_val = ds["time"].values
        if np.isscalar(time_val):
            run_time_utc = pd.to_datetime(time_val)
        else:
            run_time_utc = pd.to_datetime(time_val[0])
    else:
        run_time_utc = None

    # Gültigkeitszeit
    if "valid_time" in ds:
        valid_time_val = ds["valid_time"].values
        if np.isscalar(valid_time_val):
            valid_time_utc = pd.to_datetime(valid_time_val)
        else:
            valid_time_utc = pd.to_datetime(valid_time_val[0])
    elif "step" in ds and run_time_utc is not None:
        step = pd.to_timedelta(ds["step"].values)
        if np.isscalar(step):
            valid_time_utc = run_time_utc + step
        else:
            valid_time_utc = run_time_utc + step[0]
    else:
        valid_time_utc = None

    # Lokale Zeitzone
    if valid_time_utc is not None:
        valid_time_local = valid_time_utc.tz_localize("UTC").tz_convert("Europe/Berlin")
    else:
        valid_time_local = None



    footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                                (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
    footer_ax.axis("off")

    footer_texts = {
        "temp30": "Temperatur >30°C (%)",
        "temp20": "Temperatur >20°C (%)",
        "temp0": "Temperatur <0°C (%)",
        "temp30_eu": "Temperatur >30°C (%), Europa",
        "temp20_eu": "Temperatur >20°C (%), Europa",
        "temp0_eu": "Temperatur <0°C (%), Europa",
        "tp10": "Akk. Niederschlag >10mm (%)",
        "tp20": "Akk. Niederschlag >20mm (%)",
        "tp100": "Akk. Niederschlag >100mm (%)",
        "tp10_eu": "Akk. Niederschlag >10mm (%), Europa",
        "tp20_eu": "Akk. Niederschlag >20mm (%), Europa",
        "tp100_eu": "Akk. Niederschlag >100mm (%), Europa",
        "wind60": "Windböen >60 km/h (%)",
        "wind90": "Windböen >90 km/h (%)",
        "wind120": "Windböen >120 km/h (%)",
        "wind60_eu": "Windböen >60 km/h (%), Europa",
        "wind90_eu": "Windböen >90 km/h (%), Europa",
        "wind120_eu": "Windböen >120 km/h (%), Europa",
    }
    left_text = footer_texts.get(var_type, var_type)
    if run_time_utc is not None:
        left_text += f"\nICON-EU ({run_time_utc.hour:02d}z), Deutscher Wetterdienst"

    footer_ax.text(0.01,0.85,left_text,fontsize=12,fontweight="bold",va="top",ha="left")
    footer_ax.text(0.734,0.92,"Prognose für:",fontsize=12,va="top",ha="left",fontweight="bold")
    footer_ax.text(0.99,0.68,f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
                   fontsize=12, va="top", ha="right", fontweight="bold")

    # --------------------------
    # Speichern
    # --------------------------
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.png"
    plt.savefig(os.path.join(output_dir, outname), dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()

    print(f"{filename} -> {outname} erstellt")
