# 01-create-fwi-geos
# Create initial zarr for GPM.LATE.v5
# Author: Katrina Sharonin

import re
import os
import glob

import xarray as xr
import numpy as np
import pandas as pd

import dask
import dask.array

from dask.diagnostics.progress import ProgressBar
from tqdm.auto import tqdm

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# absolute paths for data, zarr file, proc file
basedir = "/autofs/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5"
zar_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.zarr'
proc_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak.txt'

# check provided paths
if not os.path.isdir(zar_file):
    raise FileNotFoundError('Zarr path not found; check provided directory')
if not os.path.isdir(basedir):
    raise FileNotFoundError('Sipongi data input path invalid; check provided path')
if not os.path.isfile(proc_file):
    raise FileNotFoundError('Proc file not found; check provided path')

chunking = {
    "lat": 100,
    "lon": 100,
    "time": 100,
    "forecast": 9
}

# Keep all the dataset in the data directory
files = []
years = range(2017, 2023)
years = range(2014, 2015) # TODO - flagged, year creation limit

# Parse paths and add existing hourly files
for y in years:
    assert os.path.exists(f"{basedir}/{y}")
    files += sorted(glob.glob(f"{basedir}/{y}/FWI.GEOS-5.Daily.*.nc"))

assert len(files) != 0, "No files found, check provided year range"

d0 = xr.open_dataset(files[0])
dvars = list(d0.keys())
blank_array = dask.array.empty(
    (9, d0.lat.shape[0], d0.lon.shape[0]),
    dtype=np.float32
)
dims = ["forecast", "lat", "lon"]
blank_ds = xr.Dataset(
    data_vars = {k: (dims, blank_array) for k in dvars},
    coords = {
        "lat": d0.lat,
        "lon": d0.lon,
        "forecast": list(range(0, 9))
    }
)

def make_blank(pd_date):
    return (blank_ds.
            expand_dims({"time": [pd_date]}, axis=1).
            transpose("forecast", "time", "lat", "lon"))

dlist = []
for file in tqdm(files):
    # file = files[0]
    date_match = re.match(r".*\.(\d{8})\.nc", os.path.basename(file))
    assert date_match
    date = date_match.group(1)
    pd_date = pd.to_datetime(date)
    year = date[0:4]
    datedir = f"{basedir}/{year}/{date}00"
    if not os.path.exists(datedir):
        print(f"Not found: {datedir}")
        print("Skipping this timestep.")
        dlist.append(make_blank(pd_date))
        continue
    # Get list of forecast files.
    pred_files = sorted(glob.glob(f"{datedir}/*.nc"))   
    if not len(pred_files):
        print(f"No files found for date {date}. Skipping this timestep.")
        dlist.append(make_blank(pd_date))
        continue
    try:
        dat = xr.open_mfdataset([file] + pred_files, combine="nested", concat_dim="forecast")
        dnew = dat.assign_coords({
            "time": [pd_date],
            "forecast": range(0, 1+len(pred_files))
        })
        dlist.append(dnew)
    except Exception as err:
        print(f"Failed to open date {date} with error: `{str(err)}`")
        print("Skipping this timestep...")
        dlist.append(make_blank(pd_date))
        continue

dat = xr.concat(dlist, dim="time")

# Extend time series to complete chunk length
ntime = len(dat.time)
residual_chunks = chunking["time"] - (ntime % chunking["time"])

print("Extending time series for even chunking...")
dates = dat.time.values
date_max = dates.max()
newdates = pd.date_range(
    dates.max() + pd.Timedelta(1, 'D'),
    dates.max() + pd.Timedelta(residual_chunks, 'D')
)
assert len(newdates) == residual_chunks
alldates = np.concatenate((dates, newdates))
assert len(alldates) % chunking["time"] == 0
dat_extended = dat.reindex({"time": alldates}, fill_value = np.nan)
assert len(dat_extended.time) % chunking["time"] == 0
dat_extended = dat_extended.chunk(chunking)

print("Writing FWI GEOS-5 output zarr...")
with ProgressBar():
    dat.to_zarr(zar_file, mode="w")

# write to bak for tracking
with open(proc_file, "a") as f:
	for afile in files:
		f.write("\n" + afile)

# Test output
print("Testing Zarr: FWI GEOS-5")
dtest = xr.open_zarr(zar_file)
dtest_sub = dtest.sel(lat=39.74, lon=-104.9903, method="nearest")
dtest_sub.mean(["time", "forecast"])
print("Done!")
