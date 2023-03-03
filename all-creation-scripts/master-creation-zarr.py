# master-creation-update

# 1: FWI GEOS
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

zar_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.zarr'
proc_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak.txt' # TODO - generate procfile check functionality

if not os.path.isdir(zar_file):
	raise FileNotFoundError('Zarr directory does not exist; check provided path')
if not os.path.isfile(proc_file):
	raise FileNotFoundError('Proc file does not exist; check provided path')

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

basedir = "/autofs/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5"

# add all found files to bak -> see lower
assert os.path.exists(basedir)
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

with ProgressBar():
    dat.to_zarr(zar_file, mode="w")

# write to bak for tracking
with open(proc_file, "a") as f:
	for afile in files:
		f.write("\n" + afile)

# Test output
dtest = xr.open_zarr(zar_file)
dtest_sub = dtest.sel(lat=39.74, lon=-104.9903, method="nearest")
dtest_sub.mean(["time", "forecast"])
print ('End of FWI GEOS')

# ----------------------------------------
# 2: FWI GEOS Hourly

# Provide absolute path to bak + zarr
zar_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.Hourly.zarr'
procfile = '/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak-hourly.txt'

# Test if dir zar_file and procfile exist
if os.path.isdir(zar_file):
    pass
else:
    raise FileNotFoundError('Zarr directory does not exist. Provide a valid path or create a new directory')
if os.path.isfile(procfile):
    pass
else:
    raise FileNotFoundError('The processing .txt file does not exist. Provide a valid path or create a .txt file')

chunking = {
    "lat": 100,
    "lon": 100,
    "time": 120,
}

# Keep all the dataset in the data directory
files = []
years = range(2017, int(datetime.now().year) + 1)
years = range(2020, 2021)  # TODO - flagged, this is an artificial limit

# base directory
basedir = "/lovelace/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5/"

# Parse paths and add existing hourly files
assert os.path.exists(basedir)
for y in years:
    assert os.path.exists(f"{basedir}/{y}")
    files += sorted(glob.glob(f"{basedir}/{y}/FWI.GEOS-5.Hourly.*.nc"))

assert len(files) != 0, "No Files found, check provided year range"

d0 = xr.open_dataset(files[0])
dvars = list(d0.keys())
blank_array = dask.array.empty(
    (d0.time.shape[0], d0.lat.shape[0], d0.lon.shape[0]),
    dtype=np.float32
)

dims = ["time", "lat", "lon"]
blank_ds = xr.Dataset(
    data_vars={k: (dims, blank_array) for k in dvars},
    coords={
        "lat": d0.lat,
        "lon": d0.lon,
    }
)


def make_blank(pd_date):
    return (blank_ds.
            expand_dims({"time": [pd_date.replace(hour=n) for n in [*range(0, 24)]]}, axis=1).
            transpose("time", "lat", "lon"))


dlist = []
# MODIFIED ARR - slice for only two files
files = files[:2:]

# Sort files using annual directories
for file in tqdm(files):
    date_match = re.match(r".*\.(\d{8})\.nc", os.path.basename(file))
    assert date_match
    date = date_match.group(1)
    pd_date = pd.to_datetime(date)
    assert os.path.exists(file)

    try:
        dat = xr.open_mfdataset(file)
        hours = [*range(0, 24)]
        dnew = dat.assign_coords({
            "time": [pd_date.replace(hour=n) for n in hours],
        })

        dlist.append(dnew)
    except Exception as err:
        print(f"Failed to open date {date} with error: `{str(err)}`")
        print("Skipping this timestep... Trying blank insert")
        dlist.append(make_blank(pd_date))
        continue

# Concatenate all files along time dimension
dat = xr.concat(dlist, dim="time")

# Extend time series to complete chunk length
ntime = len(dat.time)
residual_chunks = (chunking["time"] - (ntime % chunking["time"]))

print("Extending time series for even chunking...")
dates = dat.time.values
date_max = dates.max()
# add +1 day to max date
range_start = dates.max() + pd.Timedelta(1, 'D')
range_start = range_start.replace(hour=0)
# add remaining days using residual / 24
residual_days = int(residual_chunks / 24)
range_end = dates.max() + pd.Timedelta(residual_days, 'D')
range_end = range_end.replace(hour=23)
newdates = pd.date_range(
    range_start,
    range_end,
    freq='1H'
)

assert len(newdates) == residual_chunks
alldates = np.concatenate((dates, newdates))
assert len(alldates) % chunking["time"] == 0
dat_extended = dat.reindex({"time": alldates}, fill_value=np.nan)
assert len(dat_extended.time) % chunking["time"] == 0
dat_extended = dat_extended.chunk(chunking)

# Write to file + progress bar
with ProgressBar():
    dat_extended.to_zarr(zar_file, mode="w")

# TODO - update bak to include all written file names
with open(procfile, "a") as f:
    for afile in files:
        f.write("\n" + afile)

# Test output
# See Katrina Sharonin's local testing module
print("Initiate testing: GEOS-5_FFMC Contents")
open_test = xr.open_zarr(zar_file)
open_test1 = open_test.isel(time=slice(0, 24))['GEOS-5_FFMC'].values
open_test1_count = np.count_nonzero(~np.isnan(open_test1))
print(open_test1_count)

print ('End of FWI GEOS Hourly')

# ------------------------------------------------------
# 3: FWI GPM LATE v5

basedir = "/lovelace/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GPM.LATE.v5/"
zar_file = "/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GPM.Late.zarr"
bak_file = "/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak-late.txt"

if not os.path.isdir(zar_file):
    raise FileNotFoundError('Zarr path not found; check provided directory')
if not os.path.isdir(basedir):
    raise FileNotFoundError('Sipongi data input path invalid; check provided path')
if not os.path.isfile(bak_file):
    raise FileNotFoundError('Bak file not found; check provided path')

files = []
years = range(2017, int(datetime.now().year) + 1)
years = range(2014, 2015)  # TODO - flagged, aritifical limit

for y in years:
    assert os.path.exists(f"{basedir}/{y}")

    # DEBUGGING - show contents
    # print(os.listdir(f"{basedir}/{y}"))
    # FWI.GPM.LATE.v5.Daily.Default.20221231.nc
    files += sorted(list(glob.glob(f"{basedir}/{y}/FWI.GPM.LATE.v5.Daily.Default.*.nc")))

if len(files) == 0:
    raise AttributeError('File search returned empty; check provided directories / year range')

chunking = {
    "time": 50,
    "lat": 100,
    "lon": 100
}


def parse_date(path):
    ms = re.search(r"\.(\d{4})(\d{2})(\d{2})\.nc$", path)
    year = int(ms.group(1))
    month = int(ms.group(2))
    day = int(ms.group(3))
    result_date = date(year, month, day)
    return result_date


print("Reading dataset...")
dat_raw = xr.open_mfdataset(files, combine="nested", concat_dim="time")
# The time coordinate here is wrong, so we reassign it by parsing the file name
print("Fixing time coordinate...")
dates = pd.to_datetime([parse_date(f) for f in files])
dat = dat_raw.assign_coords(time=dates)

ntime = len(dat.time)
residual_chunks = chunking["time"] - (ntime % chunking["time"])

print("Extending time series for even chunking...")
# dummy = xr.full_like(dat.isel(time = slice(residual_chunks)), np.nan)
# assert len(dummy.time) == residual_chunks
newdates = pd.date_range(
    dates.max() + pd.Timedelta(1, 'D'),
    dates.max() + pd.Timedelta(residual_chunks, 'D')
)
assert len(newdates) == residual_chunks
alldates = np.concatenate((dates, newdates))
assert len(alldates) % chunking["time"] == 0
dat_extended = dat.reindex({"time": alldates}, fill_value=np.nan)
assert len(dat_extended.time) % chunking["time"] == 0
dat_extended = dat_extended.chunk(chunking)

print("Writing output zarr...")
with ProgressBar():
    dat_extended.to_zarr(zar_file, mode="w")

# NOTE: bak syntax different from others... test appearance
with open(bak_file, "w") as f:
    f.writelines(file + "\n" for file in files)

print("Testing zarr")
dtest = xr.open_zarr(zar_file)
print("Done!")

print ('End of FWI GPM LATE v5')
