# master-update-zarr

# 1: FWI GEOS

import datetime
import re
import glob
import sys
import dask
import os
from os.path import exists
from datetime import datetime
from tqdm.auto import tqdm
import netCDF4 as nc

import xarray as xr
import pandas as pd
import numpy as np

zarrpath = "/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.zarr"
procfile = "/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak.txt"
basedir = "/autofs/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5"

# check provided paths
if not os.path.isdir(zarrpath):
    raise FileNotFoundError('Zarr path not found; check provided directory')
if not os.path.isdir(basedir):
    raise FileNotFoundError('Sipongi data input path invalid; check provided path')
if not os.path.isfile(procfile):
    raise FileNotFoundError('Proc file not found; check provided path')

allfiles = []
years = range(2017, int(datetime.now().year) + 1)

for y in years:
    assert os.path.exists(f"{basedir}/{y}")
    allfiles += sorted(glob.glob(f"{basedir}/{y}/FWI.GEOS-5.Daily.*.nc"))
# allfiles = sorted(list(glob.glob(f"{basedir}/*/*.Daily.*.nc")))

# identify old vs new files
with open(procfile, "r") as f:
    # Remove empty lines
    procfiles = [l for l in f.read().splitlines() if l != '']
newfiles = sorted(set(allfiles) - set(procfiles))

# if no new files, exit program
if len(newfiles) == 0:
    sys.exit("No new files to process!")

def parse_date(path):
    ms = re.search(r"\.(\d{4})(\d{2})(\d{2})\.nc$", path)
    year = int(ms.group(1))
    month = int(ms.group(2))
    day = int(ms.group(3))
    date = datetime.date(year, month, day)
    return date

# given list of new files, create DataArray with new data
d0 = xr.open_dataset(newfiles[0]) # read-only access
dvars = list(d0.keys())
blank_array = dask.array.empty(
    (9, d0.lat.shape[0], d0.lon.shape[0]),
    dtype=np.float32
)
# include forecast dimension
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

for file in tqdm(newfiles):
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

# concatenate all new files via time dim
dat = xr.concat(dlist, dim="time")
dnew_all = dat

chunking = {
    "lat": 100,
    "lon": 100,
    "time": 100,
    "forecast": 9
}

# chunk data
dnew_all = dnew_all.chunk(chunking)

# now given "main" zarr, look for timestamps of the new files
# if timestamp appears once -> no new extension needed
# if timestamp not present -> extend to force presence
for idt, t in enumerate(dnew_all.time):

    # select single time dim
    dnew = dnew_all.isel(time=slice(idt, idt + 1))
    assert dnew_all.time[idt].values == dnew.time.values[0], "Mismatching enumeration"
    print(dnew.time.values[0])
    # check within current zarr if time exists
    dtarget = xr.open_zarr(zarrpath)
    is_in = dtarget.time.isin(dnew.time)
    n_is_in = is_in.sum()

    # there should only exist one unique time stamp
    if n_is_in > 1:
        raise Exception(f"Found {n_is_in} matching times! Something is wrong with this situation.")

    # timestamp exists -> no need for block extension
    elif n_is_in == 1:
        # identify location of the timestamp
        itime = int(np.where(is_in)[0])
        assert dtarget.isel(time=itime).time == dnew.time, "Mismatch in times"
        print(f"Inserting into time location {itime}")

        # slice and insert
        dnew.to_zarr(zarrpath, region={
            "time": slice(itime, itime + 1),
            "lat": slice(0, dnew.sizes["lat"]),
            "lon": slice(0, dnew.sizes["lon"]),
            # added slice - check if appropriately placed
            "forecast": slice(0, 9)
        })

    # timestamp DNE -> must extend our blocks to include time stamp
    elif n_is_in == 0:

        tchunk = dtarget.chunks["time"][-1]
        print(f"Creating new time chunk of size {tchunk}...")

        newdates = pd.date_range(
            dtarget.time.max().values + pd.Timedelta(1, 'D'),
            dtarget.time.max().values + pd.Timedelta(tchunk, 'D')
        )

        print(f"...from {newdates.min()} to {newdates.max()}.")
        assert len(newdates) == tchunk
        # Extend the current timestep out to size `tchunk`, filling with `Nan`s
        dummy = dnew.reindex({"time": newdates}, fill_value=np.nan)
        assert len(dummy.time) == tchunk
        # Now, append to the existing Zarr data store
        dummy.to_zarr(zarrpath, append_dim="time")


# Write to bak file with new files
with open(procfile, "a") as f:
    for afile in newfiles:
        f.write("\n" + afile)

print('Update FWI GEOS zarr process complete!')

# ---------------------------------------------------------------

# 2: FWI GEOS Hourly

import datetime
import re
import glob
import sys
from datetime import datetime
from os.path import exists

import dask
import dask.array

import os
from tqdm.auto import tqdm
import netCDF4 as nc

import xarray as xr
import pandas as pd
import numpy as np

zarrpath = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.Hourly.zarr'
procfile = "/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak-hourly.txt"
basedir = "/lovelace/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5/"

# check provided paths
if not os.path.isdir(zarrpath):
    raise FileNotFoundError('Zarr path not found; check provided directory')
if not os.path.isdir(basedir):
    raise FileNotFoundError('Sipongi data input path invalid; check provided path')
if not os.path.isfile(procfile):
    raise FileNotFoundError('Proc file not found; check provided path')


allfiles = []
years = range(2017, int(datetime.now().year) + 1)

for y in years:
    assert os.path.exists(f"{basedir}/{y}")
    allfiles += sorted(glob.glob(f"{basedir}/{y}/FWI.GEOS-5.Hourly.*.nc"))

# identify old vs new files
with open(procfile, "r") as f:
    procfiles = [l for l in f.read().splitlines() if l != '']
newfiles = sorted(set(allfiles) - set(procfiles))

# if no new files, exit program
if len(newfiles) == 0:
    sys.exit("No new files to process!")

# Function to extract date from file string
def parse_date(path):
    ms = re.search(r"\.(\d{4})(\d{2})(\d{2})\.nc$", path)
    year = int(ms.group(1))
    month = int(ms.group(2))
    day = int(ms.group(3))
    date = datetime.date(year, month, day)
    return date

# given list of new files, create DataArray with new data
d0 = xr.open_dataset(newfiles[0]) # read-only access
dvars = list(d0.keys())
blank_array = dask.array.empty(
    (d0.time.shape[0], d0.lat.shape[0], d0.lon.shape[0]),
    dtype=np.float32
)

dims = ["time", "lat", "lon"]
blank_ds = xr.Dataset(
    data_vars = {k: (dims, blank_array) for k in dvars},
    coords = {
        "lat": d0.lat,
        "lon": d0.lon,
    }
)

def make_blank(pd_date):
    return (blank_ds.
            expand_dims({"time": [pd_date.replace(hour=n) for n in [*range(0,24)]]}, axis=1).
            transpose("time", "lat", "lon"))


# base directory
dlist = []

for file in tqdm(newfiles):
    date_match = re.match(r".*\.(\d{8})\.nc", os.path.basename(file))
    assert date_match
    date = date_match.group(1)
    pd_date = pd.to_datetime(date)
    assert os.path.exists(file)

    try:
        dat = xr.open_mfdataset(file)
        hours = [*range(0,24)]
        dnew = dat.assign_coords({
            "time": [pd_date.replace(hour=n) for n in hours]
        })
        dlist.append(dnew)
    except Exception as err:
        print(f"Failed to open date {date} with error: `{str(err)}`")
        print("Skipping this timestep...Trying blank insert")
        dlist.append(make_blank(pd_date))
        continue

# Concatenate all new files via time dimension
dat = xr.concat(dlist, dim="time")
dnew_all = dat

chunking = {
    "lat": 100,
    "lon": 100,
    "time": 120
}

# chunk data
dnew_all = dnew_all.chunk(chunking)

# Given current zarr, look for timestamps of the new files
# if timestamp appears once -> no new extension needed
# if timestamp not present -> extend by chunk size to force presence
for idt, t in enumerate(dnew_all.time):

    # skip over any non-0 hour time, only iterate on every 24 hours of data
    if t.values not in dnew_all.time[0::24]:
        continue

    # select single time dim
    dnew = dnew_all.isel(time=slice(idt, idt + 24))
    assert dnew_all.time[idt].values == dnew.time.values[0], "Mismatching enumeration"
    print(dnew.time.values[0])
    # check within current zarr if time exists
    dtarget = xr.open_zarr(zarrpath)
    is_in = dtarget.time.isin(dnew.time[0])
    n_is_in = is_in.sum()

    if n_is_in > 1:
        raise Exception(f"Found {n_is_in} matching times! Something is wrong with this situation.")

    # timestamp with date exists -> no need for block extension
    elif n_is_in == 1:
        # identify location of the timestamp
        itime = int(np.where(is_in)[0])
        assert dtarget.isel(time=itime).time == dnew.time[0], "Mismatch in times"
        print(f"Inserting into time location {itime}")

        # slice and insert all hours
        dnew.to_zarr(zarrpath, region={
            "time": slice(itime, itime + 24),
            "lat": slice(0, dnew.sizes["lat"]),
            "lon": slice(0, dnew.sizes["lon"]),
        })

    # timestamp does NOT exist -> must extend our blocks to include time stamp
    elif n_is_in == 0:

        tchunk = dtarget.chunks["time"][-1]
        print(f"Creating new time chunk of size {tchunk}...")

        # add +1 day to max date
        range_start = dtarget.time.max().values + pd.Timedelta(1, 'D')
        range_start = range_start.replace(hour=0)
        # add remaining days using residual days / 24
        residual_days = int(tchunk/24)
        range_end = dtarget.time.max().values + pd.Timedelta(residual_days, 'D')
        range_end = range_end.replace(hour=23)

        newdates = pd.date_range(
            range_start,
            range_end,
            freq='1H'
        )

        print(f"...from {newdates.min()} to {newdates.max()}.")
        assert len(newdates) == tchunk
        # Extend the current timestep out to size `tchunk`, filling with `Nan`s
        dummy = dnew.reindex({"time": newdates}, fill_value=np.nan)
        assert len(dummy.time) == tchunk

        # Append to the existing Zarr data store
        dummy.to_zarr(zarrpath, append_dim="time")


# Write to bak file with new files
with open(procfile, "a") as f:
    for afile in newfiles:
        f.write("\n" + afile)
  
print('Update FWI GEOS Hourly zarr process complete!')

# ---------------------------------------------------------------

# 3: FWI GPM LATE v5

import datetime
import re
import glob
import sys
import os

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from os.path import exists


zarrpath = "/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GPM.Late.zarr"
procfile = "/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak-late.txt"
basedir = "/lovelace/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GPM.LATE.v5/"

# check provided paths
if not os.path.isdir(zarrpath):
    raise FileNotFoundError('Zarr path not found; check provided directory')
if not os.path.isdir(basedir):
    raise FileNotFoundError('Sipongi data input path invalid; check provided path')
if not os.path.isfile(procfile):
    raise FileNotFoundError('Proc file not found; check provided path')

allfiles = []
years = range(2017, int(datetime.now().year) + 1)

for y in years:
    assert os.path.exists(f"{basedir}/{y}")
    allfiles += sorted(glob.glob(f"{basedir}/{y}/FWI.GPM.LATE.v5.Daily.Default.*.nc"))
    
# allfiles = sorted(list(glob.glob(f"{basedir}/*/*.Daily.*.nc")))

# identify old vs new files
with open(procfile, "r") as f:
    # Remove empty lines
    procfiles = [l for l in f.read().splitlines() if l != '']
newfiles = sorted(set(allfiles) - set(procfiles))

# if no new files, exit program
if len(newfiles) == 0:
    sys.exit("No new files to process!")

def parse_date(path):
    ms = re.search(r"\.(\d{4})(\d{2})(\d{2})\.nc$", path)
    year = int(ms.group(1))
    month = int(ms.group(2))
    day = int(ms.group(3))
    date = datetime.date(year, month, day)
    return date

dnew_raw = xr.open_mfdataset(newfiles, combine="nested", concat_dim="time")
dates = pd.to_datetime([parse_date(f) for f in newfiles])
dnew_all = dnew_raw.assign_coords(time=dates)

# For now, do this individually for each timestep. It's very inefficient (lots
# of writes) but safe and conceptually simple, and shouldn't take too long for
# small numbers of files.
for idt, t in enumerate(dnew_all.time):
    # idt = 1; t = dnew_all.time[idt]
    dnew = dnew_all.isel(time=slice(idt, idt+1))
    print(dnew.time.values[0])
    # NOTE: Have to re-read `dtarget` each iteration because it may have been
    # expanded by the code below.
    dtarget = xr.open_zarr(zarrpath)
    is_in = dtarget.time.isin(dnew.time)
    n_is_in = is_in.sum()
    if n_is_in > 1:
        raise Exception(f"Found {n_is_in} matching times! Something is wrong with this situation.")
    elif n_is_in == 1:
        # Yes! Where exactly?
        itime = int(np.where(is_in)[0])
        assert dtarget.isel(time=itime).time == dnew.time, "Mismatch in times"
        print(f"Inserting into time location {itime}")
        # Write to that exact location (replace NaNs with new data)
        dnew.to_zarr(zarrpath, region={
            "time": slice(itime, itime+1),
            "lat": slice(0, dnew.sizes["lat"]),
            "lon": slice(0, dnew.sizes["lon"])
        })
    elif n_is_in == 0:
        # No, so we need to extend the time series.
        # For performance, we extend by one full time chunksize (`tchunk`)
        tchunk = dtarget.chunks["time"][-1]
        print(f"Creating new time chunk of size {tchunk}...")
        newdates = pd.date_range(
            dtarget.time.max().values + pd.Timedelta(1, 'D'),
            dtarget.time.max().values + pd.Timedelta(tchunk, 'D')
        )
        print(f"...from {newdates.min()} to {newdates.max()}.")
        assert len(newdates) == tchunk
        # Extend the current timestep out to size `tchunk`, filling with `Nan`s
        dummy = dnew.reindex({"time": newdates}, fill_value=np.nan)
        assert len(dummy.time) == tchunk
        # Now, append to the existing Zarr data store
        dummy.to_zarr(zarrpath, append_dim="time")

    # Write to bak file with new file
    ifile = newfiles[idt]
    with open(procfile, "a") as f:
        f.write("\n" + ifile)

# Test the result
print("Testing new Zarr...")
dtest = xr.open_zarr(zarrpath)
dtest.sel(time = slice("2022-06-01", "2022-06-15"))

print('Update FWI GPM Late v5 zarr process complete!')
