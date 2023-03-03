# master-update-zarr

# 1: FWI GEOS
# non-localized version of update zarr for forecast

import datetime
import re
import glob
import sys
import dask
import os
from tqdm.auto import tqdm
import netCDF4 as nc

import xarray as xr
import pandas as pd
import numpy as np

timevar = "time"
zarrpath = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.zarr'

if not os.path.isdir(zarrpath):
	raise FileNotFoundError('Zarr file does not exist; check provided path')

procfile = "/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak.txt"

if not os.path.isfile(procfile):
	raise FileNotFoundError('Proc file txt does not exist; check provided path')

allfiles = sorted(list(glob.glob("raw-input/*/*.Daily.*.nc")))  # flag - unsure if this would be different
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
basedir = "/autofs/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5" # flag - unsure if this would be different

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

print('Update zarr process complete!')
print ('End of FWI GEOS')
# for testing, see katrina's local file set up

# ---------------------------------------------------------------

# 2: FWI GEOS Hourly

timevar = "time"
zarrpath = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GEOS-5.Hourly.zarr'
procfile = "/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak-hourly.txt" # flag - may be different

# Test if zar_file and procfile exist
if os.path.isdir(zarrpath):
	pass
else:
	raise FileNotFoundError('Zarr directory does not exist. Provie a valid path or create a new dir')
if os.path.isfile(procfile):
	pass
else:
	raise FileNotFoundError('The processing .txt. file does not exist. Provide a valid path or create a new .txt')

allfiles = []
basedir = "/lovelace/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GEOS-5/"
assert os.path.exists(basedir)
years = range(2017, int(datetime.now().year) + 1)
years = range(2022, 2024) # TODO - Artificial Limit on new files, comment out

for y in years:
    assert os.path.exists(f"{basedir}/{y}")
    allfiles += sorted(glob.glob(f"{basedir}/{y}/FWI.GEOS-5.Hourly.*.nc"))

# Open txt file to id current files in zarr
with open(procfile, "r") as f:
    procfiles = [l for l in f.read().splitlines() if l != '']
newfiles = sorted(set(allfiles) - set(procfiles))

# Terminate program if no new files
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

# TODO MODIFIED: list cut down
newfiles = newfiles[:6:]

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

print('Update zarr process complete!')

# Write to bak file with new files
with open(procfile, "a") as f:
	for afile in newfiles:
		f.write("\n" + afile)
print ('End of FWI GEOS Hourly')

# ---------------------------------------------------------------

# 3: FWI GPM LATE v5

timevar = "time"
zarrpath = "FWI_GPM_LATE_v5_Daily.zarr"

procfile = "processed-files-bak.txt"
allfiles = sorted(list(glob.glob("raw-input/*/*.Daily.*.nc")))
with open(procfile, "r") as f:
    # Remove empty lines
    procfiles = [l for l in f.read().splitlines() if l != '']
newfiles = sorted(set(allfiles) - set(procfiles))

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
# TODO The much faster way to do this is to split up `dnew_all` into existing
# vs. new chunks and then do those separately.
for idt, t in enumerate(dnew_all.time):
    # idt = 1; t = dnew_all.time[idt]
    dnew = dnew_all.isel(time=slice(idt, idt+1))
    print(dnew.time.values[0])
    # Is the current timestamp in the Zarr file?
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

    ifile = newfiles[idt]
    with open(procfile, "a") as f:
        # Note: Flipping the position of the \n means this adds a blank line
        # between "groups" of processed files. That's a minor feature -- it lets us
        # see when groups of files have been processed.
        f.write("\n" + ifile)

# Test the result
print("Testing new GPM Late Zarr...")
dtest = xr.open_zarr(zarrpath)
dtest.sel(time = slice("2022-06-01", "2022-06-15"))

print("End of FWI GPM Late v5")