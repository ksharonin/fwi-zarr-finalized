# 02-update-fwi-gpm-late-v5
# Update current zarr for GPM.LATE.v5
# Author: Katrina Sharonin

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
