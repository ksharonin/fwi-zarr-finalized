# 01-create-fwi-gpm-late-v5
# Create initial zarr for GPM.LATE.v5
# Author: Katrina Sharonin

import glob
import datetime
import re
import os

import xarray as xr
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime

from dask.diagnostics.progress import ProgressBar

# absolute paths for data, zarr file, proc file
basedir = "/lovelace/brewer/rfield1/storage/observations/GFWED/Sipongi/fwiCalcs.GEOS-5/Default/GPM.LATE.v5/"
zar_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/FWI.GPM.Late.zarr'
proc_file = '/autofs/brewer/eisfire/katrina/update-zarr-utils/processed-files-bak-late.txt'

# check provided paths
if not os.path.isdir(zar_file):
	raise FileNotFoundError('Zarr path not found; check provided directory')
if not os.path.isdir(basedir):
	raise FileNotFoundError('Sipongi data input path invalid; check provided path')
if not os.path.isfile(proc_file):
	raise FileNotFoundError('Proc file not found; check provided path')

chunking = {
    "time": 50,
    "lat": 100,
    "lon": 100
}

files = []
years = range(2017, int(datetime.now().year) + 1)
years = range(2014,2015) # TODO - flagged, aritifical limit

# Parse paths and add existing hourly files
for y in years:
	assert os.path.exists(f"{basedir}/{y}")
	files += sorted(list(glob.glob(f"{basedir}/{y}/FWI.GPM.LATE.v5.Daily.Default.*.nc")))

assert len(files) != 0, "No files found, check provided year range"

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

print("Writing FWI GPM output zarr...")
with ProgressBar():
    dat_extended.to_zarr(zar_file, mode="w")

# NOTE: bak syntax different from others... test appearance
with open(proc_file, "w") as f:
    f.writelines(file + "\n" for file in files)

print("Testing Zarr: FWI GPM Late v5")
dtest = xr.open_zarr(zar_file)
print("Done!")
