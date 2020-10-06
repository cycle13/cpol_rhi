"""
Raw radar RHIs processing.

@title: calibrate_rhi
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institution: Australian Bureau of Meteorology
@date: 06/10/2020
@version: 0.1

.. autosummary::
    :toctree: generated/

    mkdir
    remove
    extract_zip
    get_radar_archive_file
    get_metadata
    get_calib_offset
    get_zdr_offset
    get_dbz_name
    create_level1a
    buffer
    main
"""
import gc
import os
import sys
import glob
import gzip
import uuid
import pickle
import zipfile
import argparse
import datetime
import warnings
import traceback

import pyart
import cftime
import crayons
import numpy as np
import pandas as pd
import xarray as xr
import dask.bag as db

import cpol_processing


def mkdir(path: str):
    """
    Create the DIRECTORY(ies), if they do not already exist
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    return None


def remove(flist):
    """
    Remove file if it exists.
    """
    flist = [f for f in flist if f is not None]
    for f in flist:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return None


def extract_zip(inzip: str, path: str):
    """
    Extract content of a zipfile inside a given directory.

    Parameters:
    ===========
    inzip: str
        Input zip file.
    path: str
        Output path.

    Returns:
    ========
    namelist: List
        List of files extracted from  the zip.
    """
    with zipfile.ZipFile(inzip) as zid:
        zid.extractall(path=path)
        namelist = [os.path.join(path, f) for f in zid.namelist()]
    return namelist


def get_radar_archive_file(date) -> str:
    """
    Return the archive containing the radar file for a given date.

    Parameters:
    ===========
    date: datetime
        Date.

    Returns:
    ========
    file: str
        Radar archive if it exists at the given date.
    """
    datestr = date.strftime("%Y%m%d")
    file = os.path.join(INPATH, f"{date.year}", f"{datestr}.zip")

    if not os.path.exists(file):
        return None

    return file


def get_metadata(radar):
    # Lat/lon informations
    today = datetime.datetime.utcnow()
    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    radar_end_date = cftime.num2pydate(radar.time["data"][-1], radar.time["units"])

    latitude = radar.gate_latitude["data"]
    longitude = radar.gate_longitude["data"]
    maxlon = longitude.max()
    minlon = longitude.min()
    maxlat = latitude.max()
    minlat = latitude.min()
    origin_altitude = "50"
    origin_latitude = "-12.2491"
    origin_longitude = "131.0444"

    unique_id = str(uuid.uuid4())
    metadata = {
        "Conventions": "CF-1.6, ACDD-1.3",
        "acknowledgement": "This work has been supported by the U.S. Department of Energy Atmospheric Systems Research Program through the grant DE-SC0014063. Data may be freely distributed.",
        "country": "Australia",
        "creator_email": "valentin.louf@bom.gov.au",
        "creator_name": "Valentin Louf",
        "creator_url": "github.com/vlouf",
        "date_created": today.isoformat(),
        "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
        "geospatial_lat_max": f"{maxlat:0.6}",
        "geospatial_lat_min": f"{minlat:0.6}",
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_max": f"{maxlon:0.6}",
        "geospatial_lon_min": f"{minlon:0.6}",
        "geospatial_lon_units": "degrees_east",
        "history": "created by Valentin Louf on gadi.nci.org.au at " + today.isoformat() + " using Py-ART",
        "id": unique_id,
        "institution": "Bureau of Meteorology",
        "instrument": "radar",
        "instrument_name": "CPOL",
        "instrument_type": "radar",
        "keywords": "radar, tropics, Doppler, dual-polarization",
        "license": "CC BY-NC-SA 4.0",
        "naming_authority": "au.org.nci",
        "origin_altitude": origin_altitude,
        "origin_latitude": origin_latitude,
        "origin_longitude": origin_longitude,
        "platform_is_mobile": "false",
        "processing_level": "b1",
        "project": "CPOL",
        "publisher_name": "NCI",
        "publisher_url": "nci.gov.au",
        "product_version": f"v{today.year}.{today.month:02}",
        "references": "doi:10.1175/JTECH-D-18-0007.1",
        "site_name": "Gunn Pt",
        "source": "radar",
        "state": "NT",
        "standard_name_vocabulary": "CF Standard Name Table v71",
        "summary": "RHI scan from CPOL dual-polarization Doppler radar (Darwin, Australia)",
        "time_coverage_start": radar_start_date.isoformat(),
        "time_coverage_end": radar_end_date.isoformat(),
        "time_coverage_duration": "P10M",
        "time_coverage_resolution": "PT10M",
        "title": "radar RHI volume from CPOL",
        "uuid": unique_id,
        "version": radar.metadata["version"],
    }
    return metadata


def get_calib_offset(mydate) -> float:
    """
    Get calibration offset for given date.

    Parameter:
    ==========
    mydate: datetime
        Date of treatment.

    Returns:
    ========
    calib_offset: float
        Calibration offset value. Z_calib = Z_cpol + calib_offset.
    """
    calib_offset = None
    if IS_CALIB_PERIOD:
        for datest, dateed, rval in zip(CALIB_DATE_START, CALIB_DATE_END, CALIB_VALUE):
            if (mydate >= datest) & (mydate <= dateed):
                calib_offset = rval

        # If no calibration offset has been found, then looking for the closest one.
        if calib_offset is None:
            daydelta = np.array([(cd - mydate).days for cd in CALIB_DATE_START])
            pos = np.argmax(daydelta[daydelta < 0])
            calib_offset = CALIB_VALUE[pos]
    else:
        daydelta = np.array([mydate], dtype=np.datetime64)
        pos = np.argmin(np.abs(CALIB_DATE - daydelta))
        calib_offset = CALIB_VALUE[pos]

    return calib_offset


def get_zdr_offset(mydate) -> float:
    """
    Return the ZDR calibration offset for CPOL. Requires these global variables:
    - ZDR_CALIB_DATE_START
    - ZDR_CALIB_VALUE

    Parameters:
    ===========
    mydate: datetime
        Date of treatment.

    Returns:
    ========
    off: float
        Offset in dB to apply on ZDR. ZDR_CALIB = ZDR + OFFSET
    """
    # Transform into numpy type, so that we can use numpy functions
    mydate = np.array([mydate], dtype=np.datetime64)

    pos = np.argmin(np.abs(ZDR_CALIB_DATE_START - mydate))
    off = ZDR_CALIB_VALUE[pos]

    return off


def get_dbz_name(radar):
    """
    Find the name of the reflectivity field in radar
    """
    for dbz_name in ["DBZ", "UZ", "CZ", "Refl", None]:
        if dbz_name not in radar.fields.keys():
            continue
        else:
            break

    if dbz_name is None:
        raise KeyError(f"No reflectivity field found.")

    return dbz_name


def create_level1a(input_file: str) -> None:
    """
    Process level 0, apply calibration offsets and clean the metadata.

    Parameters:
    ==========
    input_file: str
        Name of the radar file
    """
    try:
        radar = pyart.io.read(input_file, file_field_names=True)
    except Exception:
        print(f"Problem with {input_file}.")
        traceback.print_exc()
        return None

    dtime = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    if dtime.year > 2020:  # Wrong century...
        dtime = dtime.replace(year=dtime.year - 100)

    output_dir = os.path.join(OUTDIR, str(dtime.year))
    mkdir(output_dir)
    output_dir = os.path.join(output_dir, dtime.strftime("%Y%m%d"))
    mkdir(output_dir)
    filename = "twp10cpolrhi.a1.{}.{}00.nc".format(dtime.strftime("%Y%m%d"), dtime.strftime("%H%M"))
    outfilename = os.path.join(output_dir, filename)
    if os.path.exists(outfilename):
        print(f"{outfilename} already exists. Doing nothing.")
        return None

    dbz_name = get_dbz_name(radar)
    zdr_calib_offset = get_zdr_offset(dtime)
    refl_calib_offset = get_calib_offset(dtime)

    radar.instrument_parameters["radar_beam_width_h"]["data"] = np.array([1], dtype=np.int32)
    radar.instrument_parameters["radar_beam_width_v"]["data"] = np.array([1], dtype=np.int32)

    fv = np.float32(-9999.0)
    refl = pyart.config.get_metadata("reflectivity")
    refl["data"] = np.ma.masked_equal(radar.fields[dbz_name]["data"].copy().filled(fv).astype(np.float32), fv)
    refl["data"] += np.float32(refl_calib_offset)
    refl["_FillValue"] = fv
    refl["_Least_significant_digit"] = 2
    radar.add_field("DBZ", refl, replace_existing=True)

    zdr = pyart.config.get_metadata("differential_reflectivity")
    zdr["data"] = np.ma.masked_equal(radar.fields["ZDR"]["data"].copy().filled(fv).astype(np.float32), fv)
    zdr["data"] += np.float32(zdr_calib_offset)
    zdr["_FillValue"] = fv
    zdr["_Least_significant_digit"] = 2
    radar.add_field("ZDR", zdr, replace_existing=True)

    width = pyart.config.get_metadata("spectrum_width")
    width["data"] = np.ma.masked_equal(radar.fields["WIDTH"]["data"].copy().filled(fv).astype(np.float32), fv)
    width["_FillValue"] = fv
    width["_Least_significant_digit"] = 2
    radar.add_field("WIDTH", width, replace_existing=True)

    phidp = pyart.config.get_metadata("differential_phase")
    phidp["data"] = np.ma.masked_equal(radar.fields["PHIDP"]["data"].copy().filled(fv).astype(np.float32), fv)
    phidp["_FillValue"] = fv
    phidp["_Least_significant_digit"] = 2
    radar.add_field("PHIDP", phidp, replace_existing=True)

    try:
        rhohv = pyart.config.get_metadata("cross_correlation_ratio")
        rhohv["data"] = np.ma.masked_equal(radar.fields["RHOHV"]["data"].copy().filled(fv).astype(np.float32), fv)
        rhohv["_FillValue"] = fv
        rhohv["_Least_significant_digit"] = 4
        radar.add_field("RHOHV", rhohv, replace_existing=True)
    except KeyError:
        pass

    vel = pyart.config.get_metadata("velocity")
    vel["data"] = np.ma.masked_equal(radar.fields["VEL"]["data"].copy().filled(fv).astype(np.float32), fv)
    vel["_FillValue"] = fv
    vel["_Least_significant_digit"] = 2
    vel["units"] = "m s-1"
    radar.add_field("VEL", vel, replace_existing=True)

    goodkeys = ["DBZ", "ZDR", "WIDTH", "PHIDP", "RHOHV", "VEL"]
    for k in list(radar.fields.keys()):
        if k not in goodkeys:
            radar.fields.pop(k)

    instrument_keys = ["frequency", "prt_mode", "prt", "prt_ratio", "polarization_mode", "nyquist_velocity"]
    for k in list(radar.instrument_parameters.keys()):
        if k not in instrument_keys:
            radar.instrument_parameters.pop(k)

    cpol_processing.cfmetadata.correct_standard_name(radar)
    cpol_processing.cfmetadata.coverage_content_type(radar)
    cpol_processing.cfmetadata.correct_units(radar)

    radar.metadata = get_metadata(radar)

    pyart.io.write_cfradial(outfilename, radar, format="NETCDF4", arm_time_variables=True, time_reference=True)
    print(f"{outfilename} written.")
    return None


def buffer(infile: str) -> None:
    """
    Buffer function to catch and kill errors about missing Sun hit.

    Parameters:
    ===========
    infile: str
        Input radar file.

    Returns:
    ========
    rslt: pd.DataFrame
        Pandas dataframe with the results from the solar calibration code.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            create_level1a(infile)
        except Exception:
            print(f"Problem with file {infile}.")
            traceback.print_exc()

    return None


def main(date_range) -> None:
    for date in date_range:
        # Get zip and extract.
        zipfile = get_radar_archive_file(date)
        if zipfile is None:
            print(crayons.red(f"No file found for date {date}."))
            continue
        namelist = extract_zip(zipfile, path=ZIPDIR)
        print(crayons.yellow(f"{len(namelist)} files to process for {date}."))

        # Process
        bag = db.from_sequence(namelist).map(buffer)
        bag.compute()

        # Clean up
        remove(namelist)
        gc.collect()

    return None


if __name__ == "__main__":
    INPATH = "/g/data/hj10/admin/cpol_level_0/rhi/"
    CALIBRATION_FILE = "../data/CALIB_OFFSET_october2017.pkl.gz"
    ZDR_CALIBRATION_FILE = "../data/CPOL_ZDR_calibration_offset.pkl.gz"
    OUTDIR = "/scratch/kl02/vhl548/cpol/cpol_level_1a/"
    ZIPDIR = "/scratch/kl02/vhl548/unzipdir/"

    parser_description = """Raw radar PPIs processing. It provides Quality
control, filtering, attenuation correction, dealiasing, unfolding, hydrometeors
calculation, and rainfall rate estimation."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-s", "--start-date", dest="start_date", default=None, type=str, help="Starting date.", required=True
    )
    parser.add_argument("-e", "--end-date", dest="end_date", default=None, type=str, help="Ending date.", required=True)

    args = parser.parse_args()
    START_DATE = args.start_date
    END_DATE = args.end_date
    # Check date
    try:
        start = datetime.datetime.strptime(START_DATE, "%Y%m%d")
        end = datetime.datetime.strptime(END_DATE, "%Y%m%d")
        if start > end:
            parser.error("Start date older than end date.")
        date_range = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1,)]
    except ValueError:
        parser.error("Invalid dates.")
        sys.exit()

    with gzip.GzipFile(CALIBRATION_FILE, "r") as gzid:
        tmp_data = pickle.load(gzid)
        # Z_CALIBRATED = Z_CPOL + CALIBRATION_VALUE
        try:
            CALIB_DATE_START = tmp_data["period_start"]
            CALIB_DATE_END = tmp_data["period_end"]
            CALIB_VALUE = tmp_data["calibration_value"]
            IS_CALIB_PERIOD = True
        except KeyError:
            CALIB_DATE = tmp_data["date"]
            CALIB_VALUE = tmp_data["calibration_value"]
            IS_CALIB_PERIOD = False

    # Opening calibration data file.
    with gzip.GzipFile(ZDR_CALIBRATION_FILE, "r") as gzid:
        tmp_data = pickle.load(gzid)
        ZDR_CALIB_DATE_START = tmp_data["period_start"]
        ZDR_CALIB_DATE_END = tmp_data["period_end"]
        ZDR_CALIB_VALUE = tmp_data["calibration_value"]

    main(date_range)
