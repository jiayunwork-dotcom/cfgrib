#
# Copyright 2017-2021 European Centre for Medium-Range Weather Forecasts (ECMWF).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
#   Alessandro Amici - B-Open - https://bopen.eu
#   Aureliana Barghini - B-Open - https://bopen.eu
#   Leonardo Barcaroli - B-Open - https://bopen.eu
#

import datetime
import itertools
import logging
import typing as T
import warnings

import numpy as np
import xarray as xr
from packaging import version

if version.parse(xr.__version__) < version.parse("2025.9.1"):
    from xarray.backends import api as backends_api
else:
    # Refactor: https://github.com/pydata/xarray/pull/10771/files
    from xarray.backends import writers as backends_api

from . import cfmessage, dataset, messages

LOGGER = logging.getLogger(__name__)


DEFAULT_GRIB_KEYS = {
    "centre": 255,  # missing value, see: http://apps.ecmwf.int/codes/grib/format/grib1/centre/0/
    "typeOfLevel": "surface",
}
TYPE_OF_LEVELS_SFC = ["surface", "meanSea", "cloudBase", "cloudTop"]
TYPE_OF_LEVELS_PL = ["isobaricInhPa", "isobaricInPa"]
TYPE_OF_LEVELS_ML = ["hybrid"]
ALL_TYPE_OF_LEVELS = TYPE_OF_LEVELS_SFC + TYPE_OF_LEVELS_PL + TYPE_OF_LEVELS_ML
GRID_TYPES = [
    "polar_stereographic",
    "reduced_gg",
    "reduced_ll",
    "regular_gg",
    "regular_ll",
    "rotated_gg",
    "rotated_ll",
    "sh",
]


def regular_ll_params(values, min_value=-180.0, max_value=360.0):
    # type: (np.ndarray, float, float) -> T.Tuple[float, float, int]
    start, stop, num = float(values[0]), float(values[-1]), len(values)
    if min(start, stop) < min_value or max(start, stop) > max_value:
        raise ValueError("Unsupported spatial grid: out of bounds (%r, %r)" % (start, stop))
    check_values = np.linspace(start, stop, num)
    if not np.allclose(check_values, values):
        raise ValueError("Unsupported spatial grid: not regular %r" % (check_values,))
    return (start, stop, num)


def detect_regular_ll_grib_keys(lon, lat):
    # type: (np.ndarray, np.ndarray) -> T.Dict[str, T.Any]
    grib_keys = {}  # type: T.Dict[str, T.Any]

    lon_start, lon_stop, lon_num = regular_ll_params(lon)
    lon_scan_negatively = lon_stop < lon_start
    lon_step = abs(lon_stop - lon_start) / (lon_num - 1.0)
    if lon_start < 0.0:
        lon_start += 360.0
    if lon_stop < 0.0:
        lon_stop += 360.0
    grib_keys["longitudeOfFirstGridPointInDegrees"] = lon_start
    grib_keys["longitudeOfLastGridPointInDegrees"] = lon_stop
    grib_keys["Ni"] = lon_num
    grib_keys["iDirectionIncrementInDegrees"] = lon_step
    grib_keys["iScansNegatively"] = lon_scan_negatively

    lat_start, lat_stop, lat_num = regular_ll_params(lat, min_value=-90.0, max_value=90.0)
    grib_keys["latitudeOfFirstGridPointInDegrees"] = lat_start
    grib_keys["latitudeOfLastGridPointInDegrees"] = lat_stop
    grib_keys["Nj"] = lat_num
    grib_keys["jDirectionIncrementInDegrees"] = abs(lat_stop - lat_start) / (lat_num - 1.0)
    grib_keys["jScansPositively"] = lat_stop > lat_start
    grib_keys["gridType"] = "regular_ll"

    return grib_keys


def detect_grib_keys(data_var, default_grib_keys, grib_keys={}):
    # type: (xr.DataArray, T.Dict[str, T.Any], T.Dict[str, T.Any]) -> T.Tuple[T.Dict[str, T.Any], T.Dict[str, T.Any]]
    detected_grib_keys = {}
    suggested_grib_keys = default_grib_keys.copy()

    for key_raw, value in data_var.attrs.items():
        key = str(key_raw)
        if key[:5] == "GRIB_":
            suggested_grib_keys[key[5:]] = value

    if "latitude" in data_var.dims and "longitude" in data_var.dims:
        try:
            regular_ll_keys = detect_regular_ll_grib_keys(data_var.longitude, data_var.latitude)
            detected_grib_keys.update(regular_ll_keys)
        except:
            pass

    for tol in ALL_TYPE_OF_LEVELS:
        if tol in data_var.dims or tol in data_var.coords:
            detected_grib_keys["typeOfLevel"] = tol

    if "number" in data_var.dims or "number" in data_var.coords and grib_keys.get("edition") != 1:
        # cannot set 'number' key without setting a productDefinitionTemplateNumber in GRIB2
        detected_grib_keys["productDefinitionTemplateNumber"] = 1

    if "values" in data_var.dims:
        detected_grib_keys["numberOfPoints"] = data_var.shape[data_var.dims.index("values")]

    return detected_grib_keys, suggested_grib_keys


def detect_sample_name(grib_keys, sample_name_template="{geography}_{vertical}_grib{edition}"):
    # type: (T.Mapping[str, T.Any], str) -> str
    edition = grib_keys.get("edition", 2)

    if grib_keys["gridType"] in GRID_TYPES:
        geography = grib_keys["gridType"]
    else:
        LOGGER.warning("unknown 'gridType': %r. Using GRIB2 template", grib_keys["gridType"])
        return "GRIB2"

    if grib_keys["typeOfLevel"] in TYPE_OF_LEVELS_PL:
        vertical = "pl"
    elif grib_keys["typeOfLevel"] in TYPE_OF_LEVELS_SFC:
        vertical = "sfc"
    elif grib_keys["typeOfLevel"] in TYPE_OF_LEVELS_ML:
        vertical = "ml"
    else:
        LOGGER.warning("unknown 'typeOfLevel': %r. Using GRIB2 template", grib_keys["typeOfLevel"])
        return "GRIB2"

    sample_name = sample_name_template.format(**locals())
    return sample_name


def merge_grib_keys(grib_keys, detected_grib_keys, default_grib_keys):
    # type: (T.Dict[str, T.Any], T.Dict[str, T.Any], T.Dict[str, T.Any]) -> T.Dict[str, T.Any]
    merged_grib_keys = {k: v for k, v in grib_keys.items()}
    dataset.dict_merge(merged_grib_keys, detected_grib_keys)
    for key, value in default_grib_keys.items():
        if key not in merged_grib_keys:
            merged_grib_keys[key] = value
    return merged_grib_keys


def expand_dims(data_var: xr.DataArray) -> T.Tuple[T.List[str], xr.DataArray]:
    coords_names = []  # type: T.List[str]
    for coord_name in dataset.ALL_HEADER_DIMS + ALL_TYPE_OF_LEVELS + dataset.ALL_REF_TIME_KEYS:
        if coord_name in data_var.coords and data_var.coords[coord_name].size == 1 and coord_name not in data_var.dims:
            data_var = data_var.expand_dims(coord_name)
        if coord_name in data_var.dims:
            coords_names.append(coord_name)
    return coords_names, data_var


def make_template_message(merged_grib_keys, template_path=None, sample_name=None):
    # type: (T.Dict[str, T.Any], T.Optional[str], T.Optional[str]) -> messages.Message
    if template_path and sample_name:
        raise ValueError("template_path and sample_name should not be both set")

    if template_path:
        with open(template_path, "rb") as file:
            template_message = cfmessage.CfMessage.from_file(file)
    else:
        if sample_name is None:
            sample_name = detect_sample_name(merged_grib_keys)
        template_message = cfmessage.CfMessage.from_sample_name(sample_name)

    for key, value in merged_grib_keys.items():
        try:
            template_message[key] = value
        except KeyError:
            LOGGER.exception("skipping key due to errors: %r" % key)

    return template_message


def canonical_dataarray_to_grib(data_var, file, grib_keys={}, default_grib_keys=DEFAULT_GRIB_KEYS, **kwargs):
    # type: (xr.DataArray, T.IO[bytes], T.Dict[str, T.Any], T.Dict[str, T.Any], T.Any) -> None
    """
    Write a ``xr.DataArray`` in *canonical* form to a GRIB file.
    """
    # validate Dataset keys, DataArray names, and attr keys/values
    detected_keys, suggested_keys = detect_grib_keys(data_var, default_grib_keys, grib_keys)
    merged_grib_keys = merge_grib_keys(grib_keys, detected_keys, suggested_keys)
    merged_grib_keys["missingValue"] = messages.MISSING_VAUE_INDICATOR

    if "gridType" not in merged_grib_keys:
        raise ValueError("required grib_key 'gridType' not passed nor auto-detected")

    template_message = make_template_message(merged_grib_keys, **kwargs)

    coords_names, data_var = expand_dims(data_var)

    header_coords_values = [list(data_var.coords[name].values) for name in coords_names]

    for items in itertools.product(*header_coords_values):
        select = {n: v for n, v in zip(coords_names, items)}
        field_values = data_var.sel(**select).values.flat[:]

        # Missing values handling
        invalid_field_values = np.logical_not(np.isfinite(field_values))

        # There's no need to save a message full of missing values
        if invalid_field_values.all():
            continue

        missing_value = merged_grib_keys.get("GRIB_missingValue", messages.MISSING_VAUE_INDICATOR)
        field_values[invalid_field_values] = missing_value

        message = cfmessage.CfMessage.from_message(template_message)
        for coord_name, coord_value in zip(coords_names, items):
            if coord_name in ALL_TYPE_OF_LEVELS:
                coord_name = "level"
            message[coord_name] = coord_value

        if invalid_field_values.any():
            message["bitmapPresent"] = 1
        message["missingValue"] = missing_value

        # OPTIMIZE: convert to list because Message.message_set doesn't support np.ndarray
        message["values"] = field_values.tolist()

        message.write(file)


def canonical_dataset_to_grib(dataset, path, mode="wb", no_warn=False, grib_keys={}, **kwargs):
    # type: (xr.Dataset, str, str, bool, T.Dict[str, T.Any], T.Any) -> None
    """
    Write a ``xr.Dataset`` in *canonical* form to a GRIB file.
    """
    if not no_warn:
        warnings.warn("GRIB write support is experimental, DO NOT RELY ON IT!", FutureWarning)

    # validate Dataset keys, DataArray names, and attr keys/values
    backends_api._validate_dataset_names(dataset)  # type: ignore
    # _validate_attrs takes the engine name as its 2nd arg from xarray 2024.09.0
    try:
        backends_api._validate_attrs(dataset)  # type: ignore
    except TypeError:
        backends_api._validate_attrs(dataset, "cfgrib")  # type: ignore

    real_grib_keys = {str(k)[5:]: v for k, v in dataset.attrs.items() if str(k)[:5] == "GRIB_"}
    real_grib_keys.update(grib_keys)

    # NOTE: interval_totals results are validated *before* the output file is
    #   opened, so an invalid or self-contradictory request can never truncate an
    #   existing file or leave a semantically mixed half-written product behind.
    interval_plans = {}
    for name, data_var in dataset.data_vars.items():
        if is_interval_totals(data_var):
            interval_plans[name] = prepare_interval_write(data_var, dataset, grib_keys=real_grib_keys, **kwargs)

    with open(path, mode=mode) as file:
        for name, data_var in dataset.data_vars.items():
            if name in interval_plans:
                write_interval_messages(data_var, file, interval_plans[name])
            else:
                canonical_dataarray_to_grib(data_var, file, grib_keys=real_grib_keys, **kwargs)


to_grib = canonical_dataset_to_grib


#
# Writing ``interval_totals`` results back to GRIB2
#

INTERVAL_TOTALS_MARKER = "GRIB_interval_totals"
INTERVAL_BOUNDS_COORD = "step_bounds"


def is_interval_totals(data_var: T.Any) -> bool:
    """Return ``True`` when ``data_var`` is an ``interval_totals`` result."""
    encoding = getattr(data_var, "encoding", None) or {}
    if encoding.get("interval_totals"):
        return True
    attrs = getattr(data_var, "attrs", None) or {}
    return bool(attrs.get(INTERVAL_TOTALS_MARKER))


def _timedelta_seconds(value: T.Any) -> T.Optional[int]:
    try:
        delta = np.timedelta64(value)
    except (TypeError, ValueError):
        return None
    if np.isnat(delta):
        return None
    total = delta / np.timedelta64(1, "s")
    if not float(total).is_integer():
        return None
    return int(total)


def _validate_bounds_array(bounds: T.Any, label: str) -> T.List[T.Tuple[int, int]]:
    bounds = np.asarray(bounds)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("%s must have shape (interval, 2), got %r" % (label, bounds.shape))
    intervals = []
    for start, end in bounds:
        start_seconds = _timedelta_seconds(start)
        end_seconds = _timedelta_seconds(end)
        if start_seconds is None or end_seconds is None:
            raise ValueError("%s must contain whole-second timedeltas" % label)
        if start_seconds < 0 or end_seconds <= start_seconds:
            raise ValueError(
                "%s has an invalid interval (%d, %d): endpoints must be non-negative "
                "and start must be smaller than end" % (label, start_seconds, end_seconds)
            )
        intervals.append((start_seconds, end_seconds))
    if len(set(intervals)) != len(intervals):
        raise ValueError("%s contains duplicate target intervals" % label)
    return intervals


def _bounds_from_coordinate(data_var: T.Any) -> T.List[T.Tuple[int, int]]:
    """Read and validate a ``step_bounds`` coordinate on a DataArray or Dataset."""
    if INTERVAL_BOUNDS_COORD not in data_var.coords:
        raise ValueError("interval_totals data variable is missing the %r coordinate" % INTERVAL_BOUNDS_COORD)
    bounds_coord = data_var.coords[INTERVAL_BOUNDS_COORD]
    if "interval" not in bounds_coord.dims:
        raise ValueError("%r coordinate must depend on the 'interval' dimension" % INTERVAL_BOUNDS_COORD)
    return _validate_bounds_array(bounds_coord.values, repr(INTERVAL_BOUNDS_COORD))


def _bounds_from_seconds(values: T.Any) -> T.List[T.Tuple[int, int]]:
    """Convert a sequence of integer-second ``(start, end)`` pairs from an encoding."""
    intervals = []
    for item in values:
        try:
            pair = list(item)
        except TypeError:
            raise ValueError("encoded target intervals must be (start, end) pairs")
        if len(pair) != 2:
            raise ValueError("encoded target intervals must be (start, end) pairs")
        start_seconds, end_seconds = pair
        if not isinstance(start_seconds, (int, np.integer)) or not isinstance(end_seconds, (int, np.integer)):
            raise ValueError("encoded target interval endpoints must be integer seconds")
        intervals.append((int(start_seconds), int(end_seconds)))
    dummy = np.array(
        [[np.timedelta64(start, "s"), np.timedelta64(end, "s")] for start, end in intervals],
        dtype="timedelta64[s]",
    )
    return _validate_bounds_array(dummy, "encoded target intervals")


def _resolve_intervals(data_var: T.Any, dataset: T.Any = None) -> T.List[T.Tuple[int, int]]:
    """Collect every description of the target intervals and reject conflicts.

    The ``step_bounds`` coordinate can live on the DataArray (when it is passed on
    its own) or on the enclosing Dataset (``bounds`` is not a dimension of the data
    variable, so xarray drops that coordinate from a selected DataArray).  The
    reader also stores the exact intervals in the variable encoding; the two must
    agree, otherwise the writer would silently emit a different product.
    """
    candidates: T.Dict[str, T.List[T.Tuple[int, int]]] = {}
    if INTERVAL_BOUNDS_COORD in getattr(data_var, "coords", {}):
        candidates["step_bounds coordinate"] = _bounds_from_coordinate(data_var)
    elif dataset is not None and INTERVAL_BOUNDS_COORD in dataset.coords:
        candidates["step_bounds coordinate"] = _bounds_from_coordinate(dataset)

    encoding = getattr(data_var, "encoding", None) or {}
    encoded = encoding.get("interval_totals_intervals")
    if encoded is not None:
        candidates["encoded intervals"] = _bounds_from_seconds(encoded)

    if not candidates:
        raise ValueError(
            "interval_totals data variable has neither a %r coordinate nor encoded target intervals" % INTERVAL_BOUNDS_COORD
        )
    first_label, first_value = next(iter(candidates.items()))
    for label, value in candidates.items():
        if value != first_value:
            raise ValueError(
                "conflicting target intervals: %r (%r) disagrees with %r (%r)" % (first_label, first_value, label, value)
            )
    return first_value


def _interval_jobs(
    data_var: T.Any,
    intervals: T.Sequence[T.Tuple[int, int]],
) -> T.List[T.Tuple[int, int, np.datetime64]]:
    """Validate the time axis and return one (time index, interval index, time) per message."""
    if "time" not in data_var.dims:
        raise ValueError("interval_totals data variable must depend on the 'time' dimension")
    if "interval" not in data_var.dims:
        raise ValueError("interval_totals data variable must depend on the 'interval' dimension")
    geo_dims = [dim for dim in data_var.dims if dim not in ("time", "interval")]
    if set(geo_dims) != {"latitude", "longitude"}:
        raise ValueError("interval_totals data variable must be indexed by latitude and longitude, got %r" % (geo_dims,))
    times = np.asarray(data_var.coords["time"].values)
    if times.dtype.kind != "M":
        raise ValueError("'time' coordinate must contain datetime64 values")
    if np.isnat(times).any():
        raise ValueError("'time' coordinate must not contain NaT")
    if len(np.unique(times)) != len(times):
        raise ValueError("'time' coordinate must not contain duplicate reference times")
    if "units" not in data_var.attrs:
        raise ValueError("interval_totals data variable must carry a 'units' attribute")
    jobs = []
    for time_index, time_value in enumerate(times):
        for interval_index in range(len(intervals)):
            jobs.append((time_index, interval_index, time_value))
    return jobs


def _apply_accumulation_keys(
    message: T.Any,
    reference: np.datetime64,
    start_seconds: int,
    end_seconds: int,
    missing_value: float,
    product_keys: T.Mapping[str, T.Any],
) -> None:
    """Configure ``message`` as one deterministic GRIB2 accumulation interval.

    Endpoints are encoded with second resolution (GRIB2 time unit 13), which can
    represent any integer-second interval without rounding.
    """
    for key, value in product_keys.items():
        try:
            message[key] = value
        except Exception:
            LOGGER.debug("skipping interval_totals grib key %r", key)
    message["productDefinitionTemplateNumber"] = 8
    message["typeOfStatisticalProcessing"] = 1
    # deterministic forecast, not an ensemble or a re-analysis
    message["typeOfProcessedData"] = 1
    reference_seconds = int(np.datetime64(reference, "s").astype(np.int64))
    reference_dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=reference_seconds)
    message["dataDate"] = int(reference_dt.strftime("%Y%m%d"))
    message["dataTime"] = int(reference_dt.strftime("%H%M"))
    message["stepUnits"] = 13
    message["startStep"] = start_seconds
    message["endStep"] = end_seconds
    message["missingValue"] = missing_value


def _interval_product_keys(data_var: T.Any, grib_keys: T.Mapping[str, T.Any]) -> T.Dict[str, T.Any]:
    """Build the message keys that describe the physical product of the result."""
    keys: T.Dict[str, T.Any] = {}
    # The result only carries physical attributes that describe the source product;
    # derived headers (dataType, cfName, ...) are recomputed by ecCodes from the
    # parameter codes and must not be forced through a template.
    for key in (
        "shortName",
        "paramId",
        "typeOfLevel",
        "gridType",
        "Ni",
        "Nj",
        "Nx",
        "Ny",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "iDirectionIncrementInDegrees",
        "jDirectionIncrementInDegrees",
        "iScansNegatively",
        "jScansPositively",
        "jPointsAreConsecutive",
    ):
        value = grib_keys.get(key, data_var.attrs.get("GRIB_" + key))
        if value is not None:
            keys[key] = value
    return keys


class IntervalWritePlan:
    """Validated, read-only description of an ``interval_totals`` write request."""

    def __init__(
        self,
        data_var: T.Any,
        intervals: T.List[T.Tuple[int, int]],
        jobs: T.List[T.Tuple[int, int, np.datetime64]],
        product_keys: T.Dict[str, T.Any],
        sample_name: str,
        packing_keys: T.Dict[str, T.Any],
        units: str,
    ):
        self.data_var = data_var
        self.intervals = intervals
        self.jobs = jobs
        self.product_keys = product_keys
        self.sample_name = sample_name
        self.packing_keys = packing_keys
        self.units = units


def prepare_interval_write(
    data_var: T.Any,
    dataset: T.Any = None,
    grib_keys: T.Dict[str, T.Any] = {},
    default_grib_keys: T.Dict[str, T.Any] = DEFAULT_GRIB_KEYS,
    **kwargs: T.Any,
) -> IntervalWritePlan:
    """Validate an ``interval_totals`` write request without touching the output.

    Everything that can fail does so here: interval coordinates, time axis, units,
    grid layout and template selection.  The returned plan is then written message
    by message, so an error can never leave a partially written file.
    """
    intervals = _resolve_intervals(data_var, dataset)
    jobs = _interval_jobs(data_var, intervals)

    merged_grib_keys = dict(default_grib_keys)
    merged_grib_keys.update({str(k)[5:]: v for k, v in data_var.attrs.items() if str(k)[:5] == "GRIB_"})
    merged_grib_keys.update(grib_keys)
    product_keys = _interval_product_keys(data_var, merged_grib_keys)
    if "gridType" not in product_keys:
        raise ValueError("required grib_key 'gridType' not passed nor auto-detected")

    units = data_var.attrs.get("units")
    if units is None:
        raise ValueError("interval_totals data variable must carry a 'units' attribute")
    if grib_keys.get("units") is not None and str(grib_keys["units"]) != str(units):
        raise ValueError(
            "interval_totals data variable has units %r but the requested grib_key 'units' is %r" % (units, grib_keys["units"])
        )

    template_path = kwargs.get("template_path")
    if template_path is not None:
        raise ValueError(
            "template_path is not supported for interval_totals results; the accumulation product definition is fixed"
        )
    sample_name = kwargs.get("sample_name") or "regular_ll_sfc_grib2"

    # Packing precision is only taken from explicitly requested keys, so the
    # default keeps ecCodes' own packing (24 bits by default for these messages).
    packing_keys = {}
    for key in ("bitsPerValue", "packingType", "decimalScaleFactor", "binaryScaleFactor"):
        if key in merged_grib_keys:
            packing_keys[key] = merged_grib_keys[key]
        elif key in grib_keys:
            packing_keys[key] = grib_keys[key]

    return IntervalWritePlan(
        data_var=data_var,
        intervals=intervals,
        jobs=jobs,
        product_keys=product_keys,
        sample_name=sample_name,
        packing_keys=packing_keys,
        units=str(units),
    )


def write_interval_messages(
    data_var: T.Any,
    file: T.IO[bytes],
    plan: IntervalWritePlan,
) -> None:
    """Write one message per (reference time, target interval), all-missing included."""
    intervals = plan.intervals
    geo_dims = [dim for dim in data_var.dims if dim not in ("time", "interval")]
    if geo_dims != ["latitude", "longitude"]:
        data_var = data_var.transpose("time", "interval", "latitude", "longitude")

    missing_value = float(messages.MISSING_VAUE_INDICATOR)
    template_message = messages.Message.from_sample_name(plan.sample_name)
    try:
        for time_index, interval_index, time_value in plan.jobs:
            field_values = np.asarray(
                data_var.isel(time=time_index, interval=interval_index).values, dtype=np.float64
            ).reshape(-1)
            invalid = ~np.isfinite(field_values)
            start_seconds, end_seconds = intervals[interval_index]
            message = messages.Message.from_message(template_message)
            try:
                _apply_accumulation_keys(
                    message,
                    time_value,
                    start_seconds,
                    end_seconds,
                    missing_value,
                    plan.product_keys,
                )
                for key, value in plan.packing_keys.items():
                    message[key] = value
                if invalid.any():
                    # NOTE: an all-missing field still gets a message, so the bitmap
                    #   is enabled even when every point is missing.
                    message["bitmapPresent"] = 1
                    field_values[invalid] = missing_value
                else:
                    message["bitmapPresent"] = 0
                message["values"] = field_values.tolist()
                message.write(file)
            finally:
                del message
    finally:
        del template_message


def canonical_interval_dataarray_to_grib(
    data_var: T.Any,
    file: T.IO[bytes],
    grib_keys: T.Dict[str, T.Any] = {},
    default_grib_keys: T.Dict[str, T.Any] = DEFAULT_GRIB_KEYS,
    dataset: T.Any = None,
    **kwargs: T.Any,
) -> None:
    """Write an ``interval_totals`` variable as one message per (time, interval)."""
    plan = prepare_interval_write(data_var, dataset, grib_keys=grib_keys, default_grib_keys=default_grib_keys, **kwargs)
    write_interval_messages(data_var, file, plan)
