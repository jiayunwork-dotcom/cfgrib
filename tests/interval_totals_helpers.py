import datetime
import typing as T

import eccodes  # type: ignore
import numpy as np

#: default 4x3 regular_ll grid: latitudes 10, 5, 0 and longitudes 0, 10, 20, 30
DEFAULT_NI = 4
DEFAULT_NJ = 3


def make_accumulation_message(
    start_seconds: int,
    end_seconds: int,
    values: T.Sequence[float],
    *,
    reference: T.Tuple[int, int, int, int, int] = (2020, 1, 1, 0, 0),
    ni: int = DEFAULT_NI,
    nj: int = DEFAULT_NJ,
    short_name: str = "tp",
    unit: int = 13,
    bits_per_value: int = 24,
    product_definition_template: int = 8,
    statistical_processing: int = 1,
    type_of_processed_data: int = 1,
    grid_type: str = "regular_ll",
    type_of_level: str = "surface",
    extra_keys: T.Optional[T.Mapping[str, T.Any]] = None,
) -> bytes:
    """Return the serialised bytes of one small GRIB2 accumulation message."""
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        scale = 1 if unit == 13 else 3600
        keys = {
            "Ni": ni,
            "Nj": nj,
            "numberOfDataPoints": ni * nj,
            "gridType": grid_type,
            "latitudeOfFirstGridPointInDegrees": 10.0,
            "latitudeOfLastGridPointInDegrees": 0.0,
            "longitudeOfFirstGridPointInDegrees": 0.0,
            "longitudeOfLastGridPointInDegrees": 30.0,
            "iDirectionIncrementInDegrees": 10.0,
            "jDirectionIncrementInDegrees": 5.0,
            "iScansNegatively": 0,
            "jScansPositively": 0,
            "productDefinitionTemplateNumber": product_definition_template,
            "typeOfStatisticalProcessing": statistical_processing,
            "typeOfProcessedData": type_of_processed_data,
            "dataDate": reference[0] * 10000 + reference[1] * 100 + reference[2],
            "dataTime": reference[3] * 100 + reference[4],
            "typeOfLevel": type_of_level,
            "shortName": short_name,
            "stepUnits": unit,
            "startStep": start_seconds // scale,
            "endStep": end_seconds // scale,
        }
        if extra_keys:
            keys.update(extra_keys)
        for key, value in keys.items():
            try:
                eccodes.codes_set(handle, key, value)
            except eccodes.GribInternalError:
                # A key that the product template does not carry is left to ecCodes.
                pass
        # NOTE: switching the product definition template resets the step keys, so
        #   the template and the step range are re-applied after everything else.
        eccodes.codes_set(handle, "productDefinitionTemplateNumber", product_definition_template)
        if product_definition_template in (8, 11):
            eccodes.codes_set(handle, "typeOfStatisticalProcessing", statistical_processing)
            eccodes.codes_set(handle, "stepUnits", unit)
            eccodes.codes_set(handle, "startStep", start_seconds // scale)
            eccodes.codes_set(handle, "endStep", end_seconds // scale)
        if extra_keys:
            for key, value in extra_keys.items():
                eccodes.codes_set(handle, key, value)
        # dataType is derived from typeOfProcessedData, which the template switch
        # can reset, so it is applied last.
        eccodes.codes_set(handle, "typeOfProcessedData", type_of_processed_data)
        array = np.array(values, dtype=np.float64)
        missing = ~np.isfinite(array)
        if missing.any():
            eccodes.codes_set(handle, "bitmapPresent", 1)
            array[missing] = eccodes.codes_get(handle, "missingValue")
        if bits_per_value is not None:
            eccodes.codes_set(handle, "bitsPerValue", bits_per_value)
        eccodes.codes_set_array(handle, "values", array)
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


def write_grib(path: str, messages: T.Sequence[bytes]) -> str:
    """Append ``messages`` to ``path`` (so several calls can build one file)."""
    with open(path, "ab") as file:
        for message in messages:
            file.write(message)
    return path


def build_interval_file(
    path: str,
    accumulations: T.Sequence[T.Tuple[int, int, T.Sequence[float]]],
    **kwargs: T.Any,
) -> str:
    """Write one file from ``(start_seconds, end_seconds, values)`` triples."""
    messages = []
    for start_seconds, end_seconds, values in accumulations:
        messages.append(make_accumulation_message(start_seconds, end_seconds, values, **kwargs))
    return write_grib(path, messages)


def read_messages(path: str) -> T.List[T.Dict[str, T.Any]]:
    """Return headers and values of every message, read back with ecCodes."""
    result = []
    with open(path, "rb") as file:
        while True:
            handle = eccodes.codes_grib_new_from_file(file)
            if handle is None:
                break
            try:
                header = {}
                for key in (
                    "edition",
                    "gridType",
                    "Ni",
                    "Nj",
                    "numberOfPoints",
                    "stepType",
                    "stepUnits",
                    "startStep",
                    "endStep",
                    "forecastTime",
                    "lengthOfTimeRange",
                    "indicatorOfUnitOfTimeRange",
                    "indicatorOfUnitForTimeRange",
                    "dataDate",
                    "dataTime",
                    "validityDate",
                    "validityTime",
                    "productDefinitionTemplateNumber",
                    "typeOfStatisticalProcessing",
                    "typeOfProcessedData",
                    "dataType",
                    "shortName",
                    "units",
                    "paramId",
                    "typeOfLevel",
                    "bitmapPresent",
                    "numberOfMissing",
                    "numberOfValues",
                    "bitsPerValue",
                ):
                    try:
                        header[key] = eccodes.codes_get(handle, key)
                    except eccodes.KeyValueNotFoundError:
                        header[key] = None
                header["values"] = eccodes.codes_get_array(handle, "values")
                header["missingValue"] = eccodes.codes_get(handle, "missingValue")
            finally:
                eccodes.codes_release(handle)
            result.append(header)
    return result


def seconds(value: T.Any) -> int:
    """Convert a timedelta-ish value to integer seconds (test helper only)."""
    if isinstance(value, datetime.timedelta):
        return int(value.total_seconds())
    return int(value / np.timedelta64(1, "s"))
