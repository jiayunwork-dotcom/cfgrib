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
#

import datetime
import itertools
import json
import logging
import os
import typing as T

import attr
import numpy as np

from . import __version__, abc, cfmessage, messages

LOG = logging.getLogger(__name__)

#
# Edition-independent keys in ecCodes namespaces. Documented in:
#   https://software.ecmwf.int/wiki/display/ECC/GRIB%3A+Namespaces
#
GLOBAL_ATTRIBUTES_KEYS = ["edition", "centre", "centreDescription", "subCentre"]

DATA_ATTRIBUTES_KEYS = [
    "paramId",
    "dataType",
    "numberOfPoints",
    "typeOfLevel",
    "stepUnits",
    "stepType",
    "gridType",
    "uvRelativeToGrid",
]

EXTRA_DATA_ATTRIBUTES_KEYS = [
    "shortName",
    "units",
    "name",
    "cfName",
    "cfVarName",
    "missingValue",
    "totalNumber",
    "numberOfDirections",
    "numberOfFrequencies",
    "NV",
    "gridDefinitionDescription",
]

GRID_TYPE_MAP = {
    "regular_ll": [
        "Nx",
        "iDirectionIncrementInDegrees",
        "iScansNegatively",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "Ny",
        "jDirectionIncrementInDegrees",
        "jPointsAreConsecutive",
        "jScansPositively",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
    ],
    "rotated_ll": [
        "Nx",
        "Ny",
        "angleOfRotationInDegrees",
        "iDirectionIncrementInDegrees",
        "iScansNegatively",
        "jDirectionIncrementInDegrees",
        "jPointsAreConsecutive",
        "jScansPositively",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "latitudeOfSouthernPoleInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "longitudeOfSouthernPoleInDegrees",
    ],
    "reduced_ll": [
        "Ny",
        "jDirectionIncrementInDegrees",
        "jPointsAreConsecutive",
        "jScansPositively",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
    ],
    "regular_gg": [
        "N",
        "Ni",
        "Nj",
        "iDirectionIncrementInDegrees",
        "iScansNegatively",
        "jScansPositively",
        "jPointsAreConsecutive",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
    ],
    "rotated_gg": [
        "Nx",
        "Ny",
        "angleOfRotationInDegrees",
        "iDirectionIncrementInDegrees",
        "iScansNegatively",
        "jPointsAreConsecutive",
        "jScansPositively",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "latitudeOfSouthernPoleInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "longitudeOfSouthernPoleInDegrees",
        "N",
    ],
    "lambert": [
        "LaDInDegrees",
        "LoVInDegrees",
        "iScansNegatively",
        "jPointsAreConsecutive",
        "jScansPositively",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfSouthernPoleInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfSouthernPoleInDegrees",
        "DyInMetres",
        "DxInMetres",
        "Latin2InDegrees",
        "Latin1InDegrees",
        "Ny",
        "Nx",
    ],
    "reduced_gg": ["N", "pl"],
    "sh": ["M", "K", "J"],
}
GRID_TYPE_KEYS = sorted(set(k for _, ks in GRID_TYPE_MAP.items() for k in ks))

ENSEMBLE_KEYS = ["number"]
VERTICAL_KEYS = ["level:float"]
DATA_TIME_KEYS = ["dataDate", "dataTime", "endStep"]
ALL_REF_TIME_KEYS = [
    "time",
    "step",
    "valid_time",
    "verifying_time",
    "forecastMonth",
    "indexing_time",
    "valid_month",
]
SPECTRA_KEYS = ["directionNumber", "frequencyNumber"]

ALL_HEADER_DIMS = ENSEMBLE_KEYS + VERTICAL_KEYS + SPECTRA_KEYS

HASH_KEYS = ["md5GridSection"]

INDEX_KEYS = sorted(
    GLOBAL_ATTRIBUTES_KEYS + DATA_ATTRIBUTES_KEYS + DATA_TIME_KEYS + ALL_HEADER_DIMS + HASH_KEYS
)

COORD_ATTRS = {
    # geography
    "latitude": {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude"},
    "longitude": {"units": "degrees_east", "standard_name": "longitude", "long_name": "longitude"},
    # vertical
    "depthBelowLand": {
        "units": "m",
        "positive": "down",
        "long_name": "soil depth",
        "standard_name": "depth",
    },
    "depthBelowLandLayer": {
        "units": "m",
        "positive": "down",
        "long_name": "soil depth",
        "standard_name": "depth",
    },
    "hybrid": {
        "units": "1",
        "positive": "down",
        "long_name": "hybrid level",
        "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
    },
    "heightAboveGround": {
        "units": "m",
        "positive": "up",
        "long_name": "height above the surface",
        "standard_name": "height",
    },
    "isobaricInhPa": {
        "units": "hPa",
        "positive": "down",
        "stored_direction": "decreasing",
        "standard_name": "air_pressure",
        "long_name": "pressure",
    },
    "isobaricInPa": {
        "units": "Pa",
        "positive": "down",
        "stored_direction": "decreasing",
        "standard_name": "air_pressure",
        "long_name": "pressure",
    },
    "isobaricLayer": {
        "units": "Pa",
        "positive": "down",
        "standard_name": "air_pressure",
        "long_name": "pressure",
    },
    # ensemble
    "number": {
        "units": "1",
        "standard_name": "realization",
        "long_name": "ensemble member numerical id",
    },
    # duplicate messages
    "grib_occurrence": {
        "units": "1",
        "long_name": "occurrence index of a GRIB message within a header coordinate cell",
    },
    # time
    "step": {
        "units": "hours",
        "standard_name": "forecast_period",
        "long_name": "time since forecast_reference_time",
        "dtype": "timedelta64[ns]",
    },
    "time": {
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "proleptic_gregorian",
        "standard_name": "forecast_reference_time",
        "long_name": "initial time of forecast",
    },
    "indexing_time": {
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "proleptic_gregorian",
        "standard_name": "forecast_reference_time",
        "long_name": "nominal initial time of forecast",
    },
    "valid_time": {
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "proleptic_gregorian",
        "standard_name": "time",
        "long_name": "time",
    },
    "valid_month": {
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "proleptic_gregorian",
        "standard_name": "time",
        "long_name": "time",
    },
    "verifying_time": {
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "proleptic_gregorian",
        "standard_name": "time",
        "long_name": "time",
    },
    "forecastMonth": {"units": "1", "long_name": "months since forecast_reference_time"},
}

GEOCACHE: T.Dict[
    T.Hashable, T.Tuple[T.Tuple[str, ...], T.Tuple[int, ...], T.Dict[str, "Variable"]]
] = {}


class DatasetBuildError(ValueError):
    def __str__(self) -> str:
        return str(self.args[0])


def enforce_unique_attributes(index, attributes_keys, filter_by_keys={}):
    # type: (T.Mapping[str, T.List[T.Any]], T.Sequence[str], T.Dict[str, T.Any]) -> T.Dict[str, T.Any]
    attributes = {}  # type: T.Dict[str, T.Any]
    for key in attributes_keys:
        values = index.get(key, [])
        if len(values) > 1:
            fbks = []
            for value in values:
                fbk = {key: value}
                fbk.update(filter_by_keys)
                fbks.append(fbk)
            raise DatasetBuildError("multiple values for key %r" % key, key, fbks)
        if values and values[0] not in ("undef", "unknown"):
            attributes["GRIB_" + key] = values[0]
    return attributes


@attr.attrs(auto_attribs=True, eq=False)
class Variable:
    dimensions: T.Tuple[str, ...]
    data: np.ndarray
    attributes: T.Dict[str, T.Any] = attr.attrib(default={}, repr=False)

    def __eq__(self, other):
        # type: (T.Any) -> bool
        if other.__class__ is not self.__class__:
            return NotImplemented
        equal = (self.dimensions, self.attributes) == (other.dimensions, other.attributes)
        return equal and np.array_equal(self.data, other.data)


def expand_item(item, shape):
    # type: (T.Tuple[T.Any, ...], T.Sequence[int]) -> T.Tuple[T.List[int], ...]
    expanded_item = []
    for i, size in zip(item, shape):
        if isinstance(i, (list, np.ndarray)):
            expanded_item.append([int(e) for e in i])
        elif isinstance(i, slice):
            expanded_item.append(list(range(i.start or 0, i.stop or size, i.step or 1)))
        elif isinstance(i, int):
            expanded_item.append([i])
        else:
            raise TypeError("Unsupported index type %r" % type(i))
    return tuple(expanded_item)


def get_values_in_order(message, shape):
    # type: (abc.Field, T.Tuple[int]) -> np.ndarray
    # inform the data provider to return missing values as missing_value
    values = message["values"]
    # for 2D array (lat/lon) re-arrange if alternative row scanning
    if len(shape) == 2 and message.get("alternativeRowScanning", False):
        values = values.copy().reshape(shape)
        values[1::2, :] = values[1::2, ::-1]
        return values.flatten()
    else:
        return values


@attr.attrs(auto_attribs=True)
class OnDiskArray:
    index: abc.Index[T.Any, abc.Field]
    shape: T.Tuple[int, ...]
    field_id_index: T.Dict[T.Tuple[T.Any, ...], T.List[T.Union[int, T.Tuple[int, int]]]] = (
        attr.attrib(repr=False)
    )
    missing_value: float
    geo_ndim: int = attr.attrib(default=1, repr=False)
    dtype: np.dtype = attr.attrib(default=messages.DEFAULT_VALUES_DTYPE, repr=False)
    # NOTE: the two attributes below are set only when duplicate messages are kept. They allow
    #   addressing a single message per (non-occurrence) header cell and per occurrence, without
    #   scanning the whole index and without decoding messages that were not selected.
    occurrence_axis: T.Optional[int] = attr.attrib(default=None, repr=False)
    occurrence_values: T.Optional[np.ndarray] = attr.attrib(default=None, repr=False)
    cell_field_ids: T.Optional[
        T.Dict[T.Tuple[T.Any, ...], T.Dict[T.Any, T.Any]]
    ] = attr.attrib(default=None, repr=False)

    def build_array(self) -> np.ndarray:
        """Helper method used to test __getitem__"""
        if self.cell_field_ids is not None:
            return self.__getitem__(tuple(slice(None) for _ in self.shape))
        array = np.full(self.shape, fill_value=np.nan, dtype=self.dtype)
        for header_indexes, message_ids in self.field_id_index.items():
            # NOTE: fill a single field as found in the message
            message = self.index.get_field(message_ids[0])  # type: ignore
            values = get_values_in_order(message, array[header_indexes].shape)
            array.__getitem__(header_indexes).flat[:] = values
        array[array == self.missing_value] = np.nan
        return array

    def _getitem_stacked(self, item: T.Tuple[T.Any, ...]) -> np.ndarray:
        # NOTE: occurrence-aware lookup: only the selected header cells and the selected
        #   occurrences are decoded. Missing (cell, occurrence) combinations stay NaN.
        header_item_list = expand_item(item[: -self.geo_ndim], self.shape)
        array_field_shape = tuple(len(i) for i in header_item_list) + self.shape[-self.geo_ndim :]
        array_field = np.full(array_field_shape, fill_value=np.nan, dtype=self.dtype)
        occurrence_axis = self.occurrence_axis  # type: ignore
        assert occurrence_axis is not None and self.cell_field_ids is not None
        assert self.occurrence_values is not None
        other_axes = [axis for axis in range(len(header_item_list)) if axis != occurrence_axis]
        occurrence_labels = self.occurrence_values.tolist()
        # NOTE: iterate over the output header cells, not over their distinct values, so that
        #   repeated positions in the requested item are all filled.
        other_positions_iter = itertools.product(
            *[range(len(header_item_list[axis])) for axis in other_axes]
        )
        for other_positions in other_positions_iter:
            cell = tuple(
                header_item_list[axis][position]
                for axis, position in zip(other_axes, other_positions)
            )
            for occurrence_position in range(len(header_item_list[occurrence_axis])):
                occurrence_label = occurrence_labels[
                    header_item_list[occurrence_axis][occurrence_position]
                ]
                message_id = self.cell_field_ids.get((cell, occurrence_label))
                if message_id is None:
                    continue
                array_field_indexes = list(other_positions)
                array_field_indexes.insert(occurrence_axis, occurrence_position)
                array_field_indexes_t = tuple(array_field_indexes)
                message = self.index.get_field(message_id)
                values = get_values_in_order(message, array_field[array_field_indexes_t].shape)
                array_field[array_field_indexes_t].flat[:] = values

        array = np.asarray(array_field[(Ellipsis,) + item[-self.geo_ndim :]])
        array[array == self.missing_value] = np.nan
        for i, it in reversed(list(enumerate(item[: -self.geo_ndim]))):
            if isinstance(it, int):
                array = array[(slice(None, None, None),) * i + (0,)]
        return array

    def __getitem__(self, item):
        # type: (T.Tuple[T.Any, ...]) -> np.ndarray
        if self.cell_field_ids is not None:
            return self._getitem_stacked(item)
        header_item_list = expand_item(item[: -self.geo_ndim], self.shape)
        header_item = [{ix: i for i, ix in enumerate(it)} for it in header_item_list]
        array_field_shape = tuple(len(i) for i in header_item_list) + self.shape[-self.geo_ndim :]
        array_field = np.full(array_field_shape, fill_value=np.nan, dtype=self.dtype)
        for header_indexes, message_ids in self.field_id_index.items():
            try:
                array_field_indexes = [it[ix] for it, ix in zip(header_item, header_indexes)]
            except KeyError:
                continue
            # NOTE: fill a single field as found in the message
            message = self.index.get_field(message_ids[0])  # type: ignore
            values = get_values_in_order(message, array_field[tuple(array_field_indexes)].shape)
            array_field.__getitem__(tuple(array_field_indexes)).flat[:] = values

        array = np.asarray(array_field[(Ellipsis,) + item[-self.geo_ndim :]])
        array[array == self.missing_value] = np.nan
        for i, it in reversed(list(enumerate(item[: -self.geo_ndim]))):
            if isinstance(it, int):
                array = array[(slice(None, None, None),) * i + (0,)]
        return array


GRID_TYPES_DIMENSION_COORDS = {"regular_ll", "regular_gg"}
GRID_TYPES_2D_NON_DIMENSION_COORDS = {
    "rotated_ll",
    "rotated_gg",
    "lambert",
    "lambert_azimuthal_equal_area",
    "albers",
    "polar_stereographic",
}


def build_geography_coordinates(
    first: abc.Field, encode_cf: T.Sequence[str], errors: str, log: logging.Logger = LOG
) -> T.Tuple[T.Tuple[str, ...], T.Tuple[int, ...], T.Dict[str, Variable]]:
    geo_coord_vars = {}  # type: T.Dict[str, Variable]
    grid_type = first["gridType"]
    if "geography" in encode_cf and grid_type in GRID_TYPES_DIMENSION_COORDS:
        geo_dims = ("latitude", "longitude")  # type: T.Tuple[str, ...]
        geo_shape = (first["Ny"], first["Nx"])  # type: T.Tuple[int, ...]
        latitudes = np.array(first["distinctLatitudes"], ndmin=1)
        geo_coord_vars["latitude"] = Variable(
            dimensions=("latitude",), data=latitudes, attributes=COORD_ATTRS["latitude"].copy()
        )

        if latitudes[0] > latitudes[-1]:
            geo_coord_vars["latitude"].attributes["stored_direction"] = "decreasing"
        geo_coord_vars["longitude"] = Variable(
            dimensions=("longitude",),
            data=np.array(first["distinctLongitudes"], ndmin=1),
            attributes=COORD_ATTRS["longitude"],
        )
    elif "geography" in encode_cf and grid_type in GRID_TYPES_2D_NON_DIMENSION_COORDS:
        geo_dims = ("y", "x")
        geo_shape = (first["Ny"], first["Nx"])
        try:
            geo_coord_vars["latitude"] = Variable(
                dimensions=("y", "x"),
                data=np.array(first["latitudes"]).reshape(geo_shape),
                attributes=COORD_ATTRS["latitude"],
            )
            geo_coord_vars["longitude"] = Variable(
                dimensions=("y", "x"),
                data=np.array(first["longitudes"]).reshape(geo_shape),
                attributes=COORD_ATTRS["longitude"],
            )
        except KeyError:  # pragma: no cover
            if errors != "ignore":
                log.warning("ecCodes provides no latitudes/longitudes for gridType=%r", grid_type)
    else:
        geo_dims = ("values",)
        geo_shape = (first["numberOfPoints"],)
        # add secondary coordinates if ecCodes provides them
        try:
            latitude = first["latitudes"]
            geo_coord_vars["latitude"] = Variable(
                dimensions=("values",), data=np.array(latitude), attributes=COORD_ATTRS["latitude"]
            )
            longitude = first["longitudes"]
            geo_coord_vars["longitude"] = Variable(
                dimensions=("values",),
                data=np.array(longitude),
                attributes=COORD_ATTRS["longitude"],
            )
        except KeyError:  # pragma: no cover
            if errors != "ignore":
                log.warning("ecCodes provides no latitudes/longitudes for gridType=%r", grid_type)
    return geo_dims, geo_shape, geo_coord_vars


def encode_cf_first(data_var_attrs, encode_cf=("parameter", "time"), time_dims=("time", "step")):
    # type: (T.MutableMapping[str, T.Any], T.Sequence[str], T.Sequence[str]) -> T.List[str]
    coords_map = ENSEMBLE_KEYS[:]
    param_id = data_var_attrs.get("GRIB_paramId", "undef")
    data_var_attrs["long_name"] = "original GRIB paramId: %s" % param_id
    data_var_attrs["units"] = "1"
    if "parameter" in encode_cf:
        if "GRIB_cfName" in data_var_attrs:
            data_var_attrs["standard_name"] = data_var_attrs["GRIB_cfName"]
        if "GRIB_name" in data_var_attrs:
            data_var_attrs["long_name"] = data_var_attrs["GRIB_name"]
        if "GRIB_units" in data_var_attrs:
            data_var_attrs["units"] = data_var_attrs["GRIB_units"]
    if "time" in encode_cf:
        coords_map.extend(time_dims)
    else:
        coords_map.extend(DATA_TIME_KEYS)
    coords_map.extend(VERTICAL_KEYS)
    coords_map.extend(SPECTRA_KEYS)
    return coords_map


def read_data_var_attrs(first: abc.Field, extra_keys: T.List[str]) -> T.Dict[str, T.Any]:
    attributes = {}
    for key in extra_keys:
        try:
            value = first[key]
            if value is not None:
                attributes["GRIB_" + key] = value
        except Exception:
            pass
    return attributes


def build_header_coords(
    index: abc.Index[T.Any, abc.Field],
    range_index: abc.Index[T.Any, abc.Field],
    coords_map: T.Sequence[str],
    data_var_attrs: T.MutableMapping[str, T.Any],
    encode_cf: T.Sequence[str],
    squeeze: bool,
    occurrence_labels: T.Optional[T.Sequence[int]],
    coords_as_attributes: T.Mapping[str, str],
    log: logging.Logger,
) -> T.Tuple[T.Dict[str, str], T.Dict[str, Variable]]:
    coord_name_key_map = {}
    coord_vars = {}
    for coord_key in coords_map:
        if coord_key.partition(":")[0] == messages.GRIB_OCCURRENCE_KEY:
            # NOTE: the occurrence axis is dataset-wide: it lists the ordinals that exist in
            #   the whole selected range, so that each variable keeps the same labels and the
            #   missing occurrences are NaN.
            if occurrence_labels is not None:
                values = list(occurrence_labels)
            else:
                values = index.get(coord_key) or range_index.get(coord_key) or []
        else:
            values = range_index[coord_key]
        if not values or (len(values) == 1 and values[0] == "undef"):
            log.debug("missing from GRIB stream: %r" % coord_key)
            continue
        orig_name = coord_key.partition(":")[0]
        coord_name = orig_name
        if (
            "vertical" in encode_cf
            and coord_name == "level"
            and "GRIB_typeOfLevel" in data_var_attrs
        ):
            coord_name = data_var_attrs["GRIB_typeOfLevel"]
        if coord_name in coords_as_attributes and len(values) == 1:
            data_var_attrs[f"GRIB_{coord_name}"] = values
            continue
        coord_name_key_map[coord_name] = coord_key
        attributes = {
            "long_name": "original GRIB coordinate for key: %s(%s)" % (orig_name, coord_name),
            "units": "1",
        }
        attributes.update(COORD_ATTRS.get(coord_name, {}).copy())
        if coord_name == messages.GRIB_OCCURRENCE_KEY and occurrence_labels is not None:
            # NOTE: use the run-wide occurrence axis so that variables with fewer occurrences
            #   keep the same labels and the missing occurrences are simply NaN.
            data = np.array(sorted(occurrence_labels), dtype=int, ndmin=1)
        else:
            data = np.array(
                sorted(values, reverse=attributes.get("stored_direction") == "decreasing")
            )
        dimensions = (coord_name,)  # type: T.Tuple[str, ...]
        # NOTE: the occurrence axis must be kept even when it has a single entry so that a
        #   duplicate is not silently collapsed back to the "first" behaviour.
        if squeeze and len(values) == 1 and coord_name != messages.GRIB_OCCURRENCE_KEY:
            data = data[0]
            dimensions = ()
        coord_vars[coord_name] = Variable(dimensions=dimensions, data=data, attributes=attributes)
    return coord_name_key_map, coord_vars


def build_cell_field_ids(
    header_dimensions: T.Tuple[str, ...],
    occurrence_values: np.ndarray,
    offsets: T.Mapping[T.Tuple[int, ...], T.List[T.Any]],
) -> T.Dict[T.Tuple[T.Any, ...], T.Dict[int, T.Any]]:
    """Map a non-spatial coordinate cell and an occurrence label to a single message id."""
    occurrence_axis = header_dimensions.index(messages.GRIB_OCCURRENCE_KEY)
    other_axes = [axis for axis in range(len(header_dimensions)) if axis != occurrence_axis]
    cell_field_ids = {}
    for header_indexes, message_ids in offsets.items():
        cell = tuple(header_indexes[axis] for axis in other_axes)
        occurrence_label = int(occurrence_values[header_indexes[occurrence_axis]])
        # NOTE: keep the first message in source order if a hidden index key splits the same
        #   coordinate cell twice.
        cell_field_ids.setdefault((cell, occurrence_label), message_ids[0])
    return cell_field_ids


def build_field_id_index(
    index: abc.Index[T.Any, abc.Field],
    coord_vars: T.Mapping[str, Variable],
    coord_name_key_map: T.Mapping[str, str],
    header_dimensions: T.Tuple[str, ...],
    extra_coords: T.Mapping[str, str],
) -> T.Tuple[T.Dict[T.Tuple[int, ...], T.List[T.Any]], T.Dict[str, T.Dict[T.Any, T.Any]]]:
    offsets = {}  # type: T.Dict[T.Tuple[int, ...], T.List[T.Union[int, T.Tuple[int, int]]]]
    header_value_index = {}
    extra_coords_data: T.Dict[str, T.Dict[str, T.Any]] = {
        coord_name: {} for coord_name in extra_coords
    }
    extra_dims = tuple(extra_coords.values())
    for dim in header_dimensions + extra_dims:
        if np.isscalar(coord_vars[dim].data):
            header_value_index[dim] = {coord_vars[dim].data.item(): 0}
        else:
            header_value_index[dim] = {v: i for i, v in enumerate(coord_vars[dim].data.tolist())}
    for header_values, message_ids in index.iter_index():
        header_indexes = []  # type: T.List[int]
        for dim in header_dimensions + extra_dims:
            header_value = header_values[index.index_keys.index(coord_name_key_map.get(dim, dim))]
            if dim in header_dimensions:
                header_indexes.append(header_value_index[dim][header_value])
            for coord_name in extra_coords:
                coord_value = header_values[
                    index.index_keys.index(coord_name_key_map.get(coord_name, coord_name))
                ]
                if dim == extra_coords[coord_name]:
                    saved_coord_value = extra_coords_data[coord_name].get(
                        header_value, coord_value
                    )
                    if saved_coord_value != coord_value:
                        raise ValueError(
                            f"'{coord_name}' cannot be indexed by dimension '{extra_coords[coord_name]}': \n"
                            f"found two '{coord_name}' distinct values ({saved_coord_value}, {coord_value}) "
                            f"for '{extra_coords[coord_name]}' value {header_value}."
                        )

                    extra_coords_data[coord_name][header_value] = coord_value
        offsets[tuple(header_indexes)] = message_ids
    return offsets, extra_coords_data


def build_variable_components(
    index: abc.Index[T.Any, abc.Field],
    encode_cf: T.Sequence[str] = (),
    filter_by_keys: T.Dict[str, T.Any] = {},
    log: logging.Logger = LOG,
    errors: str = "warn",
    squeeze: bool = True,
    read_keys: T.Iterable[str] = (),
    time_dims: T.Sequence[str] = ("time", "step"),
    extra_coords: T.Dict[str, str] = {},
    coords_as_attributes: T.Dict[str, str] = {},
    cache_geo_coords: bool = True,
    values_dtype: np.dtype = messages.DEFAULT_VALUES_DTYPE,
    occurrence_labels: T.Optional[T.Sequence[int]] = None,
    range_index: T.Optional[abc.Index[T.Any, abc.Field]] = None,
) -> T.Tuple[T.Dict[str, int], Variable, T.Dict[str, Variable]]:
    # NOTE: ``index`` holds the messages that are actually read, while ``range_index`` defines
    #   the coordinate space selected by the ordinary GRIB conditions only. They differ when a
    #   ``grib_occurrence`` filter is active: a cell missing a selected occurrence keeps its
    #   coordinates and is filled with NaN instead of being dropped.
    if range_index is None:
        range_index = index
    # NOTE: a variable may have no message at all for the selected occurrence. Its coordinates
    #   still come from ``range_index`` and its values are all NaN, so the attributes must be
    #   read from whichever index actually carries the variable headers.
    attr_index = index if len(index.field_ids_index) else range_index

    data_var_attrs = enforce_unique_attributes(attr_index, DATA_ATTRIBUTES_KEYS, filter_by_keys)
    grid_type_keys = GRID_TYPE_MAP.get(range_index.getone("gridType"), [])
    extra_keys = sorted(list(read_keys) + EXTRA_DATA_ATTRIBUTES_KEYS + grid_type_keys)
    first = range_index.first()
    extra_attrs = read_data_var_attrs(first, extra_keys)
    data_var_attrs.update(**extra_attrs)
    coords_map = encode_cf_first(
        data_var_attrs,
        encode_cf,
        time_dims,
    )
    # NOTE: the occurrence axis is present only when the index was built to keep duplicates.
    if messages.GRIB_OCCURRENCE_KEY in index.index_keys:
        coords_map.append(messages.GRIB_OCCURRENCE_KEY)
    coord_name_key_map, coord_vars = build_header_coords(
        index,
        range_index,
        coords_map,
        data_var_attrs,
        encode_cf,
        squeeze,
        occurrence_labels,
        coords_as_attributes,
        log,
    )

    # NOTE: the occurrence axis is always a dimension in "stack" mode, even when its longest
    #   run is a single message, so that squeeze never hides the fact that duplicates are kept.
    header_dimensions = tuple(
        d
        for d, c in coord_vars.items()
        if not squeeze or c.data.size > 1 or d == messages.GRIB_OCCURRENCE_KEY
    )
    header_shape = tuple(coord_vars[d].data.size for d in header_dimensions)

    gds_md5sum = range_index.get("md5GridSection")
    # If parameter is associated with a single grid definition, try to cache geometry
    if cache_geo_coords and gds_md5sum and len(gds_md5sum) == 1:
        md5sum = gds_md5sum[0]
        cache_key = (md5sum, tuple(encode_cf))
        if cache_key in GEOCACHE:
            log.debug(f"cache hit for {cache_key}; using cached geometry")
            geo_coords = GEOCACHE[cache_key]
        else:
            geo_coords = build_geography_coordinates(range_index.first(), encode_cf, errors)
            GEOCACHE[cache_key] = geo_coords
    else:
        geo_coords = build_geography_coordinates(range_index.first(), encode_cf, errors)

    geo_dims, geo_shape, geo_coord_vars = geo_coords
    dimensions = header_dimensions + geo_dims
    shape = header_shape + geo_shape
    coord_vars.update(geo_coord_vars)

    offsets, extra_coords_data = build_field_id_index(
        index, coord_vars, coord_name_key_map, header_dimensions, extra_coords
    )

    # NOTE: when the occurrence axis is present, keep a sparse mapping from a non-spatial
    #   coordinate cell (identified by its coordinate labels) plus an occurrence label to a
    #   single message id. This is proportional to the number of messages, never to the
    #   cartesian product of the header axes times the number of grid points, and it lets
    #   ``OnDiskArray`` decode only the messages that are actually selected.
    occurrence_axis = None
    occurrence_values = None
    cell_field_ids = None
    if messages.GRIB_OCCURRENCE_KEY in header_dimensions:
        occurrence_axis = header_dimensions.index(messages.GRIB_OCCURRENCE_KEY)
        run = coord_vars[messages.GRIB_OCCURRENCE_KEY].data
        occurrence_values = np.array(run, dtype=int, ndmin=1)
        cell_field_ids = build_cell_field_ids(header_dimensions, occurrence_values, offsets)

    missing_value = data_var_attrs.get("missingValue", messages.MISSING_VAUE_INDICATOR)
    on_disk_array = OnDiskArray(
        index=index,
        shape=shape,
        field_id_index=offsets,
        missing_value=missing_value,
        geo_ndim=len(geo_dims),
        dtype=values_dtype,
        occurrence_axis=occurrence_axis,
        occurrence_values=occurrence_values,
        cell_field_ids=cell_field_ids,
    )

    if "time" in coord_vars and "step" in coord_vars:
        # add the 'valid_time' secondary coordinate
        time_dims, time_data = cfmessage.build_valid_time(
            coord_vars["time"].data, coord_vars["step"].data
        )
        attrs = COORD_ATTRS["valid_time"]
        coord_vars["valid_time"] = Variable(dimensions=time_dims, data=time_data, attributes=attrs)

    for coord_name in extra_coords:
        coord_data = np.array(list(extra_coords_data[coord_name].values()))
        if extra_coords[coord_name] not in header_dimensions:
            coord_dims: T.Tuple[str, ...] = ()
            coord_data = coord_data.reshape(())
        else:
            coord_dims = (extra_coords[coord_name],)
        attrs = COORD_ATTRS.get(coord_name, {}).copy()
        coord_vars[coord_name] = Variable(dimensions=coord_dims, data=coord_data, attributes=attrs)

    data_var_attrs["coordinates"] = " ".join(coord_vars.keys())
    # OnDiskArray is close enough to np.ndarray to work, but not to make mypy happy
    data_var = Variable(dimensions=dimensions, data=on_disk_array, attributes=data_var_attrs)  # type: ignore
    dims = {d: s for d, s in zip(dimensions, data_var.data.shape)}
    return dims, data_var, coord_vars


def dict_merge(master, update):
    # type: (T.Dict[str, T.Any], T.Dict[str, T.Any]) -> None
    for key, value in update.items():
        if key not in master:
            master[key] = value
        elif master[key] == value:
            pass
        else:
            raise DatasetBuildError(
                "key present and new value is different: "
                "key=%r value=%r new_value=%r" % (key, master[key], value)
            )


def build_dataset_attributes(index, filter_by_keys, encoding):
    # type: (abc.Index[T.Any, abc.Field], T.Dict[str, T.Any], T.Dict[str, T.Any]) -> T.Dict[str, T.Any]
    attributes = enforce_unique_attributes(index, GLOBAL_ATTRIBUTES_KEYS, filter_by_keys)
    attributes["Conventions"] = "CF-1.7"
    if "GRIB_centreDescription" in attributes:
        attributes["institution"] = attributes["GRIB_centreDescription"]
    attributes_namespace = {
        "cfgrib_version": __version__,
        "cfgrib_open_kwargs": json.dumps(encoding),
        "eccodes_version": messages.eccodes_version,
        "timestamp": datetime.datetime.now().isoformat().partition(".")[0][:16],
    }
    history_in = (
        "{timestamp} GRIB to CDM+CF via "
        "cfgrib-{cfgrib_version}/ecCodes-{eccodes_version} with {cfgrib_open_kwargs}"
    )
    attributes["history"] = history_in.format(**attributes_namespace)
    return attributes


def compute_occurrence_labels(
    index: abc.Index[T.Any, abc.Field],
) -> T.Optional[T.List[int]]:
    """Return the ordinal labels of the occurrence axis shared by a whole dataset."""
    # NOTE: the labels are the ordinals that both exist in the selected range and were
    #   selected, so they are read from the filtered index whenever it still has messages.
    range_index = getattr(index, "range_index", None)
    fallback_index = range_index if range_index is not None else index
    axis_index = index if index.field_ids_index else fallback_index
    if messages.GRIB_OCCURRENCE_KEY not in axis_index.index_keys:
        return None
    if index.field_ids_index:
        return sorted(set(index[messages.GRIB_OCCURRENCE_KEY]))
    # NOTE: the variable has no message for any selected occurrence, but it must not be
    #   dropped: keep the selected ordinals and fill the whole variable with NaN.
    selected = index.filter_by_keys.get(messages.GRIB_OCCURRENCE_KEY)
    if selected is None:
        selected = axis_index.get(messages.GRIB_OCCURRENCE_KEY, [])
    return sorted(set(selected))


def build_dataset_components(
    index: abc.Index[T.Any, abc.Field],
    errors: str = "warn",
    encode_cf: T.Sequence[str] = ("parameter", "time", "geography", "vertical"),
    squeeze: bool = True,
    log: logging.Logger = LOG,
    read_keys: T.Iterable[str] = (),
    time_dims: T.Sequence[str] = ("time", "step"),
    extra_coords: T.Dict[str, str] = {},
    coords_as_attributes: T.Dict[str, str] = {},
    cache_geo_coords: bool = True,
    values_dtype: np.dtype = messages.DEFAULT_VALUES_DTYPE,
) -> T.Tuple[T.Dict[str, int], T.Dict[str, Variable], T.Dict[str, T.Any], T.Dict[str, T.Any]]:
    dimensions = {}  # type: T.Dict[str, int]
    variables = {}  # type: T.Dict[str, Variable]
    filter_by_keys = index.filter_by_keys

    # Warn about time_dims here to prevent repeasted messages in build_variable_components
    if errors != "ignore" and not set(time_dims).issubset(ALL_REF_TIME_KEYS):
        log.warning(
            "Not all time_dimensions are recognised, those which are not in the following list will not "
            " be decoded as datetime objects:\n"
            f"{ALL_REF_TIME_KEYS}"
        )

    # NOTE: the occurrence axis is shared by all the variables in the dataset, so that a
    #   variable with fewer occurrences keeps the dataset-wide labels and is filled with NaN
    #   where it has no message instead of losing the axis or being renumbered.
    range_index = getattr(index, "range_index", None)
    occurrence_labels = compute_occurrence_labels(index)

    # NOTE: with a ``grib_occurrence`` filter the coordinate range comes from the index of all
    #   the messages selected by the ordinary conditions, so variables without any message for
    #   the selected occurrence are still represented as an all-NaN variable.
    param_index = range_index if range_index is not None else index
    param_ids = sorted(set(param_index.get("paramId", [])))

    for param_id in param_ids:
        var_index = index.subindex(paramId=param_id)
        try:
            dims, data_var, coord_vars = build_variable_components(
                var_index,
                encode_cf,
                filter_by_keys,
                errors=errors,
                squeeze=squeeze,
                read_keys=read_keys,
                time_dims=time_dims,
                extra_coords=extra_coords,
                coords_as_attributes=coords_as_attributes,
                cache_geo_coords=cache_geo_coords,
                values_dtype=values_dtype,
                occurrence_labels=occurrence_labels,
                range_index=getattr(var_index, "range_index", None),
            )
        except DatasetBuildError as ex:
            # NOTE: When a variable has more than one value for an attribute we need to raise all
            #   the values in the file, not just the ones associated with that variable. See #54.
            key = ex.args[1]
            error_message = "multiple values for unique key, try re-open the file with one of:"
            fbks = []
            for value in index[key]:
                fbk = {key: value}
                fbk.update(filter_by_keys)
                fbks.append(fbk)
                error_message += "\n    filter_by_keys=%r" % fbk
            raise DatasetBuildError(error_message, key, fbks)
        short_name = data_var.attributes.get("GRIB_shortName", "paramId_%d" % param_id)
        var_name = data_var.attributes.get("GRIB_cfVarName", "unknown")
        if "parameter" in encode_cf and var_name not in ("undef", "unknown"):
            short_name = var_name
        try:
            dict_merge(variables, coord_vars)
            dict_merge(variables, {short_name: data_var})
            dict_merge(dimensions, dims)
        except ValueError:
            if errors == "ignore":
                pass
            elif errors == "raise":
                raise
            else:
                log.exception("skipping variable: paramId==%r shortName=%r", param_id, short_name)
    encoding = {"source": index.source(), "filter_by_keys": filter_by_keys, "encode_cf": encode_cf}
    attributes = build_dataset_attributes(index, filter_by_keys, encoding)
    return dimensions, variables, attributes, encoding


@attr.attrs(auto_attribs=True)
class Dataset:
    """
    Map a GRIB file to the NetCDF Common Data Model with CF Conventions.
    """

    dimensions: T.Dict[str, int]
    variables: T.Dict[str, Variable]
    attributes: T.Dict[str, T.Any]
    encoding: T.Dict[str, T.Any]


def compute_index_keys(
    time_dims: T.Sequence[str] = ("time", "step"),
    extra_coords: T.Dict[str, str] = {},
    filter_by_keys: T.Dict[str, T.Any] = {},
    duplicate_messages: T.Any = messages.DUPLICATE_MESSAGES_FIRST,
) -> T.List[str]:
    keys = set(INDEX_KEYS) | {
        k for k in filter_by_keys if k.partition(":")[0] != messages.GRIB_OCCURRENCE_KEY
    }
    keys |= set(time_dims) | set(extra_coords)
    mode = messages.resolve_duplicate_messages(duplicate_messages)
    if mode == messages.DUPLICATE_MESSAGES_STACK:
        keys.add(messages.GRIB_OCCURRENCE_KEY)
    return sorted(keys)


def open_from_index(
    index: abc.Index[T.Any, abc.Field],
    read_keys: T.Sequence[str] = (),
    time_dims: T.Sequence[str] = ("time", "step"),
    extra_coords: T.Dict[str, str] = {},
    **kwargs: T.Any,
) -> Dataset:
    dimensions, variables, attributes, encoding = build_dataset_components(
        index, read_keys=read_keys, time_dims=time_dims, extra_coords=extra_coords, **kwargs
    )
    return Dataset(dimensions, variables, attributes, encoding)


def _prepare_indexing(
    filter_by_keys: T.Dict[str, T.Any], duplicate_messages: T.Any
) -> T.Tuple[str, T.Dict[str, T.Any]]:
    # NOTE: ``messages.normalize_filter_by_keys`` validates the mode and the occurrence filter
    #   together, so that ``first`` mode can never silently ignore a ``grib_occurrence`` query.
    return messages.normalize_filter_by_keys(filter_by_keys, duplicate_messages)


def open_fieldset(
    fieldset: T.Union[abc.Fieldset[abc.Field], abc.MappingFieldset[T.Any, abc.Field]],
    indexpath: T.Optional[str] = None,
    filter_by_keys: T.Dict[str, T.Any] = {},
    read_keys: T.Sequence[str] = (),
    ignore_keys: T.Sequence[str] = [],
    time_dims: T.Sequence[str] = ("time", "step"),
    extra_coords: T.Dict[str, str] = {},
    computed_keys: messages.ComputedKeysType = cfmessage.COMPUTED_KEYS,
    duplicate_messages: str = messages.DUPLICATE_MESSAGES_FIRST,
    strict_occurrence: bool = True,
    log: logging.Logger = LOG,
    **kwargs: T.Any,
) -> Dataset:
    """Builds a ``cfgrib.Dataset`` form a mapping of mappings."""
    if indexpath is not None and indexpath is not messages.DEFAULT_INDEXPATH:
        log.warning(f"indexpath value {indexpath} is ignored")

    mode, filter_by_keys = _prepare_indexing(filter_by_keys, duplicate_messages)
    index_keys = compute_index_keys(time_dims, extra_coords, filter_by_keys, mode)
    index_keys = [key for key in index_keys if key not in ignore_keys]
    index = messages.FieldsetIndex.from_fieldset(fieldset, index_keys, computed_keys)
    filtered_index = index.subindex(filter_by_keys, strict_occurrence=strict_occurrence)
    return open_from_index(filtered_index, read_keys, time_dims, extra_coords, **kwargs)


def open_fileindex(
    stream: messages.FileStream,
    indexpath: str = messages.DEFAULT_INDEXPATH,
    index_keys: T.Sequence[str] = INDEX_KEYS + ["time", "step"],
    ignore_keys: T.Sequence[str] = [],
    filter_by_keys: T.Dict[str, T.Any] = {},
    computed_keys: messages.ComputedKeysType = cfmessage.COMPUTED_KEYS,
    duplicate_messages: T.Any = messages.DUPLICATE_MESSAGES_FIRST,
    strict_occurrence: bool = True,
) -> messages.FileIndex:
    mode, filter_by_keys = _prepare_indexing(filter_by_keys, duplicate_messages)
    keys = set(index_keys) | {
        k for k in filter_by_keys if k.partition(":")[0] != messages.GRIB_OCCURRENCE_KEY
    }
    if mode == messages.DUPLICATE_MESSAGES_STACK:
        keys.add(messages.GRIB_OCCURRENCE_KEY)
    index_keys = [key for key in sorted(keys) if key not in ignore_keys]
    index = messages.FileIndex.from_indexpath_or_filestream(
        stream, index_keys, indexpath=indexpath, computed_keys=computed_keys
    )
    return index.subindex(filter_by_keys, strict_occurrence=strict_occurrence)


def open_file(
    path: T.Union[str, "os.PathLike[str]"],
    errors: str = "warn",
    indexpath: str = messages.DEFAULT_INDEXPATH,
    filter_by_keys: T.Dict[str, T.Any] = {},
    read_keys: T.Sequence[str] = (),
    time_dims: T.Sequence[str] = ("time", "step"),
    extra_coords: T.Dict[str, str] = {},
    ignore_keys: T.Sequence[str] = [],
    duplicate_messages: T.Any = messages.DUPLICATE_MESSAGES_FIRST,
    strict_occurrence: bool = True,
    **kwargs: T.Any,
) -> Dataset:
    """Open a GRIB file as a ``cfgrib.Dataset``."""
    path = os.fspath(path)
    stream = messages.FileStream(path, errors=errors)
    mode, filter_by_keys = _prepare_indexing(filter_by_keys, duplicate_messages)
    index_keys = compute_index_keys(time_dims, extra_coords, filter_by_keys, mode)
    index = open_fileindex(
        stream,
        indexpath,
        index_keys,
        ignore_keys=ignore_keys,
        filter_by_keys=filter_by_keys,
        duplicate_messages=mode,
        strict_occurrence=strict_occurrence,
    )
    return open_from_index(index, read_keys, time_dims, extra_coords, errors=errors, **kwargs)
