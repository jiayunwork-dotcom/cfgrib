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
"""Exact totals over arbitrary time intervals from GRIB2 accumulation messages.

A ``stepType=accum`` message stores the total accumulated between two steps that
are relative to *its own* reference time (the interval is left-open/right-closed).
Turning a set of such messages into the total over an arbitrary target interval is
a small linear-algebra problem: every message constrains the difference between an
unknown cumulative field at its two endpoints, and every target is another such
difference.

A target is only returned when it is the same in *every* assignment of cumulative
values that reproduces the observations exactly.  No proportional splitting, no
assumption of a uniform rate and no minimum-norm (or other) preference is used to
fill in the remaining degrees of freedom.

For each reference time and grid point the finite observations form an unweighted
graph whose vertices are interval endpoints and whose edges are observed totals.
Least-squares potentials on that graph give the observations' residual: if a
single raw observation has an absolute residual above ``RESIDUAL_TOLERANCE`` then
*all* targets at that point and reference time are missing.  A target is uniquely
determined exactly when its two endpoints lie in the same connected component of
the observed graph (its potential difference is then gauge independent).

Every grid point whose missing observations follow the same pattern shares the
same linear algebra: the least-squares projector and the target coefficients are
computed once per pattern and applied to all of those points at once.  Values are
decoded in chunks whose total size is at most one spatial field, so no
``messages x grid`` resident copy of the field values is ever materialised, and
reading a single reference time never decodes messages of the other ones.
"""

import collections
import datetime
import logging
import os
import typing as T

import attr
import eccodes  # type: ignore
import numpy as np

from . import dataset, messages

try:  # pragma: no cover - xarray is an optional dependency
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None  # type: ignore

if xr is not None:  # pragma: no cover - exercised only when xarray is missing
    _BackendArrayBase = xr.backends.BackendArray
else:  # pragma: no cover

    class _BackendArrayBase:  # type: ignore[no-redef]
        """Placeholder base class so the module imports without xarray."""


LOG = logging.getLogger(__name__)

#: Absolute residual (in the physical units of the field) above which every target
#: at the affected grid point and reference time is considered unusable.
RESIDUAL_TOLERANCE = 1e-8

#: GRIB2 code table 4.4 time units accepted for accumulation endpoints.
SUPPORTED_STEP_UNITS = {
    0: 60,  # minute
    1: 3600,  # hour
    2: 86400,  # day
    13: 1,  # second
}

#: dataType values that identify a deterministic (non-ensemble) *forecast*.
#: ``an`` (analysis) is a deterministic product but not a forecast, so it is
#: rejected: its reference time is not a forecast initialisation time.
DETERMINISTIC_DATA_TYPES = frozenset({"fc", "af"})

#: GRIB2 product definition template for an accumulation over one time range.
ACCUMULATION_PDT = 8

EPOCH = datetime.datetime(1970, 1, 1)

INTERVAL_INDEX_KEYS = sorted(
    {
        "edition",
        "gridType",
        "typeOfLevel",
        "dataType",
        "typeOfProcessedData",
        "stepType",
        "numberOfTimeRange",
        "productDefinitionTemplateNumber",
        "typeOfStatisticalProcessing",
        "indicatorOfUnitOfTimeRange",
        "indicatorOfUnitForTimeRange",
        "forecastTime",
        "lengthOfTimeRange",
        "startStep",
        "endStep",
        "dataDate",
        "dataTime",
        "paramId",
        "shortName",
        "units",
        "name",
        "cfName",
        "cfVarName",
        "md5GridSection",
        "numberOfPoints",
        "Nx",
        "Ny",
        "Ni",
        "Nj",
        "iDirectionIncrementInDegrees",
        "jDirectionIncrementInDegrees",
        "iScansNegatively",
        "jScansPositively",
        "jPointsAreConsecutive",
        "alternativeRowScanning",
        "longitudeOfFirstGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "latitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "missingValue",
        "bitmapPresent",
        "centre",
        "subCentre",
        "discipline",
        "parameterCategory",
        "parameterNumber",
    }
)

BACKEND_KWARGS_ALLOWED = frozenset({"filter_by_keys", "indexpath", "errors"})

#: Attributes copied from the source messages onto the result data variable.
DATA_VAR_GRIB_KEYS = [
    "shortName",
    "units",
    "name",
    "cfName",
    "cfVarName",
    "paramId",
    "dataType",
    "typeOfLevel",
    "gridType",
    "missingValue",
    "numberOfPoints",
    "Nx",
    "Ny",
    "Ni",
    "Nj",
    "iDirectionIncrementInDegrees",
    "iDirectionIncrement",
    "jDirectionIncrementInDegrees",
    "jDirectionIncrement",
    "iScansNegatively",
    "jScansPositively",
    "jPointsAreConsecutive",
    "longitudeOfFirstGridPointInDegrees",
    "longitudeOfLastGridPointInDegrees",
    "latitudeOfFirstGridPointInDegrees",
    "latitudeOfLastGridPointInDegrees",
    "bitmapPresent",
]


class IntervalTotalsError(ValueError):
    """Raised for invalid arguments or for non-conforming GRIB headers."""


#
# Interval argument validation
#


def _endpoint_seconds(value: T.Any, position: int) -> int:
    """Convert one user supplied interval endpoint to a non-negative integer second."""
    if isinstance(value, bool):
        raise IntervalTotalsError("interval endpoint %d must be a timedelta, got bool %r" % (position, value))
    if isinstance(value, np.timedelta64):
        if np.isnat(value):
            raise IntervalTotalsError("interval endpoint %d must not be NaT" % position)
        try:
            seconds = value / np.timedelta64(1, "s")
        except (TypeError, ValueError):
            raise IntervalTotalsError("interval endpoint %d cannot be converted to seconds: %r" % (position, value))
        if not float(seconds).is_integer():
            raise IntervalTotalsError("interval endpoint %d is not a whole number of seconds: %r" % (position, value))
        seconds_int = int(seconds)
    elif isinstance(value, datetime.timedelta):
        total = value.total_seconds()
        if not total.is_integer():
            raise IntervalTotalsError("interval endpoint %d is not a whole number of seconds: %r" % (position, value))
        seconds_int = int(total)
    else:
        raise IntervalTotalsError(
            "interval endpoint %d must be numpy.timedelta64 or datetime.timedelta, got %r" % (position, type(value).__name__)
        )
    if seconds_int < 0:
        raise IntervalTotalsError("interval endpoint %d must be non-negative, got %d seconds" % (position, seconds_int))
    return seconds_int


def normalize_intervals(intervals: T.Any) -> T.List[T.Tuple[int, int]]:
    """Validate ``intervals`` and return ``(start, end)`` integer seconds pairs."""
    if intervals is None or isinstance(intervals, (str, bytes)):
        raise IntervalTotalsError("intervals must be a non-empty sequence of (start, end) pairs")
    try:
        items = list(intervals)
    except TypeError:
        raise IntervalTotalsError("intervals must be a non-empty sequence of (start, end) pairs")
    if not items:
        raise IntervalTotalsError("intervals must not be empty")
    normalized = []
    seen = set()
    for position, item in enumerate(items):
        if isinstance(item, (str, bytes)):
            raise IntervalTotalsError("interval %d must be a (start, end) pair" % position)
        try:
            pair = list(item)
        except TypeError:
            raise IntervalTotalsError("interval %d must be a (start, end) pair" % position)
        if len(pair) != 2:
            raise IntervalTotalsError("interval %d must contain exactly two endpoints, got %d" % (position, len(pair)))
        start = _endpoint_seconds(pair[0], 2 * position)
        end = _endpoint_seconds(pair[1], 2 * position + 1)
        if start >= end:
            raise IntervalTotalsError(
                "interval %d start must be strictly smaller than its end, got (%d, %d)" % (position, start, end)
            )
        if (start, end) in seen:
            raise IntervalTotalsError("duplicate target interval (%d, %d) at position %d" % (start, end, position))
        seen.add((start, end))
        normalized.append((start, end))
    return normalized


def _validate_backend_kwargs(backend_kwargs: T.Optional[T.Mapping[str, T.Any]]) -> T.Dict[str, T.Any]:
    if backend_kwargs is None:
        return {"filter_by_keys": {}, "indexpath": messages.DEFAULT_INDEXPATH, "errors": "warn"}
    if not isinstance(backend_kwargs, T.Mapping):
        raise IntervalTotalsError("backend_kwargs must be a mapping or None")
    kwargs = dict(backend_kwargs)
    unknown = set(kwargs) - BACKEND_KWARGS_ALLOWED
    if unknown:
        raise IntervalTotalsError(
            "unsupported backend_kwargs %r, supported are %r" % (sorted(unknown), sorted(BACKEND_KWARGS_ALLOWED))
        )
    errors = kwargs.get("errors", "warn")
    if errors not in ("ignore", "warn", "raise"):
        raise IntervalTotalsError("backend_kwargs['errors'] must be 'ignore', 'warn' or 'raise'")
    indexpath = kwargs.get("indexpath", messages.DEFAULT_INDEXPATH)
    if not isinstance(indexpath, str):
        raise IntervalTotalsError("backend_kwargs['indexpath'] must be a string")
    filter_by_keys = kwargs.get("filter_by_keys", {})
    if not isinstance(filter_by_keys, T.Mapping):
        raise IntervalTotalsError("backend_kwargs['filter_by_keys'] must be a mapping")
    if messages.grib_occurrence_filter_key(filter_by_keys) is not None:
        raise IntervalTotalsError("backend_kwargs['filter_by_keys'] does not support grib_occurrence")
    return {"filter_by_keys": dict(filter_by_keys), "indexpath": indexpath, "errors": errors}


# Extended headers used to catch semantic duplicates and to validate products.
INTERVAL_INDEX_KEYS = sorted(
    set(INTERVAL_INDEX_KEYS)
    | {
        "typeOfGeneratingProcess",
        "number",
        "perturbationNumber",
        "derivedForecast",
        "numberOfForecastsInEnsemble",
    }
)

GRID_SIGNATURE_KEYS = [
    "gridType",
    "md5GridSection",
    "numberOfPoints",
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
    "alternativeRowScanning",
]

QUANTITY_SIGNATURE_KEYS = [
    "edition",
    "discipline",
    "parameterCategory",
    "parameterNumber",
    "paramId",
    "shortName",
    "units",
    "typeOfLevel",
]

# The product family and grid this entry point is restricted to.
REQUIRED_TYPE_OF_LEVEL = "surface"
REQUIRED_GRID_TYPE = "regular_ll"


def _norm(value: T.Any) -> T.Any:
    """Normalise a header value so that it is hashable and comparable."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if value is None:
        return "undef"
    if isinstance(value, (list, tuple)):
        return tuple(_norm(item) for item in value)
    return value


def _signature(headers: T.Mapping[str, T.Any], keys: T.Sequence[str]) -> T.Tuple[T.Any, ...]:
    return tuple((key, _norm(headers.get(key, "undef"))) for key in keys)


@attr.attrs(auto_attribs=True)
class AccumulationMessage:
    """One selected GRIB2 accumulation message, described by its header only."""

    message_id: T.Any
    reference_seconds: int
    start_seconds: int
    end_seconds: int
    headers: T.Dict[str, T.Any]
    source_position: int

    @property
    def interval_key(self) -> T.Tuple[int, int]:
        return (self.start_seconds, self.end_seconds)


@attr.attrs(auto_attribs=True)
class ReferenceGroup:
    """All accumulation constraints that share one reference time."""

    reference_seconds: int
    messages: T.List[AccumulationMessage]
    vertices: T.List[int]
    edges: T.List[T.Tuple[int, int]]  # (start vertex, end vertex) per message
    targets: T.List[T.Tuple[int, int]]  # (start vertex, end vertex) per target
    solver_cache: T.MutableMapping[bytes, T.Any] = attr.attrib(factory=lambda: collections.OrderedDict(), repr=False)
    subsets: T.MutableMapping[bytes, "ReferenceGroup"] = attr.attrib(factory=dict, repr=False, eq=False)


@attr.attrs(auto_attribs=True)
class IntervalPlan:
    """Header-only description of one ``interval_totals`` request."""

    path: str
    errors: str
    intervals: T.List[T.Tuple[int, int]]
    index: T.Any
    messages: T.List[AccumulationMessage]
    groups: T.List[ReferenceGroup]
    geo_dims: T.Tuple[str, ...]
    geo_shape: T.Tuple[int, ...]
    geo_coord_vars: T.Dict[str, "dataset.Variable"]
    data_var_attrs: T.Dict[str, T.Any]
    dataset_attrs: T.Dict[str, T.Any]
    variable_name: str


#
# Header validation
#


def _integer_header(headers: T.Mapping[str, T.Any], key: str, position: int) -> int:
    value = headers.get(key, "undef")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise IntervalTotalsError("message %d: expected an integer %r header, got %r" % (position, key, value))
    return int(value)


def _string_header(headers: T.Mapping[str, T.Any], key: str, position: int) -> str:
    value = headers.get(key, "undef")
    if not isinstance(value, str) or value in ("undef", "unknown"):
        raise IntervalTotalsError("message %d: expected a %r header, got %r" % (position, key, value))
    return value


def _reference_seconds(headers: T.Mapping[str, T.Any], position: int) -> int:
    date = _integer_header(headers, "dataDate", position)
    time = _integer_header(headers, "dataTime", position)
    year, month, day = date // 10000, date // 100 % 100, date % 100
    hour, minute = time // 100, time % 100
    try:
        reference = datetime.datetime(year, month, day, hour, minute)
    except ValueError as ex:
        raise IntervalTotalsError("message %d: invalid reference time: %s" % (position, ex))
    return int((reference - EPOCH).total_seconds())


def _interval_seconds(headers: T.Mapping[str, T.Any], position: int) -> T.Tuple[int, int]:
    """Return the ``(start, end)`` seconds of an accumulation message.

    Both endpoints use the *numeric* GRIB2 time-unit indicators, never the human
    readable ``stepUnits`` concept, so the interpretation does not depend on any
    naming convention.
    """
    start_unit = _integer_header(headers, "indicatorOfUnitOfTimeRange", position)
    length_unit = _integer_header(headers, "indicatorOfUnitForTimeRange", position)
    for unit in (start_unit, length_unit):
        if unit not in SUPPORTED_STEP_UNITS:
            raise IntervalTotalsError(
                "message %d: unsupported GRIB2 time unit indicator %r, supported are %r"
                % (position, unit, sorted(SUPPORTED_STEP_UNITS))
            )
    forecast_time = _integer_header(headers, "forecastTime", position)
    length = _integer_header(headers, "lengthOfTimeRange", position)
    if forecast_time < 0 or length < 0:
        raise IntervalTotalsError(
            "message %d: accumulation endpoints must be non-negative, got (%d, %d)" % (position, forecast_time, length)
        )
    start_seconds = forecast_time * SUPPORTED_STEP_UNITS[start_unit]
    end_seconds = start_seconds + length * SUPPORTED_STEP_UNITS[length_unit]
    if end_seconds <= start_seconds:
        raise IntervalTotalsError(
            "message %d: accumulation interval must have a positive length, got (%d, %d)"
            % (position, start_seconds, end_seconds)
        )
    return start_seconds, end_seconds


def _require_product_identity(headers: T.Mapping[str, T.Any], position: int) -> None:
    """Check the GRIB edition, grid and vertical level the entry point supports."""
    edition = _integer_header(headers, "edition", position)
    if edition != 2:
        raise IntervalTotalsError("message %d: only GRIB2 is supported, found edition %r" % (position, edition))
    grid_type = _string_header(headers, "gridType", position)
    if grid_type != REQUIRED_GRID_TYPE:
        raise IntervalTotalsError(
            "message %d: only gridType=%r is supported, found %r" % (position, REQUIRED_GRID_TYPE, grid_type)
        )
    level_type = _string_header(headers, "typeOfLevel", position)
    if level_type != REQUIRED_TYPE_OF_LEVEL:
        raise IntervalTotalsError(
            "message %d: only typeOfLevel=%r is supported, found %r" % (position, REQUIRED_TYPE_OF_LEVEL, level_type)
        )


def _require_accumulation_template(headers: T.Mapping[str, T.Any], position: int) -> None:
    """Check the product definition describes exactly one accumulation range."""
    step_type = _string_header(headers, "stepType", position)
    if step_type != "accum":
        raise IntervalTotalsError("message %d: only stepType='accum' is supported, found %r" % (position, step_type))
    number_of_time_ranges = _integer_header(headers, "numberOfTimeRange", position)
    if number_of_time_ranges != 1:
        raise IntervalTotalsError(
            "message %d: only a single statistical time range is supported, found %d" % (position, number_of_time_ranges)
        )
    template = headers.get("productDefinitionTemplateNumber", "undef")
    if _norm(template) != ACCUMULATION_PDT:
        raise IntervalTotalsError(
            "message %d: only productDefinitionTemplateNumber=%d is supported, found %r"
            % (position, ACCUMULATION_PDT, template)
        )
    statistical_processing = _integer_header(headers, "typeOfStatisticalProcessing", position)
    if statistical_processing != 1:
        raise IntervalTotalsError(
            "message %d: only accumulation (typeOfStatisticalProcessing=1) is supported, "
            "found %r" % (position, statistical_processing)
        )


def _require_deterministic_product(headers: T.Mapping[str, T.Any], position: int) -> None:
    """Check the message is a deterministic forecast, not an ensemble product."""
    data_type = headers.get("dataType", "undef")
    if data_type not in DETERMINISTIC_DATA_TYPES:
        raise IntervalTotalsError(
            "message %d: only deterministic forecast products are supported, found dataType=%r" % (position, data_type)
        )
    for ensemble_key in ("number", "perturbationNumber", "numberOfForecastsInEnsemble"):
        if headers.get(ensemble_key, "undef") not in ("undef", None):
            raise IntervalTotalsError(
                "message %d: ensemble products are not supported (%r is present)" % (position, ensemble_key)
            )
    for scanning_key in ("jPointsAreConsecutive", "alternativeRowScanning"):
        if _norm(headers.get(scanning_key, 0)) != 0:
            raise IntervalTotalsError("message %d: %s=1 is not supported" % (position, scanning_key))


def _validate_product(headers: T.Mapping[str, T.Any], position: int) -> None:
    _require_product_identity(headers, position)
    _require_accumulation_template(headers, position)
    _require_deterministic_product(headers, position)


def _open_index(path: str, filter_by_keys: T.Mapping[str, T.Any], indexpath: str, errors: str):
    stream = messages.FileStream(path, errors=errors)
    keys = set(INTERVAL_INDEX_KEYS)
    keys |= {key for key in filter_by_keys if key.partition(":")[0] != messages.GRIB_OCCURRENCE_KEY}
    index = messages.FileIndex.from_indexpath_or_filestream(
        stream,
        sorted(keys),
        indexpath=indexpath,
        computed_keys={},
    )
    return index.subindex(filter_by_keys, strict_occurrence=False)


def _parse_messages(index: T.Any) -> T.List[AccumulationMessage]:
    """Validate headers of every selected message, without decoding any value."""
    parsed: T.List[AccumulationMessage] = []
    for cell_number, (header_values, message_ids) in enumerate(index.iter_index()):
        headers = dict(zip(index.index_keys, header_values))
        position = cell_number
        _validate_product(headers, position)
        reference = _reference_seconds(headers, position)
        start_seconds, end_seconds = _interval_seconds(headers, position)
        if len(message_ids) != 1:
            raise IntervalTotalsError(
                "message %d: the accumulation interval %d-%d at reference time %d is "
                "carried by %d messages; refusing to choose one of them"
                % (position, start_seconds, end_seconds, reference, len(message_ids))
            )
        parsed.append(
            AccumulationMessage(
                message_id=message_ids[0],
                reference_seconds=reference,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                headers=headers,
                source_position=position,
            )
        )
    if not parsed:
        raise IntervalTotalsError("no message matched the request")
    return parsed


def _check_single_product_and_grid(
    parsed: T.Sequence[AccumulationMessage],
) -> T.Tuple[T.Tuple[T.Any, ...], T.Tuple[T.Any, ...]]:
    quantities = {_signature(message.headers, QUANTITY_SIGNATURE_KEYS) for message in parsed}
    if len(quantities) != 1:
        raise IntervalTotalsError(
            "the selected messages must contain a single physical quantity, found %d: %r"
            % (len(quantities), sorted(str(q) for q in quantities))
        )
    grids = {_signature(message.headers, GRID_SIGNATURE_KEYS) for message in parsed}
    if len(grids) != 1:
        raise IntervalTotalsError("the selected messages must contain a single spatial grid, found %d" % len(grids))
    return quantities.pop(), grids.pop()


def _check_duplicates(
    parsed: T.Sequence[AccumulationMessage],
    quantity: T.Tuple[T.Any, ...],
    grid: T.Tuple[T.Any, ...],
) -> None:
    seen: T.Dict[T.Tuple[T.Any, ...], AccumulationMessage] = {}
    for message in parsed:
        key = (
            message.reference_seconds,
            quantity,
            grid,
            message.start_seconds,
            message.end_seconds,
        )
        if key in seen:
            raise IntervalTotalsError(
                "duplicate accumulation message for reference time %d and interval %d-%d"
                % (message.reference_seconds, message.start_seconds, message.end_seconds)
            )
        seen[key] = message


def _build_groups(parsed: T.Sequence[AccumulationMessage], intervals: T.Sequence[T.Tuple[int, int]]) -> T.List[ReferenceGroup]:
    by_reference: T.Dict[int, T.List[AccumulationMessage]] = {}
    for message in parsed:
        by_reference.setdefault(message.reference_seconds, []).append(message)
    groups = []
    for reference in sorted(by_reference):
        group_messages = by_reference[reference]
        vertex_values = set()
        for message in group_messages:
            vertex_values.add(message.start_seconds)
            vertex_values.add(message.end_seconds)
        for start, end in intervals:
            vertex_values.add(start)
            vertex_values.add(end)
        vertices = sorted(vertex_values)
        vertex_number = {value: index for index, value in enumerate(vertices)}
        edges = [(vertex_number[message.start_seconds], vertex_number[message.end_seconds]) for message in group_messages]
        targets = [(vertex_number[start], vertex_number[end]) for start, end in intervals]
        groups.append(
            ReferenceGroup(
                reference_seconds=reference,
                messages=list(group_messages),
                vertices=vertices,
                edges=edges,
                targets=targets,
            )
        )
    return groups


#
# Exact interval solver
#
# Every finite observation of one reference time and one grid point is an edge
# ``potential(end) - potential(start) == observed``.  With ``B`` the signed
# incidence matrix of the observed edges and ``y`` the observed values, the
# least-squares potentials are ``p = pinv(B) y`` and the raw residuals are
# ``B p - y = (B pinv(B) - I) y``.
#
# A target between vertices ``s`` and ``t`` is the linear functional ``c . p``
# with ``c`` a difference of two unit vectors.  It is the same in *every*
# minimum-residual assignment exactly when ``c`` is in the row space of ``B``,
# i.e. when ``s`` and ``t`` are connected by observed edges.  Its value is then
# the difference of the potentials and therefore gauge independent.  When the
# endpoints are not connected, the two components can slide with respect to each
# other while keeping the same residual, so no value is returned.
#


def _incidence(edges: T.Sequence[T.Tuple[int, int]], n_vertices: int) -> np.ndarray:
    incidence = np.zeros((len(edges), n_vertices), dtype=np.float64)
    for edge_index, (start, end) in enumerate(edges):
        incidence[edge_index, start] = -1.0
        incidence[edge_index, end] = 1.0
    return incidence


def _components(n_vertices: int, edges: T.Sequence[T.Tuple[int, int]]) -> np.ndarray:
    parent = list(range(n_vertices))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for start, end in edges:
        root_start, root_end = find(start), find(end)
        if root_start != root_end:
            parent[root_start] = root_end
    return np.array([find(item) for item in range(n_vertices)], dtype=np.int64)


@attr.attrs(auto_attribs=True)
class PatternSolver:
    """Linear algebra for one pattern of finite observations on one reference time.

    ``finite`` is a boolean mask over the group's messages: ``True`` marks the
    observations that constrain the solution at the affected grid points.  The
    arrays are shared by every grid point that has exactly this pattern.
    """

    finite: np.ndarray
    finite_indexes: np.ndarray
    coefficient: np.ndarray  # (n_targets, n_finite): identifiability + target value
    identifiable: np.ndarray  # (n_targets,) bool
    residual: np.ndarray  # (n_finite, n_finite): residual = residual @ observed

    @property
    def key(self) -> bytes:
        return np.packbits(self.finite).tobytes()


def build_pattern_solver(group: "ReferenceGroup", finite: np.ndarray) -> T.Optional[PatternSolver]:
    """Build the exact-solution operator for one observation pattern.

    Returns ``None`` when the pattern carries no observation at all.
    """
    n_messages = len(group.messages)
    n_vertices = len(group.vertices)
    finite = np.asarray(finite, dtype=bool)
    if finite.shape != (n_messages,):  # pragma: no cover - internal invariant
        raise AssertionError("wrong observation mask shape")
    finite_indexes = np.flatnonzero(finite)
    n_targets = len(group.targets)
    if finite_indexes.size == 0:
        return None

    finite_edges = [group.edges[index] for index in finite_indexes]
    incidence = _incidence(finite_edges, n_vertices)
    pinv = np.linalg.pinv(incidence)
    labels = _components(n_vertices, finite_edges)

    coefficient = np.zeros((n_targets, finite_indexes.size), dtype=np.float64)
    identifiable = np.zeros((n_targets,), dtype=bool)
    for target_index, (start, end) in enumerate(group.targets):
        if labels[start] != labels[end]:
            continue
        # c is the difference of the two unit vectors, the target is c . pinv . y.
        coefficient[target_index, :] = pinv[end, :] - pinv[start, :]
        identifiable[target_index] = True

    residual = incidence @ pinv - np.eye(finite_indexes.size)
    return PatternSolver(
        finite=finite,
        finite_indexes=finite_indexes,
        coefficient=coefficient,
        identifiable=identifiable,
        residual=residual,
    )


def solve_block(group: "ReferenceGroup", observations: np.ndarray, n_targets: int) -> np.ndarray:
    """Return target values for a block of grid points.

    ``observations`` has shape ``(n_points, n_messages)`` and NaN where a message
    is missing at that point; the returned array has shape
    ``(n_points, n_targets)`` and NaN where the target is not uniquely determined
    or where the least-squares residual of any raw observation is too large.
    """
    n_points, n_messages = observations.shape
    result = np.full((n_points, n_targets), np.nan, dtype=np.float64)
    finite = np.isfinite(observations)
    # NOTE: a grid point with exactly the same missing observations shares the
    #   same linear algebra, so patterns are solved once and reused.
    packed = np.packbits(finite, axis=1)
    unique_patterns, inverse = np.unique(packed, axis=0, return_inverse=True)
    # numpy 2.0 may return a strided inverse; flattening keeps indexing predictable.
    inverse = np.asarray(inverse).reshape(-1)
    for pattern_index in range(len(unique_patterns)):
        rows = np.flatnonzero(inverse == pattern_index)
        finite_mask = np.unpackbits(unique_patterns[pattern_index])[:n_messages].astype(bool)
        solver = _cached_solver(group, finite_mask)
        if solver is None:
            continue
        finite_observations = observations[np.ix_(rows, solver.finite_indexes)]
        residuals = finite_observations @ solver.residual.T
        # NOTE: the residual check uses the physical units of the original
        #   observations and only absorbs floating point noise, it is not a
        #   statistical error model.
        consistent = np.all(np.abs(residuals) <= RESIDUAL_TOLERANCE, axis=1)
        values = finite_observations @ solver.coefficient.T
        acceptable = consistent[:, None] & solver.identifiable[None, :]
        result[rows] = np.where(acceptable, values, np.nan)
    return result


PATTERN_CACHE_SIZE = 32
SUBSET_CACHE_SIZE = 16


def _cached_solver(group: "ReferenceGroup", finite_mask: np.ndarray) -> T.Optional[PatternSolver]:
    key = np.packbits(finite_mask).tobytes()
    cache = group.solver_cache
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    solver = build_pattern_solver(group, finite_mask)
    cache[key] = solver
    if len(cache) > PATTERN_CACHE_SIZE:
        cache.popitem(last=False)
    return solver


#
# Plan construction (headers only, no values decoded)
#


def _first_field(index: T.Any) -> T.Any:
    return index.first()


def _read_data_var_attrs(field: T.Any) -> T.Dict[str, T.Any]:
    attributes: T.Dict[str, T.Any] = {}
    for key in DATA_VAR_GRIB_KEYS:
        try:
            value = field[key]
        except Exception:
            continue
        if value is None:
            continue
        attributes["GRIB_" + key] = _norm(value)
    return attributes


def _variable_name(attributes: T.Mapping[str, T.Any]) -> str:
    for key in ("GRIB_cfVarName", "GRIB_shortName"):
        value = attributes.get(key)
        if isinstance(value, str) and value not in ("undef", "unknown"):
            return value
    param_id = attributes.get("GRIB_paramId", "unknown")
    return "paramId_%s" % param_id


def build_plan(
    path: str,
    intervals: T.Sequence[T.Tuple[int, int]],
    filter_by_keys: T.Mapping[str, T.Any],
    indexpath: str,
    errors: str,
) -> IntervalPlan:
    index = _open_index(path, filter_by_keys, indexpath, errors)
    parsed = _parse_messages(index)
    quantity, grid = _check_single_product_and_grid(parsed)
    _check_duplicates(parsed, quantity, grid)
    groups = _build_groups(parsed, intervals)

    first = _first_field(index)
    geo_dims, geo_shape, geo_coord_vars = dataset.build_geography_coordinates(first, ["geography"], errors)
    if geo_dims != ("latitude", "longitude"):
        raise IntervalTotalsError("only two-dimensional latitude/longitude grids are supported, got dims %r" % (geo_dims,))

    data_var_attrs = _read_data_var_attrs(first)
    units = data_var_attrs.get("GRIB_units", "1")
    data_var_attrs["units"] = units
    name = data_var_attrs.get("GRIB_name")
    data_var_attrs["long_name"] = name if isinstance(name, str) else "unknown"
    cf_name = data_var_attrs.get("GRIB_cfName")
    if isinstance(cf_name, str) and cf_name not in ("undef", "unknown"):
        data_var_attrs["standard_name"] = cf_name

    encoding = {
        "source": index.source(),
        "filter_by_keys": dict(filter_by_keys),
        "encode_cf": ["parameter", "time", "geography"],
    }
    dataset_attrs = dataset.build_dataset_attributes(index, dict(filter_by_keys), encoding)
    return IntervalPlan(
        path=path,
        errors=errors,
        intervals=list(intervals),
        index=index,
        messages=parsed,
        groups=groups,
        geo_dims=geo_dims,
        geo_shape=geo_shape,
        geo_coord_vars=geo_coord_vars,
        data_var_attrs=data_var_attrs,
        dataset_attrs=dataset_attrs,
        variable_name=_variable_name(data_var_attrs),
    )


#
# Lazy value access
#

#: Counters used by the tests (and by users) to prove that reading a subset of
#: reference times never decodes the field values of the other ones.
DECODE_STATISTICS = {"values_calls": 0, "values_elements": 0}


def reset_decode_statistics() -> None:
    DECODE_STATISTICS["values_calls"] = 0
    DECODE_STATISTICS["values_elements"] = 0


def _codes_id(field: T.Any) -> int:
    context = getattr(field, "context", field)
    return context.codes_id


class MessageReader:
    """Read a subset of grid points from one message without decoding the rest."""

    def __init__(self, index: T.Any, message_id: T.Any):
        self.field = index.get_field(message_id)
        self.codes_id = _codes_id(self.field)
        try:
            bitmap = eccodes.codes_get_array(self.codes_id, "bitmap")
        except Exception:
            bitmap = None
        if bitmap is None:
            self.valid_packed = None
        else:
            self.valid_packed = np.packbits(np.asarray(bitmap, dtype=np.uint8))
        try:
            self.missing_value = float(eccodes.codes_get(self.codes_id, "missingValue"))
        except Exception:
            self.missing_value = float(messages.MISSING_VAUE_INDICATOR)

    def _valid_at(self, indexes: np.ndarray) -> T.Optional[np.ndarray]:
        if self.valid_packed is None:
            return None
        byte = self.valid_packed[indexes >> 3]
        shift = (7 - (indexes & 7)).astype(np.uint8)
        return ((byte >> shift) & 1) != 0

    def values_at(self, indexes: np.ndarray) -> np.ndarray:
        indexes = np.asarray(indexes, dtype=np.int64)
        DECODE_STATISTICS["values_calls"] += 1
        DECODE_STATISTICS["values_elements"] += int(indexes.size)
        values = np.asarray(
            eccodes.codes_get_double_elements(self.codes_id, "values", indexes.tolist()),
            dtype=np.float64,
        )
        valid = self._valid_at(indexes)
        if valid is not None:
            values[~valid] = np.nan
        else:
            values[values == self.missing_value] = np.nan
            values[~np.isfinite(values)] = np.nan
        return values


def _chunk_points(number_of_points: int, number_of_messages: int) -> int:
    """Largest point chunk whose decoded ``messages x points`` buffer is one field.

    Holding every message's decoded values for the whole grid would need
    ``messages x grid`` float64 values.  Instead a chunk of at most
    ``grid // messages`` points is decoded at a time, so the transient buffer never
    exceeds a single spatial field.  The price is that each message is read in a few
    disjoint slices (``ceil(grid / chunk)`` reads per message) instead of one; the
    total number of decoded elements is unchanged.
    """
    if number_of_messages <= 0:  # pragma: no cover - groups are never empty
        return max(1, number_of_points)
    return max(1, number_of_points // number_of_messages)


class IntervalBackendArray(_BackendArrayBase):  # type: ignore[valid-type,misc]
    """Lazily computed ``(time, interval, latitude, longitude)`` array."""

    def __init__(self, plan: IntervalPlan):
        self.plan = plan
        self.n_time = len(plan.groups)
        self.n_interval = len(plan.intervals)
        self.n_lat, self.n_lon = plan.geo_shape
        self.shape = (self.n_time, self.n_interval, self.n_lat, self.n_lon)
        self.dtype = np.dtype("float64")

    def __getitem__(self, key: T.Any) -> np.ndarray:
        if xr is None:  # pragma: no cover - only reachable without xarray
            raise ImportError("xarray is required to use interval_totals")
        return xr.core.indexing.explicit_indexing_adapter(
            key, self.shape, xr.core.indexing.IndexingSupport.BASIC, self._getitem
        )

    def _solve_into(
        self,
        group: ReferenceGroup,
        flat_points: np.ndarray,
        point_positions: T.Tuple[np.ndarray, np.ndarray],
        target: np.ndarray,
    ) -> None:
        """Solve ``group`` for ``flat_points`` and store ``(interval, point)`` into ``target``.

        ``target`` is the output slice for one reference time, with shape
        ``(n_intervals, n_selected_lat, n_selected_lon)``.  Only one point chunk of
        decoded observations and one open ecCodes handle are alive at any time, so
        the transient footprint is one spatial field per chunk, never
        ``messages x grid``.
        """
        n_messages = len(group.messages)
        n_targets = len(group.targets)
        chunk = _chunk_points(flat_points.size, n_messages)
        lat_positions, lon_positions = point_positions
        for start in range(0, flat_points.size, chunk):
            stop = min(start + chunk, flat_points.size)
            points = flat_points[start:stop]
            observations = np.empty((points.size, n_messages), dtype=np.float64)
            # NOTE: messages are reopened for each chunk instead of holding every
            #   message handle at once.  A handle keeps the whole compressed message
            #   and the decoded values in native memory, so holding them all would
            #   reintroduce a ``messages x grid`` resident buffer.
            for column, message in enumerate(group.messages):
                reader = MessageReader(self.plan.index, message.message_id)
                try:
                    observations[:, column] = reader.values_at(points)
                finally:
                    del reader
            solved = solve_block(group, observations, n_targets)
            # ``target`` is laid out as (interval, lat, lon); the chunk of flat
            # points maps to (lat, lon) positions we already know.
            target[:, lat_positions[start:stop], lon_positions[start:stop]] = solved.T

    def _getitem(self, key: T.Tuple[T.Any, ...]) -> np.ndarray:
        time_sel = _index_array(key[0], self.n_time)
        interval_sel = _index_array(key[1], self.n_interval)
        lat_sel = _index_array(key[2], self.n_lat)
        lon_sel = _index_array(key[3], self.n_lon)

        out = np.full(
            (time_sel.size, interval_sel.size, lat_sel.size, lon_sel.size),
            np.nan,
            dtype=np.float64,
        )
        flat_points = (lat_sel[:, None] * self.n_lon + lon_sel[None, :]).ravel()
        point_positions = (
            np.repeat(np.arange(lat_sel.size), lon_sel.size),
            np.tile(np.arange(lon_sel.size), lat_sel.size),
        )
        for time_position, time_index in enumerate(time_sel):
            group = self.plan.groups[time_index]
            target_group = _selected_group(group, interval_sel)
            # Write directly into the output slice: it is the only full-size array
            # for this reference time, no extra temporary is allocated.
            self._solve_into(target_group, flat_points, point_positions, out[time_position])
        return _squeeze_integer_axes(out, key)


def _selected_group(group: ReferenceGroup, interval_sel: np.ndarray) -> ReferenceGroup:
    """Return a solver view of ``group`` restricted to the selected target intervals.

    The requested interval subset is often the same across repeated reads, so the
    derived group (and therefore its learned observation patterns) is cached on the
    parent group.  The number of cached subsets is bounded to keep the header-side
    memory bounded for many distinct interval selections.
    """
    if np.array_equal(interval_sel, np.arange(len(group.targets))):
        return group
    key = interval_sel.tobytes()
    cached = group.subsets.get(key)
    if cached is not None:
        return cached
    subset = ReferenceGroup(
        reference_seconds=group.reference_seconds,
        messages=group.messages,
        vertices=group.vertices,
        edges=group.edges,
        targets=[group.targets[index] for index in interval_sel],
    )
    if len(group.subsets) >= SUBSET_CACHE_SIZE:
        group.subsets.pop(next(iter(group.subsets)))
    group.subsets[key] = subset
    return subset


def _squeeze_integer_axes(out: np.ndarray, key: T.Sequence[T.Any]) -> np.ndarray:
    for axis in sorted(_integer_axes(key), reverse=True):
        out = np.take(out, 0, axis=axis)
    return out


def _index_array(indexer: T.Any, size: int) -> np.ndarray:
    base = np.arange(size, dtype=np.int64)
    if isinstance(indexer, slice):
        return base[indexer]
    if isinstance(indexer, (int, np.integer)):
        # NOTE: explicit indexing keeps the axis, so a scalar becomes a length-1
        #   selection and is squeezed by the caller afterwards.
        position = int(indexer)
        if position < 0:
            position += size
        return base[position : position + 1]
    array = np.asarray(indexer)
    if array.dtype == bool:
        return np.flatnonzero(array)
    return array.astype(np.int64)


def _integer_axes(key: T.Sequence[T.Any]) -> T.List[int]:
    return [axis for axis, indexer in enumerate(key) if isinstance(indexer, (int, np.integer))]


#
# xarray dataset construction (headers and coordinates only)
#


def _coordinate_variables(plan: IntervalPlan) -> T.Dict[str, T.Any]:
    reference_ns = np.array([group.reference_seconds * 1_000_000_000 for group in plan.groups], dtype="datetime64[ns]")
    coords: T.Dict[str, T.Any] = {
        "time": xr.Variable(
            ("time",),
            reference_ns,
            {
                "units": dataset.COORD_ATTRS["time"]["units"],
                "calendar": dataset.COORD_ATTRS["time"]["calendar"],
                "standard_name": dataset.COORD_ATTRS["time"]["standard_name"],
                "long_name": dataset.COORD_ATTRS["time"]["long_name"],
            },
        ),
        "interval": xr.Variable(
            ("interval",),
            np.arange(len(plan.intervals), dtype=np.int64),
            {
                "long_name": "target interval index",
                "units": "1",
            },
        ),
    }

    bounds = np.array([plan.intervals], dtype="timedelta64[s]").astype("timedelta64[ns]")[0]
    coords["step_bounds"] = xr.Variable(
        ("interval", "bounds"),
        bounds,
        {
            "long_name": "target accumulation interval bounds since forecast_reference_time",
            "units": "seconds",
        },
    )
    right_offsets = np.array([end for _, end in plan.intervals], dtype="timedelta64[s]")
    right_offsets = right_offsets.astype("timedelta64[ns]")
    coords["step"] = xr.Variable(
        ("interval",),
        right_offsets,
        {
            "long_name": "target accumulation interval right endpoint",
            "standard_name": "forecast_period",
            "units": "hours",
        },
    )

    valid_bounds = reference_ns[:, None, None] + bounds[None, :, :]
    coords["valid_time_bounds"] = xr.Variable(
        ("time", "interval", "bounds"),
        valid_bounds,
        {
            "long_name": "target accumulation interval bounds",
            "standard_name": "time",
        },
    )
    coords["valid_time"] = xr.Variable(
        ("time", "interval"),
        reference_ns[:, None] + right_offsets[None, :],
        {
            "long_name": "target accumulation interval right endpoint",
            "standard_name": "time",
        },
    )

    for name, variable in plan.geo_coord_vars.items():
        attributes = dict(variable.attributes)
        coords[name] = xr.Variable(variable.dimensions, np.asarray(variable.data), attributes)
    return coords


def build_dataset(plan: IntervalPlan) -> "xr.Dataset":
    if xr is None:  # pragma: no cover - xarray is required by the public API
        raise ImportError("xarray is required to use interval_totals")
    coords = _coordinate_variables(plan)
    backend_array = IntervalBackendArray(plan)
    data = xr.core.indexing.LazilyIndexedArray(backend_array)
    dimensions = ("time", "interval", plan.geo_dims[0], plan.geo_dims[1])
    data_var = xr.Variable(dimensions, data, plan.data_var_attrs)
    ds = xr.Dataset(
        {plan.variable_name: data_var},
        coords=coords,
        attrs=dict(plan.dataset_attrs),
    )
    ds[plan.variable_name].encoding.update(
        {
            "source": plan.path,
            "interval_totals": True,
            "original_shape": backend_array.shape,
            # NOTE: the writer needs the interval layout; it is mirrored into the
            #   variable attrs because ``to_grib`` only sees a data variable.
            "interval_totals_intervals": list(plan.intervals),
        }
    )
    ds.attrs["GRIB_interval_totals"] = True
    return ds


def interval_totals(
    path: T.Union[str, "os.PathLike[str]"],
    intervals: T.Any,
    *,
    backend_kwargs: T.Optional[T.Mapping[str, T.Any]] = None,
) -> "xr.Dataset":
    """Return the exact accumulated totals of ``path`` over ``intervals``.

    :param path: GRIB2 file with ``stepType=accum`` messages.
    :param intervals: non-empty sequence of ``(start, end)`` pairs, each endpoint
        a ``numpy.timedelta64`` or ``datetime.timedelta`` relative to the
        reference time of the forecasts.
    :param backend_kwargs: optional ``filter_by_keys``, ``indexpath`` and
        ``errors`` used to select the source messages.
    """
    if xr is None:  # pragma: no cover - xarray is required by the public API
        raise ImportError("xarray is required to use interval_totals")
    normalized = normalize_intervals(intervals)
    options = _validate_backend_kwargs(backend_kwargs)
    path_str = os.fspath(path)
    plan = build_plan(
        path_str,
        normalized,
        options["filter_by_keys"],
        options["indexpath"],
        options["errors"],
    )
    return build_dataset(plan)
