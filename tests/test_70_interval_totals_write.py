import numpy as np
import pytest

pytest.importorskip("xarray")

import xarray as xr

from cfgrib import xarray_to_grib
from cfgrib.intervals import interval_totals

from interval_totals_helpers import build_interval_file, read_messages

HOUR = 3600


def td(seconds: int) -> np.timedelta64:
    return np.timedelta64(seconds, "s")


def value_name(ds: xr.Dataset) -> str:
    return list(ds.data_vars)[0]


@pytest.fixture
def example_path(tmpdir):
    path = str(tmpdir.join("example.grib"))
    n = 12
    build_interval_file(
        path,
        [
            (0, 6 * HOUR, [9.0] * n),
            (3 * HOUR, 6 * HOUR, [4.0] * n),
            (3 * HOUR, 9 * HOUR, [10.0] * n),
        ],
    )
    return path


@pytest.fixture
def result(example_path):
    return interval_totals(
        example_path,
        [(td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )


def write(ds, path, **kwargs):
    with pytest.warns(FutureWarning):
        xarray_to_grib.to_grib(ds, path, **kwargs)
    return path


def test_written_headers_describe_each_target_interval(result, tmpdir):
    out_path = str(tmpdir.join("out.grib"))
    write(result, out_path)
    messages = read_messages(out_path)
    assert len(messages) == 2

    expected = [(0, 6 * HOUR), (3 * HOUR, 9 * HOUR)]
    for message, (start, end) in zip(messages, expected):
        assert message["edition"] == 2
        assert message["gridType"] == "regular_ll"
        assert message["stepType"] == "accum"
        assert message["productDefinitionTemplateNumber"] == 8
        assert message["typeOfStatisticalProcessing"] == 1
        assert message["indicatorOfUnitOfTimeRange"] == 13
        assert message["indicatorOfUnitForTimeRange"] == 13
        # ``startStep``/``endStep`` are resolved to the ``stepUnits`` concept (hours
        # here); the exact seconds are carried by the native unit-13 keys.
        assert message["startStep"] == start // HOUR
        assert message["endStep"] == end // HOUR
        assert message["forecastTime"] == start
        assert message["lengthOfTimeRange"] == end - start
        assert message["dataDate"] == 20200101
        assert message["dataTime"] == 0
        assert message["validityDate"] == 20200101
        assert message["validityTime"] == (end // 3600) * 100
        assert message["shortName"] == "tp"
        assert message["units"] == "m"

    np.testing.assert_allclose(messages[0]["values"], 9.0, atol=1e-4)
    np.testing.assert_allclose(messages[1]["values"], 10.0, atol=1e-4)


def test_round_trip_preserves_intervals_and_missing(tmpdir, example_path):
    ds = interval_totals(
        example_path,
        [(td(0), td(9 * HOUR)), (td(0), td(6 * HOUR)), (td(1 * HOUR), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    out_path = str(tmpdir.join("roundtrip.grib"))
    write(ds, out_path)

    again = interval_totals(
        out_path,
        [(td(0), td(9 * HOUR)), (td(0), td(6 * HOUR)), (td(1 * HOUR), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    np.testing.assert_allclose(
        again[value_name(again)].values,
        ds[value_name(ds)].values,
        equal_nan=True,
        atol=1e-4,
    )
    np.testing.assert_array_equal(again["step_bounds"].values, ds["step_bounds"].values)
    np.testing.assert_array_equal(again["time"].values, ds["time"].values)
    assert again[value_name(again)].attrs["units"] == ds[value_name(ds)].attrs["units"]
    assert again[value_name(again)].attrs["GRIB_shortName"] == "tp"


def test_all_missing_messages_are_written(tmpdir, example_path):
    ds = interval_totals(
        example_path,
        [(td(1 * HOUR), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert np.all(np.isnan(ds[value_name(ds)].values))
    out_path = str(tmpdir.join("allmissing.grib"))
    write(ds, out_path)
    messages = read_messages(out_path)
    assert len(messages) == 1
    message = messages[0]
    assert message["bitmapPresent"] == 1
    assert message["numberOfMissing"] == 12
    assert message["numberOfValues"] == 0
    assert message["forecastTime"] == HOUR
    assert message["lengthOfTimeRange"] == 2 * HOUR
    assert message["startStep"] == 1
    assert message["endStep"] == 3
    np.testing.assert_array_equal(message["values"], message["missingValue"])
    # the header still identifies the same interval and reference time
    assert message["dataDate"] == 20200101
    assert message["validityTime"] == 300


def test_every_reference_time_keeps_one_message(tmpdir):
    path = str(tmpdir.join("two_times.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * 12)])
    build_interval_file(path, [(0, 6 * HOUR, [3.0] * 12)], reference=(2020, 1, 1, 12, 0))
    ds = interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})
    out_path = str(tmpdir.join("two_times_out.grib"))
    write(ds, out_path)
    messages = read_messages(out_path)
    assert len(messages) == 2
    assert [message["dataTime"] for message in messages] == [0, 1200]
    np.testing.assert_allclose(messages[0]["values"], 9.0, atol=1e-4)
    np.testing.assert_allclose(messages[1]["values"], 3.0, atol=1e-4)


@pytest.fixture
def varying_result(tmpdir):
    """A result whose target fields all vary across the grid (not constant)."""
    path = str(tmpdir.join("varying.grib"))
    index = np.arange(12, dtype=float)
    c3 = index + 1.0
    v2 = index + 0.5
    v1 = c3 + v2
    v3 = index + 2.0
    build_interval_file(
        path,
        [
            (0, 6 * HOUR, list(v1)),
            (3 * HOUR, 6 * HOUR, list(v2)),
            (3 * HOUR, 9 * HOUR, list(v3)),
        ],
    )
    return interval_totals(
        path,
        [(td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )


@pytest.mark.parametrize("bits", [None, 12, 16, 24, 32])
def test_packing_tolerance_is_respected(varying_result, tmpdir, bits):
    out_path = str(tmpdir.join("packed.grib"))
    kwargs = {} if bits is None else {"grib_keys": {"bitsPerValue": bits}}
    write(varying_result, out_path, **kwargs)
    messages = read_messages(out_path)
    requested = messages[0]["bitsPerValue"]
    effective_bits = 24 if bits is None else bits
    assert requested == effective_bits
    source = varying_result[value_name(varying_result)].values[0, 0].ravel()
    # lossy packing: compare only within the tolerance implied by the bit count
    value_range = float(np.nanmax(source) - np.nanmin(source)) or 1.0
    tolerance = value_range / 2.0**effective_bits * 4 + 1e-6
    np.testing.assert_allclose(messages[0]["values"].ravel(), source, atol=tolerance)


def test_invalid_bounds_abort_before_first_message(result, tmpdir):
    out_path = str(tmpdir.join("untouched.grib"))
    with open(out_path, "wb") as file:
        file.write(b"unchanged")

    bad = result.assign_coords(
        step_bounds=(
            ("interval", "bounds"),
            np.array(
                [
                    [np.timedelta64(0, "s"), np.timedelta64(6 * HOUR, "s")],
                    [np.timedelta64(9 * HOUR, "s"), np.timedelta64(3 * HOUR, "s")],
                ],
                dtype="timedelta64[s]",
            ),
        )
    )
    with pytest.raises(ValueError):
        xarray_to_grib.to_grib(bad, out_path, no_warn=True)
    with open(out_path, "rb") as file:
        assert file.read() == b"unchanged"


def test_conflicting_interval_coordinates_abort(result, tmpdir):
    out_path = str(tmpdir.join("conflict.grib"))
    conflicting = result.assign_coords(
        step_bounds=(
            ("interval", "bounds"),
            np.array(
                [
                    [np.timedelta64(0, "s"), np.timedelta64(6 * HOUR, "s")],
                    [np.timedelta64(0, "s"), np.timedelta64(9 * HOUR, "s")],
                ],
                dtype="timedelta64[s]",
            ),
        )
    )
    with pytest.raises(ValueError):
        xarray_to_grib.to_grib(conflicting, out_path, no_warn=True)


def test_duplicate_intervals_abort(result, tmpdir):
    out_path = str(tmpdir.join("duplicates.grib"))
    duplicated = result.assign_coords(
        step_bounds=(
            ("interval", "bounds"),
            np.array(
                [[np.timedelta64(0, "s"), np.timedelta64(6 * HOUR, "s")]] * 2,
                dtype="timedelta64[s]",
            ),
        )
    )
    with pytest.raises(ValueError):
        xarray_to_grib.to_grib(duplicated, out_path, no_warn=True)


def test_legacy_writer_still_skips_all_missing(tmpdir):
    time = np.array(["2020-01-01T00:00"], dtype="datetime64[ns]")
    step = np.array([np.timedelta64(6, "h")])
    lat = np.array([10.0, 5.0, 0.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0])
    values = np.full((1, 1, 3, 4), np.nan)
    da = xr.DataArray(
        values,
        coords={"time": time, "step": step, "latitude": lat, "longitude": lon},
        dims=("time", "step", "latitude", "longitude"),
        name="tp",
        attrs={"units": "m"},
    )
    out_path = str(tmpdir.join("legacy.grib"))
    with pytest.warns(FutureWarning):
        xarray_to_grib.to_grib(da.to_dataset(name="tp"), out_path)
    assert read_messages(out_path) == []


def test_reordered_intervals_are_written_in_order(tmpdir, example_path):
    forward = interval_totals(
        example_path,
        [(td(0), td(3 * HOUR)), (td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    out_forward = str(tmpdir.join("forward.grib"))
    write(forward, out_forward)

    reordered = interval_totals(
        example_path,
        [(td(3 * HOUR), td(9 * HOUR)), (td(0), td(3 * HOUR)), (td(0), td(6 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    out_reordered = str(tmpdir.join("reordered.grib"))
    write(reordered, out_reordered)

    forward_messages = {(message["startStep"], message["endStep"]): message for message in read_messages(out_forward)}
    for message in read_messages(out_reordered):
        key = (message["startStep"], message["endStep"])
        np.testing.assert_allclose(message["values"], forward_messages[key]["values"], atol=1e-4)
    # the written order follows the target order, not a sorted one
    written = [(m["startStep"], m["endStep"]) for m in read_messages(out_reordered)]
    assert written == [(3, 9), (0, 3), (0, 6)]


def test_missing_values_round_trip_per_point(tmpdir):
    path = str(tmpdir.join("partial.grib"))
    first = np.full(12, 9.0)
    second = np.full(12, 4.0)
    third = np.full(12, 10.0)
    first[5] = np.nan
    build_interval_file(
        path,
        [(0, 6 * HOUR, first), (3 * HOUR, 6 * HOUR, second), (3 * HOUR, 9 * HOUR, third)],
    )
    ds = interval_totals(
        path,
        [(td(0), td(9 * HOUR)), (td(0), td(6 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    out_path = str(tmpdir.join("partial_out.grib"))
    write(ds, out_path)

    messages = read_messages(out_path)
    # (0,9] at point 5 must be missing, (0,6] too; every other point is finite
    for message, start, end in zip(messages, [0, 0], [9 * HOUR, 6 * HOUR]):
        values = message["values"]
        missing = values == message["missingValue"]
        assert missing[5]
        assert not missing[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]].any()

    again = interval_totals(
        out_path,
        [(td(0), td(9 * HOUR)), (td(0), td(6 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    np.testing.assert_allclose(again[value_name(again)].values, ds[value_name(ds)].values, equal_nan=True, atol=1e-4)


def test_conflicting_units_are_rejected_before_writing(result, tmpdir):
    out_path = str(tmpdir.join("units.grib"))
    with open(out_path, "wb") as file:
        file.write(b"unchanged")
    with pytest.raises(ValueError):
        xarray_to_grib.to_grib(result, out_path, no_warn=True, grib_keys={"units": "mm"})
    with open(out_path, "rb") as file:
        assert file.read() == b"unchanged"
