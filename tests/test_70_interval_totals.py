import datetime
import importlib

import numpy as np
import pytest

pytest.importorskip("xarray")

import xarray as xr

from cfgrib.intervals import IntervalTotalsError, interval_totals

from interval_totals_helpers import build_interval_file, make_accumulation_message, read_messages

HOUR = 3600
DAY = 86400


def dt(hours: float) -> datetime.timedelta:
    return datetime.timedelta(hours=hours)


def td(seconds: int) -> np.timedelta64:
    return np.timedelta64(seconds, "s")


def values_of(ds: xr.Dataset, time_index: int = 0) -> np.ndarray:
    name = list(ds.data_vars)[0]
    return ds[name].values[time_index]


def by_reference(messages):
    return sorted({message["dataDate"] * 10000 + message["dataTime"] for message in messages})


@pytest.fixture
def example_path(tmpdir):
    """The three overlapping accumulations from the specification."""
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


def test_overlapping_intervals_solve_but_fine_split_is_ambiguous(example_path):
    ds = interval_totals(
        example_path,
        [(td(0), td(9 * HOUR)), (td(1 * HOUR), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    result = values_of(ds)
    assert np.allclose(result[0], 15.0, atol=1e-8)
    assert np.all(np.isnan(result[1]))
    # composition: the split (0,3] + (3,9] must reproduce the total
    finer = interval_totals(
        example_path,
        [(td(0), td(3 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    split = values_of(finer)
    assert np.allclose(split[0] + split[1], 15.0, atol=1e-8)


def test_gap_in_intervals_is_unidentifiable(tmpdir):
    path = str(tmpdir.join("gap.grib"))
    n = 12
    build_interval_file(
        path,
        [
            (0, 6 * HOUR, [9.0] * n),
            (3 * HOUR, 9 * HOUR, [10.0] * n),
        ],
    )
    ds = interval_totals(
        path,
        [(td(0), td(9 * HOUR)), (td(0), td(3 * HOUR)), (td(6 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    result = values_of(ds)
    assert np.all(np.isnan(result[0]))
    assert np.all(np.isnan(result[1]))
    assert np.all(np.isnan(result[2]))


def test_missing_points_change_identifiability(tmpdir):
    path = str(tmpdir.join("missing.grib"))
    n = 12
    first = np.full(n, 9.0)
    second = np.full(n, 4.0)
    third = np.full(n, 10.0)
    # Flat index 5 (latitude=10, longitude=10) loses the (0,6] observation.
    first[5] = np.nan
    second[5] = np.nan
    build_interval_file(
        path,
        [
            (0, 6 * HOUR, first),
            (3 * HOUR, 6 * HOUR, second),
            (3 * HOUR, 9 * HOUR, third),
        ],
    )
    ds = interval_totals(
        path,
        [(td(0), td(9 * HOUR)), (td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    result = values_of(ds).reshape(3, 12)
    # every finite point still resolves (0,9] and (0,6]; only (3,9] was directly given
    assert np.allclose(result[0, :5], 15.0, atol=1e-8)
    assert np.allclose(result[1, :5], 9.0, atol=1e-8)
    assert np.allclose(result[2, :5], 10.0, atol=1e-8)
    # the point missing (0,6] cannot resolve (0,9] or (0,6] but still has (3,9]
    assert np.isnan(result[0, 5])
    assert np.isnan(result[1, 5])
    assert np.isclose(result[2, 5], 10.0, atol=1e-8)


def test_same_valid_time_from_different_references_does_not_mix(tmpdir):
    path = str(tmpdir.join("references.grib"))
    n = 12
    # Reference 00 UTC: (0,6] = 9. Reference 03 UTC: (0,6] = 4, i.e. valid 03-09.
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * n)])
    build_interval_file(path, [(0, 6 * HOUR, [4.0] * n)], reference=(2020, 1, 1, 3, 0))
    ds = interval_totals(
        path,
        [(td(0), td(6 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    result = values_of(ds) if ds.sizes["time"] == 1 else None
    assert ds.sizes["time"] == 2
    assert list(ds["time"].values) == [
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-01T03:00:00"),
    ]
    assert np.allclose(ds[list(ds.data_vars)[0]].values[:, 0, 0, 0], [9.0, 4.0], atol=1e-8)


def test_redundant_consistent_or_contradictory(tmpdir):
    n = 12
    consistent_path = str(tmpdir.join("consistent.grib"))
    build_interval_file(
        consistent_path,
        [
            (0, 6 * HOUR, [9.0] * n),
            (3 * HOUR, 6 * HOUR, [4.0] * n),
            (0, 3 * HOUR, [5.0] * n),
            (3 * HOUR, 9 * HOUR, [10.0] * n),
            (0, 9 * HOUR, [15.0] * n),
        ],
    )
    ds = interval_totals(
        consistent_path,
        [(td(0), td(9 * HOUR)), (td(0), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert np.allclose(values_of(ds)[0], 15.0, atol=1e-8)
    assert np.allclose(values_of(ds)[1], 5.0, atol=1e-8)

    contradictory_path = str(tmpdir.join("contradictory.grib"))
    build_interval_file(
        contradictory_path,
        [
            (0, 6 * HOUR, [9.0] * n),
            (3 * HOUR, 6 * HOUR, [4.0] * n),
            (0, 3 * HOUR, [6.0] * n),
        ],
    )
    ds = interval_totals(
        contradictory_path,
        [(td(0), td(9 * HOUR)), (td(0), td(6 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert np.all(np.isnan(values_of(ds)))


def test_negative_accumulations_are_kept(tmpdir):
    path = str(tmpdir.join("negative.grib"))
    n = 12
    build_interval_file(
        path,
        [
            (0, 6 * HOUR, [-9.0] * n),
            (3 * HOUR, 6 * HOUR, [-4.0] * n),
            (3 * HOUR, 9 * HOUR, [-10.0] * n),
        ],
    )
    ds = interval_totals(
        path,
        [(td(0), td(9 * HOUR)), (td(0), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert np.allclose(values_of(ds)[0], -15.0, atol=1e-8)
    assert np.allclose(values_of(ds)[1], -5.0, atol=1e-8)


def test_seconds_and_hours_mix(tmpdir):
    path = str(tmpdir.join("units.grib"))
    n = 12
    # same intervals encoded once in seconds and once in hours
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * n)], unit=13)
    build_interval_file(path, [(3 * HOUR, 6 * HOUR, [4.0] * n)], unit=1)
    build_interval_file(path, [(3 * HOUR, 9 * HOUR, [10.0] * n)], unit=1)
    ds = interval_totals(
        path,
        [(td(0), td(9 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert np.allclose(values_of(ds)[0], 15.0, atol=1e-8)
    assert np.allclose(values_of(ds)[1], 10.0, atol=1e-8)


def test_minutes_and_days_units(tmpdir):
    path = str(tmpdir.join("minutes.grib"))
    n = 12
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * n)], unit=1)
    # add a 6-hour accumulation expressed in minutes: 360 minutes
    build_interval_file(path, [(6 * HOUR, 12 * HOUR, [12.0] * n)], unit=1)
    ds = interval_totals(
        path,
        [(td(0), td(6 * HOUR)), (td(6 * HOUR), td(12 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert np.allclose(values_of(ds)[0], 9.0, atol=1e-8)
    assert np.allclose(values_of(ds)[1], 12.0, atol=1e-8)


def test_relative_grouping_is_independent_per_point(tmpdir):
    path = str(tmpdir.join("independent.grib"))
    n = 12
    first = np.full(n, 9.0)
    second = np.full(n, 4.0)
    first[0] = np.nan
    second[0] = np.nan
    build_interval_file(
        path,
        [(0, 6 * HOUR, first), (3 * HOUR, 6 * HOUR, second), (3 * HOUR, 9 * HOUR, [10.0] * n)],
    )
    ds = interval_totals(
        path,
        [(td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    result = values_of(ds).reshape(12)
    assert np.allclose(result, 10.0, atol=1e-8)


#
# Argument validation
#


def test_interval_arguments_accept_seconds_and_hours(example_path):
    seconds = interval_totals(example_path, [(td(0), td(9 * HOUR))], backend_kwargs={"indexpath": ""})
    hours_delta = interval_totals(example_path, [(dt(0), dt(9))], backend_kwargs={"indexpath": ""})
    np.testing.assert_allclose(values_of(seconds), values_of(hours_delta), equal_nan=True, atol=1e-8)


@pytest.mark.parametrize(
    "intervals",
    [
        [],
        None,
        [np.timedelta64(0, "s")],
        [(td(0), td(0))],
        [(td(9 * HOUR), td(3 * HOUR))],
        [(td(-1), td(3 * HOUR))],
        [(np.timedelta64(0, "s"), np.timedelta64(1, "ms"))],
        [(np.timedelta64(0, "s"), np.timedelta64(1500, "ms"))],
        [(np.timedelta64("NaT", "s"), np.timedelta64(3600, "s"))],
        [(td(0), td(HOUR)), (td(0), td(HOUR))],
        [(True, td(HOUR))],
        [(1, td(HOUR))],
        [(td(0), td(HOUR), td(2 * HOUR))],
    ],
)
def test_invalid_intervals_raise(example_path, intervals):
    with pytest.raises(ValueError):
        interval_totals(example_path, intervals, backend_kwargs={"indexpath": ""})


def test_subsecond_values_are_rejected_as_endpoints(example_path):
    with pytest.raises(ValueError):
        interval_totals(
            example_path,
            [(np.timedelta64(0, "ns"), np.timedelta64(1, "ns"))],
            backend_kwargs={"indexpath": ""},
        )


def test_target_order_and_overlaps_are_preserved(example_path):
    forward = interval_totals(
        example_path,
        [(td(0), td(3 * HOUR)), (td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    assert list(forward["interval"].values) == [0, 1, 2]
    assert np.allclose(values_of(forward)[1], 9.0, atol=1e-8)
    assert np.allclose(values_of(forward)[2], 10.0, atol=1e-8)
    reordered = interval_totals(
        example_path,
        [(td(3 * HOUR), td(9 * HOUR)), (td(0), td(6 * HOUR)), (td(0), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    np.testing.assert_allclose(values_of(reordered), values_of(forward)[::-1], equal_nan=True, atol=1e-8)


def test_step_and_valid_time_coordinates(example_path):
    intervals = [(td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))]
    ds = interval_totals(example_path, intervals, backend_kwargs={"indexpath": ""})
    assert ds["interval"].dims == ("interval",)
    assert ds["step_bounds"].dims == ("interval", "bounds")
    assert ds["step"].dims == ("interval",)
    assert ds["valid_time_bounds"].dims == ("time", "interval", "bounds")
    assert ds["valid_time"].dims == ("time", "interval")
    np.testing.assert_array_equal(
        ds["step_bounds"].values, np.array([[0, 6 * HOUR], [3 * HOUR, 9 * HOUR]], dtype="timedelta64[s]")
    )
    np.testing.assert_array_equal(
        ds["step"].values, np.array([6 * HOUR, 9 * HOUR], dtype="timedelta64[s]").astype("timedelta64[ns]")
    )
    reference = np.datetime64("2020-01-01T00:00:00")
    np.testing.assert_array_equal(ds["valid_time"].values[0], reference + ds["step"].values)
    np.testing.assert_array_equal(
        ds["valid_time_bounds"].values[0, 0],
        reference + np.array([0, 6 * HOUR], dtype="timedelta64[s]"),
    )
    # the data variable keeps the physical name, unit and coordinates
    name = list(ds.data_vars)[0]
    assert name == "tp"
    assert ds[name].attrs["units"] == "m"
    assert ds[name].dims == ("time", "interval", "latitude", "longitude")
    np.testing.assert_array_equal(ds["latitude"].values, [10.0, 5.0, 0.0])
    np.testing.assert_array_equal(ds["longitude"].values, [0.0, 10.0, 20.0, 30.0])


def test_single_reference_time_keeps_time_axis(example_path):
    ds = interval_totals(example_path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})
    assert ds.sizes["time"] == 1
    assert ds[list(ds.data_vars)[0]].dims[0] == "time"


def test_empty_observation_decodes_to_all_missing(tmpdir):
    path = str(tmpdir.join("allmissing.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [np.nan] * 12)])
    ds = interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})
    assert np.all(np.isnan(values_of(ds)))


#
# Header rejections (must happen before any value is decoded)
#


def test_rejects_non_accumulation(tmpdir):
    path = str(tmpdir.join("instant.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)], product_definition_template=0, statistical_processing=1)
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_no_time_range(tmpdir):
    path = str(tmpdir.join("norange.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)], product_definition_template=8, extra_keys={"numberOfTimeRange": 2})
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_ensemble_product(tmpdir):
    path = str(tmpdir.join("ensemble.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)], type_of_processed_data=4)
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_unsupported_grid(tmpdir):
    path = str(tmpdir.join("lambert.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)], grid_type="lambert")
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_unsupported_time_unit(tmpdir):
    path = str(tmpdir.join("unit3.grib"))
    # a syntactically valid message whose native unit indicator is "month" (3),
    # which this entry point deliberately refuses instead of guessing units
    build_interval_file(
        path,
        [(0, 6 * HOUR, [1.0] * 12)],
        extra_keys={
            "indicatorOfUnitOfTimeRange": 3,
            "indicatorOfUnitForTimeRange": 3,
            "forecastTime": 0,
            "lengthOfTimeRange": 6,
        },
    )
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_multiple_quantities(tmpdir):
    path = str(tmpdir.join("two_params.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)])
    build_interval_file(path, [(3 * HOUR, 6 * HOUR, [1.0] * 12)], short_name="acpcp")
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_duplicate_messages(tmpdir):
    path = str(tmpdir.join("duplicate.grib"))
    message = make_accumulation_message(0, 6 * HOUR, [9.0] * 12)
    with open(path, "wb") as file:
        file.write(message)
        file.write(message)
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_bad_backend_kwargs(example_path):
    with pytest.raises(IntervalTotalsError):
        interval_totals(example_path, [(td(0), td(HOUR))], backend_kwargs={"squeeze": False})
    with pytest.raises(IntervalTotalsError):
        interval_totals(example_path, [(td(0), td(HOUR))], backend_kwargs={"filter_by_keys": 3})


def test_header_errors_precede_value_decoding(tmpdir, monkeypatch):
    module = importlib.import_module("cfgrib.intervals")

    path = str(tmpdir.join("bad_header.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)], product_definition_template=0, statistical_processing=1)

    def fail(*args, **kwargs):
        raise AssertionError("values must not be decoded when the header is rejected")

    monkeypatch.setattr(module, "MessageReader", fail)
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


#
# filter_by_keys, cold/warm index and lazy decoding
#


def test_filter_by_keys_selects_one_quantity(tmpdir):
    path = str(tmpdir.join("filtered.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * 12)])
    build_interval_file(path, [(0, 6 * HOUR, [2.0] * 12)], short_name="acpcp")
    ds = interval_totals(
        path,
        [(td(0), td(6 * HOUR))],
        backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": "acpcp"}},
    )
    assert np.allclose(values_of(ds), 2.0, atol=1e-8)


def test_warm_and_cold_index_agree(example_path, tmpdir):
    import shutil

    copy_path = str(tmpdir.join("copy.grib"))
    shutil.copy(example_path, copy_path)
    intervals = [(td(0), td(9 * HOUR))]
    cold = interval_totals(copy_path, intervals)
    warm = interval_totals(copy_path, intervals)
    np.testing.assert_allclose(values_of(cold), values_of(warm), equal_nan=True, atol=1e-8)


def test_construction_does_not_decode_values(example_path):
    module = importlib.import_module("cfgrib.intervals")

    module.reset_decode_statistics()
    ds = interval_totals(example_path, [(td(0), td(9 * HOUR))], backend_kwargs={"indexpath": ""})
    assert module.DECODE_STATISTICS["values_calls"] == 0

    module.reset_decode_statistics()
    _ = values_of(ds)
    assert module.DECODE_STATISTICS["values_calls"] > 0


def test_reading_one_reference_time_skips_the_other(tmpdir):
    module = importlib.import_module("cfgrib.intervals")

    path = str(tmpdir.join("two_times.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * 12)])
    build_interval_file(path, [(0, 6 * HOUR, [3.0] * 12)], reference=(2020, 1, 1, 12, 0))
    ds = interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})
    assert ds.sizes["time"] == 2

    module.reset_decode_statistics()
    first = ds[list(ds.data_vars)[0]].isel(time=[0]).values
    first_calls = module.DECODE_STATISTICS["values_calls"]
    assert first_calls == 1
    assert module.DECODE_STATISTICS["values_elements"] == 12
    assert np.allclose(first, 9.0, atol=1e-8)

    module.reset_decode_statistics()
    second = ds[list(ds.data_vars)[0]].isel(time=[1]).values
    assert module.DECODE_STATISTICS["values_calls"] == 1
    assert np.allclose(second, 3.0, atol=1e-8)


def test_single_point_read_decodes_one_value(example_path):
    module = importlib.import_module("cfgrib.intervals")

    ds = interval_totals(example_path, [(td(0), td(9 * HOUR))], backend_kwargs={"indexpath": ""})
    module.reset_decode_statistics()
    value = ds[list(ds.data_vars)[0]].isel(latitude=[0], longitude=[2]).values
    assert np.allclose(value, 15.0, atol=1e-8)
    assert module.DECODE_STATISTICS["values_elements"] == 3


def test_rejects_multiple_time_ranges(tmpdir):
    path = str(tmpdir.join("tworanges.grib"))
    build_interval_file(
        path,
        [(0, 6 * HOUR, [1.0] * 12)],
        extra_keys={"numberOfTimeRange": 2, "startStep": 0, "endStep": 6},
    )
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_unsupported_step_type(tmpdir):
    path = str(tmpdir.join("max.grib"))
    # typeOfStatisticalProcessing=2 is "maximum", not an accumulation
    build_interval_file(
        path,
        [(0, 6 * HOUR, [1.0] * 12)],
        statistical_processing=2,
    )
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_non_forecast_data_type(tmpdir):
    path = str(tmpdir.join("an.grib"))
    build_interval_file(
        path,
        [(0, 6 * HOUR, [1.0] * 12)],
        type_of_processed_data=0,  # analysis
    )
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_rejects_multiple_grids(tmpdir):
    path = str(tmpdir.join("two_grids.grib"))
    build_interval_file(path, [(0, 6 * HOUR, [1.0] * 12)])
    build_interval_file(path, [(3 * HOUR, 6 * HOUR, [1.0] * 6)], ni=3, nj=2)
    with pytest.raises(IntervalTotalsError):
        interval_totals(path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})


def test_subset_and_split_requests_agree(example_path):
    full = interval_totals(
        example_path,
        [(td(0), td(3 * HOUR)), (td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    # asking for a single interval reproduces the matching column
    for column in range(3):
        single = interval_totals(
            example_path,
            [[(td(0), td(3 * HOUR)), (td(0), td(6 * HOUR)), (td(3 * HOUR), td(9 * HOUR))][column]],
            backend_kwargs={"indexpath": ""},
        )
        np.testing.assert_allclose(
            values_of(single).reshape(-1),
            values_of(full)[column].reshape(-1),
            equal_nan=True,
            atol=1e-8,
        )
    # and the whole-grid read equals a per-point read
    name = list(full.data_vars)[0]
    per_point = np.array(
        [
            [full[name].isel(interval=[j], latitude=[i // 4], longitude=[i % 4]).values[0, 0, 0, 0] for j in range(3)]
            for i in range(12)
        ]
    )
    np.testing.assert_allclose(per_point, values_of(full).reshape(3, 12).T, equal_nan=True, atol=1e-8)


def test_python_timedelta_and_numpy_timedelta_are_equivalent(example_path):
    numpy_result = interval_totals(example_path, [(td(0), td(6 * HOUR))], backend_kwargs={"indexpath": ""})
    python_result = interval_totals(
        example_path,
        [(datetime.timedelta(0), datetime.timedelta(hours=6))],
        backend_kwargs={"indexpath": ""},
    )
    np.testing.assert_allclose(values_of(numpy_result), values_of(python_result), equal_nan=True, atol=1e-8)


def test_same_valid_time_needs_matching_reference_edges(tmpdir):
    """Two references whose valid times overlap must not share constraints."""
    path = str(tmpdir.join("overlapping_valid.grib"))
    n = 12
    # Reference 00: (0,6] covers valid 00-06. Reference 03: (0,3] covers valid 03-06.
    build_interval_file(path, [(0, 6 * HOUR, [9.0] * n)])
    build_interval_file(path, [(0, 3 * HOUR, [4.0] * n)], reference=(2020, 1, 1, 3, 0))
    ds = interval_totals(
        path,
        [(td(0), td(6 * HOUR)), (td(3 * HOUR), td(6 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    name = list(ds.data_vars)[0]
    values = ds[name].values
    # reference 00: (0,6] given, (3,6] unknown
    assert np.allclose(values[0, 0], 9.0, atol=1e-8)
    assert np.all(np.isnan(values[0, 1]))
    # reference 03: (3,6] is valid 06-09, not covered; (0,6] would be valid 03-09
    assert np.all(np.isnan(values[1, 0]))
    assert np.all(np.isnan(values[1, 1]))


def test_module_imports_without_xarray():
    """The package must stay importable when the optional xarray is missing."""
    import subprocess
    import sys

    code = """
import builtins
real_import = builtins.__import__
def fake(name, *args, **kwargs):
    if name == 'xarray' or name.startswith('xarray.'):
        raise ImportError('no xarray')
    return real_import(name, *args, **kwargs)
builtins.__import__ = fake
import importlib
import cfgrib
module = importlib.import_module('cfgrib.intervals')
assert module.xr is None
try:
    module.interval_totals('unused.grib', [])
except ImportError:
    pass
else:
    raise SystemExit('interval_totals must require xarray')
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_missing_redundant_observation_keeps_other_targets(tmpdir):
    """A missing (0,3] observation does not hide targets reachable through other edges."""
    path = str(tmpdir.join("mixed.grib"))
    n = 12
    v06 = np.full(n, 9.0)
    v36 = np.full(n, 4.0)
    v03 = np.full(n, 5.0)
    v39 = np.full(n, 10.0)
    v03[4:8] = 6.0  # contradictory points
    v03[8:12] = np.nan  # redundant observation missing
    build_interval_file(
        path,
        [
            (0, 6 * HOUR, v06),
            (3 * HOUR, 6 * HOUR, v36),
            (0, 3 * HOUR, v03),
            (3 * HOUR, 9 * HOUR, v39),
        ],
    )
    ds = interval_totals(
        path,
        [(td(0), td(9 * HOUR)), (td(0), td(3 * HOUR))],
        backend_kwargs={"indexpath": ""},
    )
    result = values_of(ds).reshape(2, 12)
    # consistent points
    np.testing.assert_allclose(result[:, :4], [[15.0] * 4, [5.0] * 4], atol=1e-8)
    # contradictory points lose every target at that point
    assert np.all(np.isnan(result[:, 4:8]))
    # points whose (0,3] message is missing still resolve both targets through
    # the other three edges: 9 + 10 - 4 = 15 and 9 - 4 = 5.
    np.testing.assert_allclose(result[:, 8:], [[15.0] * 4, [5.0] * 4], atol=1e-8)


def test_random_missing_matches_brute_force(tmpdir):
    """Random per-point missing patterns agree with a direct sum of increments."""
    import cfgrib.intervals as module

    rng = np.random.default_rng(0)
    ni, nj = 4, 3
    n = ni * nj
    increments = rng.normal(size=(12, n))
    masked = increments.copy()
    for i in range(12):
        masked[i, rng.random(n) < 0.2] = np.nan
    messages = [make_accumulation_message(i * HOUR, (i + 1) * HOUR, list(masked[i]), ni=ni, nj=nj) for i in range(12)]
    path = str(tmpdir.join("random.grib"))
    with open(path, "wb") as file:
        for message in messages:
            file.write(message)

    ranges = [(0, 3), (0, 12), (3, 6)]
    ds = interval_totals(
        path,
        [(td(a * HOUR), td(b * HOUR)) for a, b in ranges],
        backend_kwargs={"indexpath": ""},
    )
    got = values_of(ds)
    for point in range(n):
        for target_index, (a, b) in enumerate(ranges):
            window = masked[a:b, point]
            expected = np.nan if np.any(~np.isfinite(window)) else window.sum()
            value = got[target_index].ravel()[point]
            assert np.isnan(expected) == np.isnan(value)
            if np.isfinite(expected):
                assert abs(value - expected) < 1e-6
