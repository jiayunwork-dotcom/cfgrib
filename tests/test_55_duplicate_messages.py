"""Regression tests for ``duplicate_messages="stack"`` and the ``grib_occurrence`` key.

All the GRIB files used here are generated locally with ecCodes, so the tests do not depend on
any remote meteorological service. Run them with::

    pytest tests/test_55_duplicate_messages.py -v

The tests that check lazy decoding count the calls to ``cfgrib.dataset.get_values_in_order``
and assert that only the selected messages have their ``values`` decoded.
"""

import typing as T

import eccodes  # type: ignore
import numpy as np
import pytest

from cfgrib import dataset, messages

xr = pytest.importorskip("xarray")


def make_message(
    shortName: str = "t",
    paramId: T.Optional[int] = None,
    dataDate: int = 20240101,
    dataTime: int = 0,
    step: int = 0,
    level: float = 0.0,
    typeOfLevel: str = "surface",
    number: T.Optional[int] = None,
    totalNumber: int = 2,
    value: float = 0.0,
    Ni: int = 4,
    Nj: int = 3,
) -> int:
    """Create a single-field ``regular_ll`` GRIB2 message with constant values."""
    gid = eccodes.codes_new_from_samples("regular_ll_sfc_grib2", eccodes.CODES_PRODUCT_GRIB)
    eccodes.codes_set(gid, "shortName", shortName)
    if paramId is not None:
        eccodes.codes_set(gid, "paramId", paramId)
    eccodes.codes_set(gid, "dataDate", dataDate)
    eccodes.codes_set(gid, "dataTime", dataTime)
    eccodes.codes_set(gid, "typeOfLevel", typeOfLevel)
    eccodes.codes_set(gid, "level", level)
    if number is not None:
        eccodes.codes_set(gid, "typeOfEnsembleForecast", 3)
        eccodes.codes_set(gid, "number", number)
        eccodes.codes_set(gid, "totalNumber", totalNumber)
    eccodes.codes_set(gid, "Ni", Ni)
    eccodes.codes_set(gid, "Nj", Nj)
    eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", 10.0)
    eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", 0.0)
    eccodes.codes_set(gid, "longitudeOfFirstGridPointInDegrees", 0.0)
    eccodes.codes_set(gid, "longitudeOfLastGridPointInDegrees", 30.0)
    eccodes.codes_set(gid, "iDirectionIncrementInDegrees", 10.0)
    eccodes.codes_set(gid, "jDirectionIncrementInDegrees", 5.0)
    eccodes.codes_set(gid, "step", step)
    eccodes.codes_set_values(gid, np.full(Ni * Nj, value, dtype="float64"))
    # NOTE: setting the number of points after the values resizes the data section only.
    eccodes.codes_set(gid, "numberOfDataPoints", Ni * Nj)
    return gid


def write_grib(path: str, specs: T.Sequence[T.Dict[str, T.Any]]) -> str:
    """Write one single-field GRIB message per spec, in order."""
    with open(path, "wb") as file:
        for spec in specs:
            gid = make_message(**spec)
            eccodes.codes_write(gid, file)
            eccodes.codes_release(gid)
    return path


def write_multi_field_grib(path: str, specs: T.Sequence[T.Dict[str, T.Any]]) -> str:
    """Write all the messages as the fields of one physical GRIB2 message."""
    gids = [make_message(**spec) for spec in specs]
    multi = eccodes.codes_grib_multi_new()
    eccodes.codes_grib_multi_append(gids[0], 1, multi)
    for gid in gids[1:]:
        eccodes.codes_grib_multi_append(gid, 4, multi)
    with open(path, "wb") as file:
        eccodes.codes_grib_multi_write(multi, file)
    eccodes.codes_grib_multi_release(multi)
    for gid in gids:
        eccodes.codes_release(gid)
    return path


def assert_values(actual: T.Any, expected: T.Any) -> None:
    """Compare nested lists of floats, treating NaN as equal to NaN."""
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), equal_nan=True)


def count_values_decoding(monkeypatch: pytest.MonkeyPatch) -> T.List[int]:
    """Patch ``get_values_in_order`` to record every message whose values are decoded."""
    calls: T.List[int] = []
    original = dataset.get_values_in_order

    def counting(message: T.Any, shape: T.Tuple[int, ...]) -> np.ndarray:
        calls.append(1)
        return original(message, shape)

    monkeypatch.setattr(dataset, "get_values_in_order", counting)
    return calls


@pytest.fixture
def duplicate_file(tmpdir: str) -> str:
    """Two parameters, several cells and unequal numbers of duplicates.

    ``t`` has 3 occurrences at ``step=0`` (the 2nd and 3rd are identical) and 2 at
    ``step=6``; ``z`` has 1 occurrence at each step. So the longest run is 3 and some
    ``(step, grib_occurrence)`` cells are empty.
    """
    path = str(tmpdir.join("duplicates.grib2"))
    return write_grib(
        path,
        [
            dict(shortName="t", step=0, value=1.0),
            dict(shortName="t", step=0, value=2.0),
            dict(shortName="t", step=0, value=2.0),
            dict(shortName="t", step=6, value=3.0),
            dict(shortName="t", step=6, value=4.0),
            dict(shortName="z", step=0, value=9.0),
            dict(shortName="z", step=6, value=10.0),
        ],
    )


def test_stack_keeps_every_message(duplicate_file: str) -> None:
    ds = xr.open_dataset(
        duplicate_file, engine="cfgrib", backend_kwargs=dict(duplicate_messages="stack")
    )
    assert ds["t"].dims == ("step", "grib_occurrence", "latitude", "longitude")
    assert ds["grib_occurrence"].values.tolist() == [0, 1, 2]
    values = ds["t"].values[..., 0, 0]
    # step 0: three occurrences, step 6: only two, not renumbered and no row lost
    assert_values(values, [[1.0, 2.0, 2.0], [3.0, 4.0, np.nan]])
    # an incomplete cell of another variable is still NaN, never filled from a neighbour
    z_values = ds["z"].values[..., 0, 0]
    assert_values(z_values, [[9.0, np.nan, np.nan], [10.0, np.nan, np.nan]])
    assert ds["z"].dims == ("step", "grib_occurrence", "latitude", "longitude")
    assert np.isnan(ds["z"].values).sum() == 2 * 2 * 4 * 3


def test_default_and_first_are_unchanged(duplicate_file: str) -> None:
    implicit = xr.open_dataset(duplicate_file, engine="cfgrib")
    explicit = xr.open_dataset(
        duplicate_file, engine="cfgrib", backend_kwargs=dict(duplicate_messages="first")
    )
    for ds in (implicit, explicit):
        assert "grib_occurrence" not in ds.dims
        assert ds["t"].dims == ("step", "latitude", "longitude")
        assert_values(ds["t"].values[:, 0, 0], [1.0, 3.0])
    assert implicit["t"].identical(explicit["t"])


def test_duplicate_messages_invalid_value(duplicate_file: str) -> None:
    with pytest.raises(ValueError):
        xr.open_dataset(
            duplicate_file, engine="cfgrib", backend_kwargs=dict(duplicate_messages="all")
        )
    with pytest.raises(ValueError):
        xr.open_dataset(duplicate_file, engine="cfgrib", backend_kwargs=dict(duplicate_messages=1))


def test_open_file_and_open_fieldset_accept_the_option(tmpdir: str) -> None:
    path = str(tmpdir.join("two.grib2"))
    write_grib(path, [dict(shortName="t", step=0, value=1.0), dict(shortName="t", value=2.0)])
    ds = dataset.open_file(path, duplicate_messages="stack")
    assert ds.dimensions["grib_occurrence"] == 2
    assert_values(np.asarray(ds.variables["t"].data[:, 0, 0]), [1.0, 2.0])

    res = dataset.open_file(path)
    assert "grib_occurrence" not in res.dimensions

    fields = [
        dict(
            gridType="regular_ll",
            Nx=2,
            Ny=3,
            distinctLatitudes=[10.0, 5.0, 0.0],
            distinctLongitudes=[0.0, 10.0],
            paramId=130,
            shortName="t",
            units="K",
            name="Temperature",
            cfName="air_temperature",
            cfVarName="t",
            typeOfLevel="surface",
            level=0,
            stepType="instant",
            stepUnits=1,
            dataDate=20240101,
            dataTime=0,
            step=0,
            gridDefinitionDescription="",
            values=[[value] * 2] * 3,
        )
        for value in (1.0, 2.0)
    ]
    ds = dataset.open_fieldset(fields, duplicate_messages="stack")
    assert ds.dimensions["grib_occurrence"] == 2
    assert ds.variables["t"].data[0, 0, 0] == 1.0
    assert ds.variables["t"].data[1, 0, 0] == 2.0
    with pytest.raises(ValueError):
        dataset.open_fieldset(fields, duplicate_messages="not-a-mode")


def test_single_occurrence_is_not_squeezed(tmpdir: str) -> None:
    path = str(tmpdir.join("single.grib2"))
    write_grib(path, [dict(shortName="t", value=1.0)])
    ds = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs=dict(duplicate_messages="stack", squeeze=True)
    )
    assert "grib_occurrence" in ds.dims
    assert ds["grib_occurrence"].values.tolist() == [0]
    assert ds["t"].dims == ("grib_occurrence", "latitude", "longitude")


@pytest.mark.parametrize("filter_value", [2, [2], [0, 2], (0, 2), np.array([1])])
def test_grib_occurrence_filter(filter_value: T.Any, duplicate_file: str) -> None:
    ds = xr.open_dataset(
        duplicate_file,
        engine="cfgrib",
        backend_kwargs=dict(
            duplicate_messages="stack", filter_by_keys=dict(grib_occurrence=filter_value)
        ),
    )
    expected = [filter_value] if np.isscalar(filter_value) else list(filter_value)
    assert ds["grib_occurrence"].values.tolist() == sorted(expected)
    # selected labels keep their original ordinal: occurrence 2 stays labelled 2
    for label in ds["grib_occurrence"].values.tolist():
        assert label in expected
    values = ds["t"].values[..., 0, 0]
    if expected == [2]:
        assert_values(values, [[2.0], [np.nan]])
    elif expected == [0, 2]:
        assert_values(values, [[1.0, 2.0], [3.0, np.nan]])


@pytest.mark.parametrize("filter_value", [True, False, -1, 1.5, [], [1, -1], [True], "1"])
def test_grib_occurrence_filter_invalid(filter_value: T.Any, duplicate_file: str) -> None:
    with pytest.raises(ValueError):
        xr.open_dataset(
            duplicate_file,
            engine="cfgrib",
            backend_kwargs=dict(
                duplicate_messages="stack", filter_by_keys=dict(grib_occurrence=filter_value)
            ),
        )


def test_grib_occurrence_filter_first_mode_rejected(duplicate_file: str) -> None:
    for mode in (None, "first"):
        kwargs = dict(filter_by_keys=dict(grib_occurrence=0))
        if mode is not None:
            kwargs["duplicate_messages"] = mode
        with pytest.raises(ValueError):
            xr.open_dataset(duplicate_file, engine="cfgrib", backend_kwargs=kwargs)
        with pytest.raises(ValueError):
            dataset.open_file(duplicate_file, **kwargs)


def test_grib_occurrence_filter_no_match(duplicate_file: str) -> None:
    with pytest.raises(ValueError):
        xr.open_dataset(
            duplicate_file,
            engine="cfgrib",
            backend_kwargs=dict(
                duplicate_messages="stack", filter_by_keys=dict(grib_occurrence=7)
            ),
        )


def test_grib_occurrence_combines_with_ordinary_keys(duplicate_file: str) -> None:
    kwargs = dict(
        duplicate_messages="stack",
        filter_by_keys=dict(grib_occurrence=1, shortName="t", step=6),
        squeeze=False,
    )
    ds = xr.open_dataset(duplicate_file, engine="cfgrib", backend_kwargs=kwargs)
    assert ds["grib_occurrence"].values.tolist() == [1]
    assert ds["step"].values.tolist() == [6 * 3600 * 10**9]  # 6 hours in ns
    assert_values(ds["t"].values[..., 0, 0], [[[[4.0]]]])

    # a combination that selects no message at all is still an error
    with pytest.raises(ValueError):
        xr.open_dataset(
            duplicate_file,
            engine="cfgrib",
            backend_kwargs=dict(
                duplicate_messages="stack",
                filter_by_keys=dict(grib_occurrence=2, step=0, shortName="z"),
            ),
        )


def test_duplicate_messages_mode_is_validated_early(duplicate_file: str) -> None:
    # the mode is checked even without touching the values, both via xarray and via open_file
    with pytest.raises(ValueError):
        dataset.open_file(duplicate_file, duplicate_messages="")
    with pytest.raises(ValueError):
        dataset.open_fileindex(
            messages.FileStream(duplicate_file), duplicate_messages="everything"
        )
    with pytest.raises(ValueError):
        dataset.open_fieldset([], duplicate_messages=["stack"])


def test_lazy_decoding_only_selected_messages(
    duplicate_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = count_values_decoding(monkeypatch)
    ds = xr.open_dataset(
        duplicate_file,
        engine="cfgrib",
        backend_kwargs=dict(
            duplicate_messages="stack", filter_by_keys=dict(grib_occurrence=2, shortName="t")
        ),
    )
    assert calls == []
    values = ds["t"].isel(step=0, grib_occurrence=0).values
    assert np.all(values == 2.0)
    # only the single selected message for that cell was decoded
    assert len(calls) == 1

    calls.clear()
    stepped = ds["t"].isel(step=1, grib_occurrence=0).values
    assert np.isnan(stepped).all()
    # the empty cell decoded no message at all
    assert calls == []

    calls.clear()
    whole = ds["t"].values
    assert np.isnan(whole[1]).all() and np.all(whole[0] == 2.0)
    assert len(calls) == 1


def test_lazy_subset_by_occurrence_and_step(
    tmpdir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmpdir.join("many.grib2"))
    specs = []
    counter = 0
    for step in range(0, 24, 6):
        for occurrence in range(3):
            counter += 1
            specs.append(dict(shortName="t", step=step, value=float(counter)))
    write_grib(path, specs)

    calls = count_values_decoding(monkeypatch)
    ds = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs=dict(duplicate_messages="stack")
    )
    assert calls == []
    assert ds["grib_occurrence"].values.tolist() == [0, 1, 2]

    # one step, one occurrence -> exactly one message decoded, with the matching value
    calls.clear()
    value = ds["t"].isel(step=3, grib_occurrence=2).values
    assert np.all(value == float(3 * 3 + 3))
    assert len(calls) == 1

    # one step, all occurrences -> three messages decoded
    calls.clear()
    values = ds["t"].isel(step=1).values
    assert_values(values[..., 0, 0], [4.0, 5.0, 6.0])
    assert len(calls) == 3

    # two contiguous steps -> six messages decoded
    calls.clear()
    values = ds["t"].isel(step=slice(0, 2)).values
    assert_values(values[:, 0, 0, 0], [1.0, 4.0])
    assert len(calls) == 6


def test_cold_and_warm_index_are_consistent(
    duplicate_file: str, tmpdir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexpath = str(tmpdir.join("file.idx"))
    # cold index
    cold = xr.open_dataset(
        duplicate_file,
        engine="cfgrib",
        backend_kwargs=dict(duplicate_messages="stack", indexpath=indexpath, squeeze=False),
    )
    cold_labels = cold["grib_occurrence"].values.tolist()
    cold_values = cold["t"].values[..., 0, 0].tolist()
    cold.close()

    # warm index, decoded lazily, must describe exactly the same occurrences
    ds = xr.open_dataset(
        duplicate_file,
        engine="cfgrib",
        backend_kwargs=dict(duplicate_messages="stack", indexpath=indexpath, squeeze=False),
    )
    warm_labels = ds["grib_occurrence"].values.tolist()
    assert warm_labels == cold_labels
    assert_values(ds["t"].values[..., 0, 0], cold_values)

    # the "first" mode builds a different index and must not reuse the stacked one
    first = dataset.open_file(duplicate_file, indexpath=indexpath)
    assert "grib_occurrence" not in first.dimensions
    assert_values(np.asarray(first.variables["t"].data[:, 0, 0]), [1.0, 3.0])


def test_old_index_not_reused_by_stack_mode(duplicate_file: str, tmpdir: str) -> None:
    # simulate an index built before the feature: same keys, no occurrence key
    legacy_indexpath = str(tmpdir.join("legacy.idx"))
    stream = messages.FileStream(duplicate_file)
    keys = dataset.compute_index_keys()
    legacy = messages.FileIndex.from_fieldset(stream, keys)
    with open(legacy_indexpath, "wb") as file:
        import pickle

        pickle.dump(legacy, file)

    ds = dataset.open_file(
        duplicate_file, duplicate_messages="stack", indexpath=legacy_indexpath
    )
    assert ds.dimensions["grib_occurrence"] == 3
    assert_values(
        np.asarray(ds.variables["t"].data[:, :, 0, 0]), [[1.0, 2.0, 2.0], [3.0, 4.0, np.nan]]
    )


def test_multi_field_grib2_message(tmpdir: str) -> None:
    path = write_multi_field_grib(
        str(tmpdir.join("multi.grib2")),
        [
            dict(shortName="t", value=1.0),
            dict(shortName="t", value=2.0),
            dict(shortName="t", value=1.0),
        ],
    )
    ds = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs=dict(duplicate_messages="stack")
    )
    assert ds["grib_occurrence"].values.tolist() == [0, 1, 2]
    assert_values(ds["t"].values[..., 0, 0], [1.0, 2.0, 1.0])

    first = xr.open_dataset(path, engine="cfgrib")
    assert "grib_occurrence" not in first.dims
    assert first["t"].values[..., 0, 0].tolist() == 1.0


def test_open_datasets_with_duplicates(duplicate_file: str) -> None:
    from cfgrib import xarray_store

    opened = xarray_store.open_datasets(
        duplicate_file, backend_kwargs=dict(duplicate_messages="stack")
    )
    by_var = {}
    for ds in opened:
        for name in ds.data_vars:
            by_var.setdefault(name, []).append(ds)
    assert set(by_var) == {"t", "z"}
    t_ds = by_var["t"][0]
    z_ds = by_var["z"][0]
    # variables with different duplicate counts may end up in different datasets, but no
    # variable is lost, labels keep their original ordinals and duplicates are not collapsed
    t_labels = t_ds["grib_occurrence"].values.tolist()
    z_labels = z_ds["grib_occurrence"].values.tolist()
    assert t_labels == [0, 1, 2]
    assert z_labels == [0]
    assert set(z_labels) <= set(t_labels)
    assert_values(z_ds["z"].values[:, 0, 0, 0], [9.0, 10.0])
    assert_values(t_ds["t"].values[..., 0, 0], [[1.0, 2.0, 2.0], [3.0, 4.0, np.nan]])


def test_different_grids_still_split(tmpdir: str) -> None:
    path = str(tmpdir.join("grids.grib2"))
    write_grib(
        path,
        [
            dict(shortName="t", value=1.0),
            dict(shortName="t", value=2.0),
            dict(shortName="t", value=3.0, Ni=8, Nj=6),
        ],
    )
    # a different grid definition cannot live in the same cube: opening the merged file fails
    with pytest.raises(dataset.DatasetBuildError):
        dataset.open_file(path, duplicate_messages="stack")
    with pytest.raises(dataset.DatasetBuildError):
        dataset.open_file(path)
    # open_datasets cannot merge two incompatible grids either, in both modes alike
    from cfgrib import xarray_store

    with pytest.raises(ValueError):
        xarray_store.open_datasets(path, backend_kwargs=dict(duplicate_messages="stack"))
    with pytest.raises(ValueError):
        xarray_store.open_datasets(path)

def test_open_datasets_occurrence_filter_keeps_variables(duplicate_file: str) -> None:
    from cfgrib import xarray_store

    # occurrence 2 exists for t but not for z; z must not be dropped and must stay all-NaN
    opened = xarray_store.open_datasets(
        duplicate_file,
        backend_kwargs=dict(
            duplicate_messages="stack", filter_by_keys=dict(grib_occurrence=2)
        ),
    )
    by_var = {}
    for ds in opened:
        for name in ds.data_vars:
            by_var[name] = ds
    assert set(by_var) == {"t", "z"}
    for name, ds in by_var.items():
        assert ds["grib_occurrence"].values.tolist() == [2]
    assert_values(by_var["t"]["t"].values[:, 0, 0, 0], [2.0, np.nan])
    assert np.isnan(by_var["z"]["z"].values).all()

    # occurrence 0 exists for both and they are merged into a single dataset
    opened = xarray_store.open_datasets(
        duplicate_file,
        backend_kwargs=dict(
            duplicate_messages="stack", filter_by_keys=dict(grib_occurrence=0)
        ),
    )
    assert len(opened) == 1
    assert set(opened[0].data_vars) == {"t", "z"}
    assert opened[0]["grib_occurrence"].values.tolist() == [0]
