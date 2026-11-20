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
#

import contextlib
import hashlib
import logging
import numbers
import os
import pickle
import typing as T

import attr
import eccodes  # type: ignore
import numpy as np

from . import abc

MISSING_VAUE_INDICATOR = np.finfo(np.float32).max

eccodes_version = eccodes.codes_get_api_version()

LOG = logging.getLogger(__name__)
_MARKER = object()

#
# MULTI-FIELD support is very tricky. Random access via the index needs multi support to be off.
#
eccodes.codes_grib_multi_support_off()


@contextlib.contextmanager
def multi_enabled(file: T.IO[bytes]) -> T.Iterator[None]:
    """Context manager that enables MULTI-FIELD support in ecCodes from a clean state"""
    eccodes.codes_grib_multi_support_on()
    #
    # Explicitly reset the multi_support global state that gets confused by random access
    #
    # @alexamici: I'm note sure this is thread-safe. See :#141
    #
    eccodes.codes_grib_multi_support_reset_file(file)
    try:
        yield
    except Exception:
        eccodes.codes_grib_multi_support_off()
        raise
    eccodes.codes_grib_multi_support_off()


KEY_TYPES = {
    "float": float,
    "int": int,
    "str": str,
    "": None,
}

DEFAULT_INDEXPATH = "{path}.{short_hash}.idx"
DEFAULT_VALUES_DTYPE = np.dtype("float32")

#
# Virtual key and modes used to keep more than one GRIB message per set of header coordinates.
#
GRIB_OCCURRENCE_KEY = "grib_occurrence"
DUPLICATE_MESSAGES_FIRST = "first"
DUPLICATE_MESSAGES_STACK = "stack"
DUPLICATE_MESSAGES_MODES = (DUPLICATE_MESSAGES_FIRST, DUPLICATE_MESSAGES_STACK)

OffsetType = T.Union[int, T.Tuple[int, int]]


def resolve_duplicate_messages(value: T.Any) -> str:
    """Validate and normalise the ``duplicate_messages`` opening option."""
    if isinstance(value, str) and value in DUPLICATE_MESSAGES_MODES:
        return value
    raise ValueError(
        "duplicate_messages must be one of %r, got %r"
        % (list(DUPLICATE_MESSAGES_MODES), value)
    )


def _validate_occurrence_scalar(value: T.Any) -> int:
    # NOTE: bool is an int subclass, but it is not a valid occurrence number
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("grib_occurrence values must be non-negative integers, got %r" % (value,))
    if value < 0:
        raise ValueError("grib_occurrence values must be non-negative integers, got %r" % (value,))
    return int(value)


def validate_grib_occurrence_filter(value: T.Any) -> T.List[int]:
    """Validate the value of a ``grib_occurrence`` filter, returning a list of ordinals."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("grib_occurrence filter cannot be an empty list")
        return [_validate_occurrence_scalar(v) for v in value]
    return [_validate_occurrence_scalar(value)]


def grib_occurrence_filter_key(filter_by_keys: T.Mapping[str, T.Any]) -> T.Optional[str]:
    """Return the key used in ``filter_by_keys`` for the occurrence filter, if any."""
    for key in filter_by_keys:
        if key.partition(":")[0] == GRIB_OCCURRENCE_KEY:
            return key
    return None


def normalize_filter_by_keys(
    filter_by_keys: T.Mapping[str, T.Any], duplicate_messages: T.Any
) -> T.Tuple[str, T.Dict[str, T.Any]]:
    """Validate the duplicate-messages mode together with the ``filter_by_keys`` mapping.

    Returns the normalised mode and a copy of ``filter_by_keys`` where the occurrence filter,
    if any, has been validated and stored under its plain virtual key.
    """
    mode = resolve_duplicate_messages(duplicate_messages)
    normalized = dict(filter_by_keys)
    occurrence_key = grib_occurrence_filter_key(normalized)
    if occurrence_key is not None:
        if mode != DUPLICATE_MESSAGES_STACK:
            raise ValueError(
                "filter_by_keys %r requires duplicate_messages=%r"
                % (occurrence_key, DUPLICATE_MESSAGES_STACK)
            )
        normalized[GRIB_OCCURRENCE_KEY] = validate_grib_occurrence_filter(
            normalized.pop(occurrence_key)
        )
    return mode, normalized


@attr.attrs(auto_attribs=True)
class Message(abc.MutableField):
    """Dictionary-line interface to access Message headers."""

    codes_id: int
    encoding: str = "ascii"
    errors: str = attr.attrib(
        default="warn", validator=attr.validators.in_(["ignore", "warn", "raise"])
    )

    @classmethod
    def from_file(cls, file, offset=None, **kwargs):
        # type: (T.IO[bytes], T.Optional[OffsetType], T.Any) -> Message
        field_in_message = 0
        if isinstance(offset, tuple):
            offset, field_in_message = offset
        if offset is not None:
            file.seek(offset)
        codes_id = None
        if field_in_message == 0:
            codes_id = eccodes.codes_grib_new_from_file(file)
        else:
            # MULTI-FIELD is enabled only when accessing additional fields
            with multi_enabled(file):
                for _ in range(field_in_message + 1):
                    codes_id = eccodes.codes_grib_new_from_file(file)

        if codes_id is None:
            raise EOFError("End of file: %r" % file)
        return cls(codes_id=codes_id, **kwargs)

    @classmethod
    def from_sample_name(cls, sample_name, **kwargs):
        # type: (str, T.Any) -> Message
        codes_id = eccodes.codes_new_from_samples(sample_name, eccodes.CODES_PRODUCT_GRIB)
        return cls(codes_id=codes_id, **kwargs)

    @classmethod
    def from_message(cls, message, **kwargs):
        # type: (Message, T.Any) -> Message
        codes_id = eccodes.codes_clone(message.codes_id)
        return cls(codes_id=codes_id, **kwargs)

    # ensure that missing values in the values array are represented by MISSING_VAUE_INDICATOR
    def __attrs_post_init__(self):
        self["missingValue"] = MISSING_VAUE_INDICATOR

    def __del__(self) -> None:
        eccodes.codes_release(self.codes_id)

    def message_get(self, item, key_type=None, default=_MARKER):
        # type: (str, T.Optional[type], T.Any) -> T.Any
        """Get value of a given key as its native or specified type."""
        try:
            if eccodes.codes_get_size(self.codes_id, item) > 1:
                values = eccodes.codes_get_array(self.codes_id, item, key_type)
            else:
                values = [eccodes.codes_get(self.codes_id, item, key_type)]

            if values is None:
                return "unsupported_key_type"

            if len(values) == 1:
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                return values[0]
            return values

        except eccodes.KeyValueNotFoundError:
            if default is _MARKER:
                raise KeyError(item)
            else:
                return default

    def message_set(self, item: str, value: T.Any) -> None:
        arr = isinstance(value, (np.ndarray, T.Sequence)) and not isinstance(value, str)
        if arr:
            eccodes.codes_set_array(self.codes_id, item, value)
        else:
            eccodes.codes_set(self.codes_id, item, value)

    def message_grib_keys(self, namespace: T.Optional[str] = None) -> T.Iterator[str]:
        iterator = eccodes.codes_keys_iterator_new(self.codes_id, namespace=namespace)
        while eccodes.codes_keys_iterator_next(iterator):
            yield eccodes.codes_keys_iterator_get_name(iterator)
        eccodes.codes_keys_iterator_delete(iterator)

    def __getitem__(self, item: str) -> T.Any:
        key, _, key_type_text = item.partition(":")
        if key_type_text not in KEY_TYPES:
            raise ValueError("key type not supported %r" % key_type_text)
        key_type = KEY_TYPES[key_type_text]
        return self.message_get(key, key_type=key_type)

    def __setitem__(self, item: str, value: T.Any) -> None:
        try:
            return self.message_set(item, value)
        except eccodes.GribInternalError as ex:
            if self.errors == "ignore":
                pass
            elif self.errors == "raise":
                raise KeyError("failed to set key %r to %r" % (item, value))
            else:
                if isinstance(ex, eccodes.ReadOnlyError):
                    # Very noisy error when trying to set computed keys
                    pass
                else:
                    LOG.warning("failed to set key %r to %r", item, value)

    def __delitem__(self, item: str) -> None:
        raise NotImplementedError

    def __iter__(self) -> T.Iterator[str]:
        for key in self.message_grib_keys():
            yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def write(self, file: T.IO[bytes]) -> None:
        eccodes.codes_write(self.codes_id, file)


GetterType = T.Callable[..., T.Any]
SetterType = T.Callable[..., None]
ComputedKeysType = T.Dict[str, T.Tuple[GetterType, SetterType]]


@attr.attrs(auto_attribs=True)
class ComputedKeysMessage(Message):
    """Extension of Message class for adding computed keys."""

    computed_keys: ComputedKeysType = {}

    def __getitem__(self, item: str) -> T.Any:
        if item in self.computed_keys:
            getter, _ = self.computed_keys[item]
            return getter(self)
        else:
            return super(ComputedKeysMessage, self).__getitem__(item)

    def __iter__(self) -> T.Iterator[str]:
        seen = set()
        for key in super(ComputedKeysMessage, self).__iter__():
            yield key
            seen.add(key)
        for key in self.computed_keys:
            if key not in seen:
                yield key

    def __setitem__(self, item: str, value: T.Any) -> None:
        if item in self.computed_keys:
            _, setter = self.computed_keys[item]
            return setter(self, value)
        else:
            return super(ComputedKeysMessage, self).__setitem__(item, value)


@attr.attrs(auto_attribs=True)
class ComputedKeysAdapter(abc.Field):
    """Extension of Message class for adding computed keys."""

    context: abc.Field
    computed_keys: ComputedKeysType = {}

    def __getitem__(self, item: str) -> T.Any:
        if item in self.computed_keys:
            getter, _ = self.computed_keys[item]
            return getter(self)
        else:
            return self.context[item]

    def __iter__(self) -> T.Iterator[str]:
        seen = set()
        for key in self.context:
            yield key
            seen.add(key)
        for key in self.computed_keys:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(self.context)


class FileStreamItems(T.ItemsView[OffsetType, Message]):
    def __init__(self, filestream: "FileStream"):
        self.filestream = filestream

    def itervalues(self) -> T.Iterator[Message]:
        errors = self.filestream.errors
        with open(self.filestream.path, "rb") as file:
            # enable MULTI-FIELD support on sequential reads (like when building the index)
            with multi_enabled(file):
                valid_message_found = False
                while True:
                    try:
                        yield self.filestream.message_from_file(file, errors=errors)
                        valid_message_found = True
                    except EOFError:
                        if not valid_message_found:
                            raise EOFError("No valid message found: %r" % self.filestream.path)
                        break
                    except Exception:
                        if errors == "ignore":
                            pass
                        elif errors == "raise":
                            raise
                        else:
                            LOG.exception("skipping corrupted Message")

    def __iter__(self) -> T.Iterator[T.Tuple[OffsetType, Message]]:
        # assumes MULTI-FIELD support in self.itervalues()
        old_offset = -1
        count = 0
        for message in self.itervalues():
            offset = message.message_get("offset", int)
            if offset == old_offset:
                count += 1
                offset_field = (offset, count)
            else:
                old_offset = offset
                count = 0
                offset_field = offset
            yield (offset_field, message)


@attr.attrs(auto_attribs=True)
class FileStream(abc.MappingFieldset[OffsetType, Message]):
    """Mapping-like access to a filestream of Messages.

    Sample usage:

    >>> filestream = FileStream("era5-levels-members.grib")
    >>> message1 = filestream[None]
    >>> message1["offset"]
    0.0
    >>> message2 = filestream[14760]
    >>> message2["offset"]
    14760.0

    Note that any offset return the first message found _after_ that offset:

    >>> message2_again = filestream[1]
    >>> message2_again["offset"]
    14760.0
    """

    path: str
    errors: str = attr.attrib(
        default="warn", validator=attr.validators.in_(["ignore", "warn", "raise"])
    )

    #
    # NOTE: we implement `.items()`, and not `__iter__()`, as a performance optimisation
    #
    def items(self) -> T.ItemsView[OffsetType, Message]:
        return FileStreamItems(self)

    def __iter__(self) -> T.Iterator[OffsetType]:
        raise NotImplementedError("use `.items()` instead")

    def message_from_file(self, file, offset=None, **kwargs):
        # type: (T.IO[bytes], T.Optional[OffsetType], T.Any) -> Message
        return Message.from_file(file, offset, **kwargs)

    def __getitem__(self, item: T.Optional[OffsetType]) -> Message:
        with open(self.path, "rb") as file:
            return self.message_from_file(file, offset=item)

    def __len__(self) -> int:
        return sum(1 for _ in self.items())


#
# NOTE: the on-disk index is keyed by the exact list of ``index_keys``. An index built before
# ``grib_occurrence`` existed does not carry that key, so it can never be reused in "stack"
# mode: the requested keys mismatch and the index is rebuilt from the GRIB file. This keeps the
# legacy index files usable in "first" mode without any change.
#
ALLOWED_PROTOCOL_VERSION = "2"


C = T.TypeVar("C", bound="FieldsetIndex")


@attr.attrs(auto_attribs=True)
class FieldsetIndex(abc.Index[T.Any, abc.Field]):
    fieldset: T.Union[abc.Fieldset[abc.Field], abc.MappingFieldset[T.Any, abc.Field]]
    index_keys: T.List[str]
    filter_by_keys: T.Dict[str, T.Any] = {}
    field_ids_index: T.List[T.Tuple[T.Tuple[T.Any, ...], T.List[abc.Field]]] = attr.attrib(
        repr=False, default=[]
    )
    computed_keys: ComputedKeysType = {}
    index_protocol_version: str = ALLOWED_PROTOCOL_VERSION
    # NOTE: set on subindices created by a ``grib_occurrence`` filter and holds the coordinate
    #   range selected by the ordinary GRIB conditions only.
    range_index: T.Optional[T.Any] = attr.attrib(default=None, repr=False, eq=False)

    @classmethod
    def from_fieldset(
        cls: T.Type[C],
        fieldset: T.Union[abc.Fieldset[abc.Field], abc.MappingFieldset[T.Any, abc.Field]],
        index_keys: T.Sequence[str],
        computed_keys: ComputedKeysType = {},
    ) -> C:
        if isinstance(fieldset, T.Mapping):
            iteritems = iter(fieldset.items())
        else:
            iteritems = enumerate(fieldset)
        return cls.from_fieldset_and_iteritems(fieldset, iteritems, index_keys, computed_keys)

    @classmethod
    def from_fieldset_and_iteritems(
        cls: T.Type[C],
        fieldset: T.Union[abc.Fieldset[abc.Field], abc.MappingFieldset[T.Any, abc.Field]],
        iteritems: T.Iterable[T.Tuple[T.Any, abc.Field]],
        index_keys: T.Sequence[str],
        computed_keys: ComputedKeysType = {},
    ) -> C:
        field_ids_index = {}  # type: T.Dict[T.Tuple[T.Any, ...], T.List[T.Any]]
        index_keys = list(index_keys)
        header_values_cache = {}  # type: T.Dict[T.Tuple[T.Any, type], T.Any]
        occurrence_index = (
            index_keys.index(GRIB_OCCURRENCE_KEY) if GRIB_OCCURRENCE_KEY in index_keys else None
        )
        occurrence_counters = {}  # type: T.Dict[T.Tuple[T.Any, ...], int]
        for field_id, raw_field in iteritems:
            field = ComputedKeysAdapter(raw_field, computed_keys)
            header_values = []
            for key in index_keys:
                if key == GRIB_OCCURRENCE_KEY:
                    # NOTE: the value is computed below from the other header values
                    header_values.append(None)
                    continue
                try:
                    try:
                        value = field[key]
                    except KeyError:
                        # get default type if Field does not support type specifier
                        if ":" not in key:
                            raise
                        else:
                            value = field[key.partition(":")[0]]
                    if value is None:
                        value = "undef"
                except Exception:
                    value = "undef"
                if isinstance(value, (np.ndarray, list)):
                    value = tuple(value)
                # NOTE: the following ensures that values of the same type that evaluate equal are
                #   exactly the same object. The optimisation is especially useful for strings and
                #   it also reduces the on-disk size of the index in a backward compatible way.
                value = header_values_cache.setdefault((value, type(value)), value)
                header_values.append(value)
            if occurrence_index is not None:
                # NOTE: the occurrence ordinal is local to a single group of non-spatial
                #   header values and follows the order the messages appear in the source.
                cell_key = tuple(header_values)
                occurrence = occurrence_counters.get(cell_key, 0)
                occurrence_counters[cell_key] = occurrence + 1
                header_values[occurrence_index] = occurrence
            field_ids_index.setdefault(tuple(header_values), []).append(field_id)
        self = cls(
            fieldset,
            index_keys,
            field_ids_index=list(field_ids_index.items()),
            computed_keys=computed_keys,
        )
        # record the index protocol version in the instance so it is dumped with pickle
        return self

    @classmethod
    def from_indexpath(cls, indexpath):
        # type: (T.Type[C], str) -> C
        with open(indexpath, "rb") as file:
            index = pickle.load(file)
            if not isinstance(index, cls):
                raise ValueError("on-disk index not of expected type {cls}")
            if index.index_protocol_version != ALLOWED_PROTOCOL_VERSION:
                raise ValueError("protocol versione to allowed {index.index_protocol_version}")
            return index

    def __iter__(self) -> T.Iterator[str]:
        return iter(self.index_keys)

    def __len__(self) -> int:
        return len(self.index_keys)

    @property
    def header_values(self) -> T.Dict[str, T.List[T.Any]]:
        if not hasattr(self, "_header_values"):
            all_header_values = {}  # type: T.Dict[str, T.Dict[T.Any, None]]
            for header_values, _ in self.field_ids_index:
                for i, value in enumerate(header_values):
                    values = all_header_values.setdefault(self.index_keys[i], {})
                    if value not in values:
                        values[value] = None
            self._header_values = {k: list(v) for k, v in all_header_values.items()}
        return self._header_values

    def __getitem__(self, item: str) -> T.List[T.Any]:
        return self.header_values[item]

    def getone(self, item):
        # type: (str) -> T.Any
        values = self[item]
        if len(values) != 1:
            raise ValueError("not one value for %r: %r" % (item, len(values)))
        return values[0]

    def subindex(self, filter_by_keys={}, strict_occurrence=True, **query):
        # type: (C, T.Mapping[str, T.Any], bool, T.Any) -> C
        query.update(filter_by_keys)
        occurrence_key = grib_occurrence_filter_key(query)
        if occurrence_key is not None:
            if GRIB_OCCURRENCE_KEY not in self.index_keys:
                raise ValueError(
                    "filter_by_keys %r requires duplicate_messages=%r"
                    % (occurrence_key, DUPLICATE_MESSAGES_STACK)
                )
            # NOTE: the occurrence filter uses the plain virtual key and keeps the original
            #   ordinal as the coordinate label, it never renumbers the selected messages.
            query[GRIB_OCCURRENCE_KEY] = validate_grib_occurrence_filter(
                query.pop(occurrence_key)
            )
        ordinary_query = {
            k: v for k, v in query.items() if k.partition(":")[0] != GRIB_OCCURRENCE_KEY
        }
        field_ids_index = self._filter_field_ids_index(query)
        if occurrence_key is not None and not field_ids_index and strict_occurrence:
            raise ValueError(
                "no message matches filter_by_keys %r"
                % {occurrence_key: query[GRIB_OCCURRENCE_KEY]}
            )
        index = type(self)(
            fieldset=self.fieldset,
            index_keys=self.index_keys,
            field_ids_index=field_ids_index,
            filter_by_keys=query,
        )
        # NOTE: remember the coordinate range selected by the ordinary GRIB conditions only,
        #   so that a cell missing a selected occurrence keeps its coordinates (NaN) instead of
        #   vanishing from the data cube.
        previous_range = getattr(self, "range_index", None)
        if previous_range is not None or occurrence_key is not None:
            base = self if previous_range is None else previous_range
            index.range_index = base.subindex(ordinary_query, strict_occurrence=False)
        return index

    def _filter_field_ids_index(
        self, query: T.Mapping[str, T.Any]
    ) -> T.List[T.Tuple[T.Tuple[T.Any, ...], T.List[T.Any]]]:
        raw_query = [(self.index_keys.index(k), v) for k, v in query.items()]
        field_ids_index = []
        for header_values, field_ids_values in self.field_ids_index:
            for idx, val in raw_query:
                # Ensure that the values to be tested is a list or tuple
                if not isinstance(val, (list, tuple)):
                    val = [val]
                if header_values[idx] not in val:
                    break
            else:
                field_ids_index.append((header_values, field_ids_values))
        return field_ids_index

    def get_field(self, message_id: T.Any) -> abc.Field:
        return ComputedKeysAdapter(self.fieldset[message_id], self.computed_keys)

    def first(self) -> abc.Field:
        first_message_id = self.field_ids_index[0][1][0]
        return self.get_field(first_message_id)

    def source(self) -> str:
        return "N/A"

    def iter_index(self) -> T.Iterator[T.Tuple[T.Tuple[T.Any, ...], T.List[T.Any]]]:
        return iter(self.field_ids_index)


@contextlib.contextmanager
def compat_create_exclusive(path):
    # type: (str) -> T.Generator[T.IO[bytes], None, None]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with open(fd, mode="wb") as file:
        try:
            yield file
        except Exception:
            file.close()
            os.unlink(path)
            raise


@attr.attrs(auto_attribs=True)
class FileIndex(FieldsetIndex):
    fieldset: FileStream
    index_keys: T.List[str]
    filter_by_keys: T.Dict[str, T.Any] = {}
    field_ids_index: T.List[T.Tuple[T.Tuple[T.Any, ...], T.List[T.Any]]] = attr.attrib(
        repr=False, default=[]
    )
    computed_keys: ComputedKeysType = {}
    index_protocol_version: str = ALLOWED_PROTOCOL_VERSION
    # NOTE: set on subindices created by a ``grib_occurrence`` filter and holds the coordinate
    #   range selected by the ordinary GRIB conditions only.
    range_index: T.Optional[T.Any] = attr.attrib(default=None, repr=False, eq=False)

    @classmethod
    def from_indexpath_or_filestream(
        cls, filestream, index_keys, indexpath=DEFAULT_INDEXPATH, computed_keys={}, log=LOG
    ):
        # type: (FileStream, T.Sequence[str], str, ComputedKeysType, logging.Logger) -> FileIndex

        # Reading and writing the index can be explicitly suppressed by passing indexpath==''.
        if not indexpath:
            return cls.from_fieldset(filestream, index_keys, computed_keys)
        repr_index_keys = repr(index_keys).encode("utf-8")
        try:
            hash = hashlib.md5(repr_index_keys, usedforsecurity=False).hexdigest()
        except TypeError:
            hash = hashlib.md5(repr_index_keys).hexdigest()
        indexpath = indexpath.format(path=filestream.path, hash=hash, short_hash=hash[:5])
        try:
            with compat_create_exclusive(indexpath) as new_index_file:
                self = cls.from_fieldset(filestream, index_keys, computed_keys)
                pickle.dump(self, new_index_file)
                return self
        except FileExistsError:
            pass
        except Exception:
            log.exception("Can't create file %r", indexpath)

        try:
            index_mtime = os.path.getmtime(indexpath)
            filestream_mtime = os.path.getmtime(filestream.path)
            if index_mtime >= filestream_mtime:
                self = cls.from_indexpath(indexpath)
                if (
                    getattr(self, "index_keys", None) == index_keys
                    and getattr(self, "fieldset", None) == filestream
                    and getattr(self, "index_protocol_version", None) == ALLOWED_PROTOCOL_VERSION
                ):
                    return self
                else:
                    log.warning("Ignoring index file %r incompatible with GRIB file", indexpath)
            else:
                log.warning("Ignoring index file %r older than GRIB file", indexpath)
        except Exception:
            log.exception("Can't read index file %r", indexpath)

        return cls.from_fieldset(filestream, index_keys, computed_keys)

    def source(self) -> str:
        try:
            return os.path.relpath(self.fieldset.path)
        except ValueError:
            return os.path.basename(self.fieldset.path)
