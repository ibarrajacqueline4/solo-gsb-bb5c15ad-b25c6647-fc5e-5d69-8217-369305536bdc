"""Structural shape inventory and the content-independent shape hash.

Purpose
-------
Callers sometimes need to know whether two JSON documents have the *same
shape*: the same nested container structure, the same object member names,
array elements in the same order and scalar/empty-container leaves of the
same kinds — while differing freely in scalar leaf values.  :func:`shape_document`
powers ``POST /shape``: it lists every node as an RFC 6901 pointer plus its
JSON type (leaf values omitted) and computes a ``shapeHash`` that is equal
exactly for same-shaped documents.

Node types are the six JSON types of RFC 8259:

``object``  ``array``  ``string``  ``number``  ``boolean``  ``null``

All JSON numbers (integer or float) share ``number`` and ``true``/``false``
share ``boolean``; empty ``{}`` / ``[]`` are object/array *leaf* nodes with
their own hash domain.

Bottom-up preimages (all hashes SHA-256, 32 bytes)
--------------------------------------------------

Scalar and empty-container leaves hash from their one-byte type tag alone;
scalar values never enter the preimage::

    null leaf          0x10
    boolean leaf       0x11   (true and false alike)
    number leaf        0x12   (integer or float, any magnitude)
    string leaf        0x13   (any content, including "")
    empty object {}    0x14
    empty array []     0x15

Non-empty containers carry a u32be child count.  Object member names are
sorted by Unicode code point (Python ``str`` ordering, the same rule as the
canonical JSON encoder) and length-prefixed; every child then follows as its
fixed 32-byte digest::

    object node        0x20
                       || u32be(n_members)
                       || for each member in code-point key order:
                              u32be(len(key_utf8)) || key_utf8
                              || child_digest(32)

    array node         0x21
                       || u32be(n_elements)
                       || child_digest(32) * n_elements   (array order kept)

The tag bytes domain-separate node kinds so one kind of node can never be
read as another, and the count/key length prefixes (plus the fixed digest
width) frame every section unambiguously — two distinct shapes cannot
serialize to the same byte string.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List

from . import pointers

HASH_LEN = 32
U32BE = struct.Struct(">I")

# Leaf / scalar domain tags (single-byte preimages).
TAG_NULL = b"\x10"
TAG_BOOLEAN = b"\x11"
TAG_NUMBER = b"\x12"
TAG_STRING = b"\x13"
TAG_EMPTY_OBJECT = b"\x14"
TAG_EMPTY_ARRAY = b"\x15"
# Non-empty container domain tags.
TAG_OBJECT = b"\x20"
TAG_ARRAY = b"\x21"

# Public wire names for node types (the six RFC 8259 JSON types).
TYPE_OBJECT = "object"
TYPE_ARRAY = "array"
TYPE_STRING = "string"
TYPE_NUMBER = "number"
TYPE_BOOLEAN = "boolean"
TYPE_NULL = "null"
NODE_TYPES = (
    TYPE_OBJECT,
    TYPE_ARRAY,
    TYPE_STRING,
    TYPE_NUMBER,
    TYPE_BOOLEAN,
    TYPE_NULL,
)


class ShapeError(ValueError):
    """Raised when a value is not a hashable JSON node."""


def _u32be_len(data: bytes) -> bytes:
    return U32BE.pack(len(data))


def _scalar_tag(value: Any) -> bytes:
    """Return the single-byte leaf tag for a JSON scalar."""
    # bool is a subclass of int, so it must be tested before numbers.
    if value is None:
        return TAG_NULL
    if isinstance(value, bool):
        return TAG_BOOLEAN
    if isinstance(value, (int, float)):
        return TAG_NUMBER
    if isinstance(value, str):
        return TAG_STRING
    raise ShapeError(f"unsupported scalar type {type(value).__name__}")


def _node_type(value: Any) -> str:
    if isinstance(value, bool):
        return TYPE_BOOLEAN
    if value is None:
        return TYPE_NULL
    if isinstance(value, (int, float)):
        return TYPE_NUMBER
    if isinstance(value, str):
        return TYPE_STRING
    if isinstance(value, dict):
        return TYPE_OBJECT
    if isinstance(value, list):
        return TYPE_ARRAY
    raise ShapeError(f"unsupported value type {type(value).__name__}")


def shape_document(doc: Any) -> Dict[str, Any]:
    """Return ``{"shapeHash", "nodes"}`` for any JSON value.

    ``nodes`` lists every node (root included) as ``{"path", "type"}`` with
    no leaf values, sorted by RFC 6901 pointer.  ``shapeHash`` is the
    lowercase-hex SHA-256 root digest computed bottom-up under the encoding
    documented above.
    """

    nodes: List[Dict[str, str]] = []

    def walk(value: Any, pointer: str) -> bytes:
        if isinstance(value, dict):
            kind = TYPE_OBJECT
            if not value:
                preimage = TAG_EMPTY_OBJECT
            else:
                parts: List[bytes] = [TAG_OBJECT, U32BE.pack(len(value))]
                for key in sorted(value.keys()):
                    child_pointer = pointers.build_pointer(pointer, key)
                    child_digest = walk(value[key], child_pointer)
                    try:
                        key_bytes = key.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        # JSON parsed normally never yields lone surrogates;
                        # validate_json_value rejects them before this point.
                        raise ShapeError(
                            f"object key at {pointer!r} cannot be encoded as "
                            f"UTF-8: {exc.reason}"
                        ) from exc
                    parts.append(_u32be_len(key_bytes))
                    parts.append(key_bytes)
                    parts.append(child_digest)
                preimage = b"".join(parts)
        elif isinstance(value, list):
            kind = TYPE_ARRAY
            if not value:
                preimage = TAG_EMPTY_ARRAY
            else:
                parts = [TAG_ARRAY, U32BE.pack(len(value))]
                for index, child in enumerate(value):
                    parts.append(walk(child, pointers.build_pointer(pointer, index)))
                preimage = b"".join(parts)
        else:
            kind = _node_type(value)
            preimage = _scalar_tag(value)

        nodes.append({"path": pointer, "type": kind})
        return hashlib.sha256(preimage).digest()

    root_digest = walk(doc, "")
    nodes.sort(key=lambda node: node["path"])
    return {"shapeHash": root_digest.hex(), "nodes": nodes}
