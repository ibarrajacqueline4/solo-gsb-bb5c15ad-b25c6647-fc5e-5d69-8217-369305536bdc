"""Unit tests for the value-free shape inventory and shape hash."""

import hashlib
import struct

import pytest

from app.shape import (
    ShapeError,
    shape_document,
)


def _h(tag_or_bytes: bytes) -> str:
    return hashlib.sha256(tag_or_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Leaf (scalar / empty container) preimages
# ---------------------------------------------------------------------------


def test_scalar_leaf_hashes_are_single_domain_tags():
    assert shape_document(None)["shapeHash"] == _h(b"\x10")
    assert shape_document(True)["shapeHash"] == _h(b"\x11")
    assert shape_document(False)["shapeHash"] == _h(b"\x11")
    assert shape_document(7)["shapeHash"] == _h(b"\x12")
    assert shape_document(-3.25)["shapeHash"] == _h(b"\x12")
    assert shape_document("anything")["shapeHash"] == _h(b"\x13")
    assert shape_document("")["shapeHash"] == _h(b"\x13")
    assert shape_document({})["shapeHash"] == _h(b"\x14")
    assert shape_document([])["shapeHash"] == _h(b"\x15")


def test_six_scalar_kinds_hash_pairwise_distinct():
    leaves = [None, True, 0, ""]
    digests = {shape_document(value)["shapeHash"] for value in leaves}
    assert len(digests) == len(leaves)
    assert _h(b"\x14") not in digests and _h(b"\x15") not in digests


def test_numbers_and_booleans_collapse_across_representations():
    assert shape_document(1)["shapeHash"] == shape_document(1.0)["shapeHash"]
    assert shape_document(0)["shapeHash"] == shape_document(10**40)["shapeHash"]
    assert shape_document(True)["shapeHash"] == shape_document(False)["shapeHash"]
    # bool must not be classified as number.
    assert (
        shape_document(True)["shapeHash"] != shape_document(0)["shapeHash"]
    )


# ---------------------------------------------------------------------------
# Container preimages: domain tag + count + framed children
# ---------------------------------------------------------------------------


def test_non_empty_object_preimage_uses_tag_count_length_prefixed_keys():
    child = hashlib.sha256(b"\x12").digest()
    key = b"a"
    expected_preimage = (
        b"\x20"
        + struct.pack(">I", 1)          # member count
        + struct.pack(">I", len(key))  # key byte length
        + key
        + child
    )
    assert shape_document({"a": 1})["shapeHash"] == _h(expected_preimage)


def test_non_empty_array_preimage_uses_tag_count_and_ordered_digests():
    number = hashlib.sha256(b"\x12").digest()
    string = hashlib.sha256(b"\x13").digest()
    expected_preimage = b"\x21" + struct.pack(">I", 2) + number + string
    assert shape_document([1, "x"])["shapeHash"] == _h(expected_preimage)


def test_object_member_names_participate_in_the_hash():
    assert (
        shape_document({"a": 1})["shapeHash"]
        != shape_document({"b": 1})["shapeHash"]
    )
    assert (
        shape_document({"a": 1})["shapeHash"]
        != shape_document({"a": 1, "b": 2})["shapeHash"]
    )


def test_length_prefixes_prevent_key_boundary_ambiguity():
    # Without the u32be key-length framing these member-name sequences could
    # be concatenated to the same bytes; with it the hashes must differ.
    assert (
        shape_document({"ab": 1, "c": 2})["shapeHash"]
        != shape_document({"a": 1, "bc": 2})["shapeHash"]
    )


def test_object_keys_ordered_by_code_point_regardless_of_insertion():
    first = shape_document({"b": 1, "a": [1, 2]})["shapeHash"]
    second = shape_document({"a": [9, 8], "b": 7})["shapeHash"]
    assert first == second

    unicode_first = shape_document(
        {"中": 1, "a": {"x": [None]}, "€": 2.5}
    )["shapeHash"]
    unicode_second = shape_document(
        {"€": 99, "a": {"x": [None]}, "中": 7}
    )["shapeHash"]
    assert unicode_first == unicode_second


def test_array_order_of_node_types_is_significant():
    assert (
        shape_document([1, "x"])["shapeHash"]
        != shape_document(["x", 1])["shapeHash"]
    )
    # Order of equal-typed *values* is not, since values never hash in.
    assert (
        shape_document([1, 2])["shapeHash"]
        == shape_document([2, 1])["shapeHash"]
    )


def test_empty_containers_distinct_from_each_other_and_non_empty():
    shapes = [shape_document(v)["shapeHash"] for v in ({}, [], {"a": 1}, [1])]
    assert len(set(shapes)) == 4


def test_shape_hash_is_bottom_up_and_content_independent():
    before = {
        "name": "Ada",
        "age": 36,
        "tags": ["x", True],
        "emptyObj": {},
        "emptyArr": [],
        "nickname": None,
    }
    after = {
        "name": "Grace",
        "age": 85.5,
        "tags": ["y", False],
        "emptyObj": {},
        "emptyArr": [],
        "nickname": None,
    }
    assert shape_document(before)["shapeHash"] == shape_document(after)["shapeHash"]


def test_structural_differences_change_the_hash():
    base = {"a": 1, "b": {"c": [1, 2]}}
    variants = [
        {"a": 1, "b": {"c": [1]}},               # shorter array
        {"a": 1, "b": {"c": [1, 2], "d": 0}},    # extra member
        {"a": 1, "b": {"c": [1, "x"]}},          # changed element type
        {"a": 1, "b": [1, 2]},                   # object -> array
        [1, {"c": [1, 2]}],                      # root object -> array
        {"a": 1, "b": {"c": [1, 2]}, "e": None},
    ]
    base_hash = shape_document(base)["shapeHash"]
    for variant in variants:
        assert shape_document(variant)["shapeHash"] != base_hash


# ---------------------------------------------------------------------------
# Node inventory
# ---------------------------------------------------------------------------


def test_inventory_lists_every_node_sorted_without_values():
    doc = {"a": 1, "b": [1, 2, {"c": 1}], "d": {}}
    nodes = shape_document(doc)["nodes"]
    assert [node["path"] for node in nodes] == [
        "",
        "/a",
        "/b",
        "/b/0",
        "/b/1",
        "/b/2",
        "/b/2/c",
        "/d",
    ]
    assert [node["type"] for node in nodes] == [
        "object",
        "number",
        "array",
        "number",
        "number",
        "object",
        "number",
        "object",
    ]
    for node in nodes:
        assert set(node) == {"path", "type"}


def test_inventory_type_names_for_all_six_json_types():
    nodes = {
        node["path"]: node["type"]
        for node in shape_document(
            {
                "s": "x",
                "n": 1,
                "f": 1.0,
                "b": True,
                "z": None,
                "o": {"k": 1},
                "a": [1],
                "eo": {},
                "ea": [],
            }
        )["nodes"]
    }
    assert nodes["/s"] == "string"
    assert nodes["/n"] == "number"
    assert nodes["/f"] == "number"
    assert nodes["/b"] == "boolean"
    assert nodes["/z"] == "null"
    assert nodes["/o"] == "object"
    assert nodes["/a"] == "array"
    assert nodes["/eo"] == "object"
    assert nodes["/ea"] == "array"


def test_root_scalar_and_empty_root_inventories():
    assert shape_document(42)["nodes"] == [{"path": "", "type": "number"}]
    assert shape_document("x")["nodes"] == [{"path": "", "type": "string"}]
    assert shape_document(True)["nodes"] == [{"path": "", "type": "boolean"}]
    assert shape_document(None)["nodes"] == [{"path": "", "type": "null"}]
    assert shape_document({})["nodes"] == [{"path": "", "type": "object"}]
    assert shape_document([])["nodes"] == [{"path": "", "type": "array"}]


def test_inventory_escapes_rfc6901_tokens():
    nodes = shape_document({"a/b": {"tilde~key": 1}})["nodes"]
    assert {"path": "/a~1b/tilde~0key", "type": "number"} in nodes
    assert {"path": "/a~1b", "type": "object"} in nodes


def test_inventory_is_path_sorted_even_when_insertion_is_not():
    doc = {"z": 1, "a": [1, 2], "m": {"y": 1, "b": 2}}
    paths = [node["path"] for node in shape_document(doc)["nodes"]]
    assert paths == sorted(paths)
    assert paths[0] == ""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_non_json_values_raise_shape_error():
    with pytest.raises(ShapeError):
        shape_document(object())
    with pytest.raises(ShapeError):
        shape_document({"a": object()})
    with pytest.raises(ShapeError):
        shape_document([object()])
