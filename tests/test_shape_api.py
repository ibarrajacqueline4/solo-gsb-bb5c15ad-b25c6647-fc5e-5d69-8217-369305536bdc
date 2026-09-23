"""End-to-end tests for POST /shape."""

import hashlib
import struct

from fastapi.testclient import TestClient

from app.main import app
from app.shape import shape_document

client = TestClient(app)


def _shape(document):
    return client.post("/shape", json={"document": document})


def _h(preimage: bytes) -> str:
    return hashlib.sha256(preimage).hexdigest()


DOC = {
    "subject": {"name": "Ada", "age": 36, "active": True},
    "emails": ["ada@example.com", "ada@analytical.engine"],
    "emptyObj": {},
    "emptyArr": [],
    "nickname": None,
}


def test_shape_returns_value_free_inventory_and_real_hash():
    resp = _shape(DOC)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data.keys()) == {"shapeHash", "nodes"}
    assert data == shape_document(DOC)
    assert data["shapeHash"] == shape_document(DOC)["shapeHash"]

    paths = [node["path"] for node in data["nodes"]]
    assert paths == sorted(paths)
    assert data["nodes"][0] == {"path": "", "type": "object"}
    assert {
        "path": "/subject/active",
        "type": "boolean",
    } in data["nodes"]
    assert {"path": "/emptyObj", "type": "object"} in data["nodes"]
    assert {"path": "/emptyArr", "type": "array"} in data["nodes"]
    assert {"path": "/nickname", "type": "null"} in data["nodes"]
    # No scalar value ever appears in a node entry.
    for node in data["nodes"]:
        assert set(node.keys()) == {"path", "type"}


def test_shape_hash_is_64_lowercase_hex():
    data = _shape(DOC).json()
    assert len(data["shapeHash"]) == 64
    assert all(ch in "0123456789abcdef" for ch in data["shapeHash"])


def test_same_shaped_documents_have_equal_hashes():
    twin = {
        "subject": {"name": "Grace", "age": 1.5, "active": False},
        "emails": ["other", "values"],
        "emptyObj": {},
        "emptyArr": [],
        "nickname": None,
    }
    assert _shape(DOC).json()["shapeHash"] == _shape(twin).json()["shapeHash"]
    # The inventories are identical too.
    assert _shape(DOC).json()["nodes"] == _shape(twin).json()["nodes"]


def test_shape_sensitive_to_structure_and_node_types():
    base_hash = _shape(DOC).json()["shapeHash"]
    changes = [
        # object member removed
        {**{k: v for k, v in DOC.items() if k != "nickname"}},
        # number -> string
        {**DOC, "subject": {**DOC["subject"], "age": "36"}},
        # shorter array
        {**DOC, "emails": ["one"]},
        # object -> array at /subject
        {**DOC, "subject": ["Ada", 36, True]},
        # empty object <-> empty array
        {**DOC, "emptyObj": [], "emptyArr": {}},
        # boolean -> null
        {**DOC, "subject": {**DOC["subject"], "active": None}},
    ]
    for changed in changes:
        assert _shape(changed).json()["shapeHash"] != base_hash


def test_swapping_same_typed_values_keeps_shape():
    swapped = {**DOC, "emails": ["second", "first"]}
    assert _shape(swapped).json()["shapeHash"] == _shape(DOC).json()["shapeHash"]
    swapped_keys = {
        "nickname": None,
        "emptyArr": [],
        "emptyObj": {},
        "emails": ["a", "b"],
        "subject": {"active": False, "age": 7, "name": "X"},
    }
    assert _shape(swapped_keys).json()["shapeHash"] == _shape(DOC).json()[
        "shapeHash"
    ]


def test_shape_object_insertion_order_does_not_matter():
    one = {"z": 1, "a": {"y": 2, "b": [1, 2]}}
    two = {"a": {"b": [9, 8], "y": 7}, "z": 3}
    assert _shape(one).json()["shapeHash"] == _shape(two).json()["shapeHash"]


def test_shape_manual_preimages_over_http():
    assert _shape(None).json()["shapeHash"] == _h(b"\x10")
    assert _shape(True).json()["shapeHash"] == _h(b"\x11")
    assert _shape(1).json()["shapeHash"] == _h(b"\x12")
    assert _shape("x").json()["shapeHash"] == _h(b"\x13")
    assert _shape({}).json()["shapeHash"] == _h(b"\x14")
    assert _shape([]).json()["shapeHash"] == _h(b"\x15")

    child = hashlib.sha256(b"\x12").digest()
    obj_pre = b"\x20" + struct.pack(">I", 1) + struct.pack(">I", 1) + b"a" + child
    assert _shape({"a": 1}).json()["shapeHash"] == _h(obj_pre)

    arr_pre = (
        b"\x21"
        + struct.pack(">I", 2)
        + hashlib.sha256(b"\x12").digest()
        + hashlib.sha256(b"\x13").digest()
    )
    assert _shape([1, "x"]).json()["shapeHash"] == _h(arr_pre)


def test_shape_root_scalar_documents():
    assert _shape(42).json() == {
        "shapeHash": _h(b"\x12"),
        "nodes": [{"path": "", "type": "number"}],
    }
    assert _shape(None).json()["nodes"] == [{"path": "", "type": "null"}]


def test_shape_escaped_pointer_tokens():
    data = _shape({"a/b": {"tilde~key": 7}}).json()
    assert {"path": "/a~1b/tilde~0key", "type": "number"} in data["nodes"]


def test_shape_request_errors():
    # body must be an object
    resp = client.post("/shape", json=[1, 2, 3])
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "body"

    # missing document
    resp = client.post("/shape", json={})
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "document"

    # malformed JSON
    resp = client.post("/shape", content=b"{not json", headers={})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_JSON"

    # NaN parses as JSON-with-extensions but fails the JSON value validator.
    resp = client.post(
        "/shape",
        content=b'{"document": NaN}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNSUPPORTED_VALUE"
    assert resp.json()["error"]["loc"] == "document"


def test_shape_listed_in_service_metadata_and_openapi():
    info = client.get("/").json()
    assert info["endpoints"]["shape"] == "POST /shape"
    assert info["version"] == app.version

    schema = client.get("/openapi.json").json()
    assert "/shape" in schema["paths"]
    assert "post" in schema["paths"]["/shape"]
    assert {"ShapeRequest", "ShapeNode", "ShapeResponse"} <= set(
        schema["components"]["schemas"]
    )
