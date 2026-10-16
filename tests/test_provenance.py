import json

from src.utils.provenance import config_hash, model_fingerprint, provenance, write_run_record


def test_config_hash_is_stable():
    assert config_hash() == config_hash()
    assert len(config_hash()) == 12


def test_model_fingerprint_hashes_the_checkpoint(dummy_model):
    fp = model_fingerprint(dummy_model)
    assert len(fp["sha256"]) == 64
    assert fp["size_bytes"] > 0
    assert fp["name"] == dummy_model.name


def test_fingerprint_of_a_missing_file_is_empty(tmp_path):
    fp = model_fingerprint(tmp_path / "nope.pth")
    assert fp["sha256"] is None


def test_provenance_lists_the_versions_that_matter(dummy_model):
    p = provenance(dummy_model, "cpu")
    assert p["versions"]["torch"] and p["versions"]["monai"]
    assert p["device"] == "cpu"
    assert p["config_hash"] == config_hash()


def test_run_record_round_trips(dummy_model, tmp_path):
    path = write_run_record({"spleen_volume_ml": 451.0}, dummy_model, out_dir=tmp_path)
    record = json.loads(path.read_text())
    assert record["result"]["spleen_volume_ml"] == 451.0
    assert record["provenance"]["model"]["sha256"]
    assert record["timestamp_utc"].endswith("Z")
