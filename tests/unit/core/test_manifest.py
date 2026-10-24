import pytest

from constellation.core.errors import ManifestError
from constellation.core.manifest import build_manifest, manifest_hash, verify_manifest


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01\x02")
    return tmp_path


def test_manifest_lists_relative_paths_with_hash_and_size(tree):
    manifest = build_manifest(tree)

    assert sorted(manifest) == ["a.txt", "sub/b.bin"]
    assert manifest["a.txt"]["size"] == 5
    assert len(manifest["a.txt"]["sha256"]) == 64
    assert manifest["a.txt"]["sha256"] != manifest["sub/b.bin"]["sha256"]


def test_hidden_files_and_the_manifest_itself_are_skipped(tree):
    (tree / ".part-1.parquet.tmp").write_text("temp")
    (tree / "manifest.json").write_text("{}")

    assert sorted(build_manifest(tree)) == ["a.txt", "sub/b.bin"]


def test_verify_accepts_an_unchanged_tree(tree):
    verify_manifest(tree, build_manifest(tree))


def test_verify_reports_changed_missing_and_unexpected_files(tree):
    manifest = build_manifest(tree)
    (tree / "a.txt").write_text("ALPHA")
    (tree / "sub" / "b.bin").unlink()
    (tree / "new.txt").write_text("surprise")

    with pytest.raises(ManifestError) as info:
        verify_manifest(tree, manifest)

    message = str(info.value)
    assert "changed=['a.txt']" in message
    assert "missing=['sub/b.bin']" in message
    assert "unexpected=['new.txt']" in message


def test_manifest_hash_is_stable_and_content_sensitive(tree):
    first = manifest_hash(build_manifest(tree))

    assert first == manifest_hash(build_manifest(tree))
    (tree / "a.txt").write_text("changed")
    assert first != manifest_hash(build_manifest(tree))
