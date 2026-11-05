import io
import tarfile
import zipfile

from qwen_symbolic_thinking import captions


def _shard(path, names):
    with tarfile.open(path, "w") as tf:
        for name in names:
            data = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_webdataset_zip_repacks_and_resumes(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _shard(src / "00000.tar", ["00000.webp", "00000.json", "00001.webp"])
    _shard(src / "01000.tar", ["01000.webp"])
    import shutil
    import huggingface_hub

    def fake_download(repo, name, repo_type=None, local_dir=None, token=None):
        dst = tmp_path / "work" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / name, dst)
        return str(dst)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    out_zip = tmp_path / "faces.zip"
    got = captions.webdataset_zip("x/y", str(out_zip), shards=["00000.tar", "01000.tar"], work_dir=str(tmp_path / "work"))
    assert got == {"shards": 2, "written": 3, "skipped": 0}
    with zipfile.ZipFile(out_zip) as zf:
        assert sorted(zf.namelist()) == ["00000.webp", "00001.webp", "01000.webp"]
        assert zf.read("00001.webp") == b"00001.webp"                      # the bytes as they were
    assert not (tmp_path / "work" / "00000.tar").exists()                 # each shard deleted after use
    again = captions.webdataset_zip("x/y", str(out_zip), shards=["00000.tar"], work_dir=str(tmp_path / "work"))
    assert again == {"shards": 1, "written": 0, "skipped": 2}
    assert captions.photos(str(out_zip)) == ["00000.webp", "00001.webp", "01000.webp"]
