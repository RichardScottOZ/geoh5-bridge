from pathlib import Path


def test_readme_omf_example_uses_reader_writer():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    assert "omf.load(" not in readme
    assert "omf.save(" not in readme
    assert "omf.OMFReader(" in readme
    assert "omf.OMFWriter(" in readme
