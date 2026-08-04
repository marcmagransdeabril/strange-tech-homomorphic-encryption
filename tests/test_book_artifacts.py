"""Validates that the built PDF/EPUB artifacts under book/ open and render correctly.

Runs as part of `pytest tests/`. Catches corrupt/truncated PDF or EPUB output
that shallow "file exists and is non-empty" checks would miss.
"""
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

pypdf = pytest.importorskip("pypdf")

ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = ROOT / "book"

PDF_FILES = sorted(BOOK_DIR.glob("*/*.pdf"))
EPUB_FILES = sorted(BOOK_DIR.glob("*/*.epub"))


def test_book_artifacts_present():
    assert PDF_FILES, "No PDF files found under book/"
    assert EPUB_FILES, "No EPUB files found under book/"
    assert len(PDF_FILES) == 2, f"Expected 2 PDFs (es/en), found {len(PDF_FILES)}"
    assert len(EPUB_FILES) == 2, f"Expected 2 EPUBs (es/en), found {len(EPUB_FILES)}"


@pytest.mark.parametrize("pdf_path", PDF_FILES, ids=lambda p: p.name)
def test_pdf_opens_and_has_pages(pdf_path):
    reader = pypdf.PdfReader(str(pdf_path))
    assert len(reader.pages) > 0, f"{pdf_path.name} has no pages"
    # Extracting text from the first/last page forces parsing of the content
    # streams, catching corruption that a shallow structural check would miss.
    reader.pages[0].extract_text()
    reader.pages[-1].extract_text()


@pytest.mark.parametrize("epub_path", EPUB_FILES, ids=lambda p: p.name)
def test_epub_opens_and_has_content(epub_path):
    assert zipfile.is_zipfile(epub_path), f"{epub_path.name} is not a valid zip archive"
    with zipfile.ZipFile(epub_path) as zf:
        bad_member = zf.testzip()
        assert bad_member is None, f"{epub_path.name} has corrupt member: {bad_member}"

        names = zf.namelist()
        assert "mimetype" in names, f"{epub_path.name} is missing the mimetype entry"
        mimetype = zf.read("mimetype").decode("ascii").strip()
        assert mimetype == "application/epub+zip", f"{epub_path.name} has wrong mimetype: {mimetype}"

        assert "META-INF/container.xml" in names, f"{epub_path.name} is missing META-INF/container.xml"
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        rootfile = container.find(".//c:rootfile", ns)
        assert rootfile is not None, f"{epub_path.name} container.xml has no rootfile"
        opf_path = rootfile.attrib["full-path"]
        assert opf_path in names, f"{epub_path.name} is missing its OPF at {opf_path}"

        opf = ET.fromstring(zf.read(opf_path))
        ns_opf = {"opf": "http://www.idpf.org/2007/opf"}
        spine = opf.find("opf:spine", ns_opf)
        items = spine.findall("opf:itemref", ns_opf) if spine is not None else []
        assert items, f"{epub_path.name} spine has no content items"
