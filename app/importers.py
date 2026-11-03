from dataclasses import dataclass
from html.parser import HTMLParser
import io
import re
import zipfile
from pathlib import PurePosixPath
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from pypdf import PdfReader


@dataclass
class Section:
    title: str
    text: str


def clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n"))).strip()


def split_text(text: str, markdown: bool) -> list[Section]:
    text = clean(text)
    if markdown:
        matches = list(re.finditer(r"(?m)^#{1,2}\s+(.+?)\s*$", text))
        if matches:
            sections = []
            for i, match in enumerate(matches):
                body = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
                sections.append(Section(match.group(1).strip(), clean(body.lstrip("\n"))))
            return [s for s in sections if s.text]
    matches = list(re.finditer(r"(?im)^\s*(chapter|part|section)\s+([\wIVXLC.-]+).*?$", text))
    if matches:
        return [Section(m.group(0).strip(), clean(text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)])) for i, m in enumerate(matches) if clean(text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)])]
    return [Section("Section 1", text)]


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}: self.skip += 1
        if tag in {"p", "div", "br", "h1", "h2", "h3", "li"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style"}: self.skip = max(0, self.skip - 1)
        if tag in {"p", "div", "h1", "h2", "h3", "li"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)


def _safe_member(name: str) -> str:
    path = PurePosixPath(unquote(name.split("#", 1)[0]))
    if path.is_absolute() or ".." in path.parts: raise ValueError("unsafe EPUB archive path")
    return str(path)


def _local(element: ET.Element, name: str) -> list[ET.Element]:
    return [x for x in element.iter() if x.tag.rsplit("}", 1)[-1] == name]


def epub_sections(data: bytes) -> list[Section]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if len(names) > 2000: raise ValueError("EPUB has too many archive members")
        if any(info.flag_bits & 0x1 for info in z.infolist()): raise ValueError("encrypted EPUBs are not supported")
        if any(info.file_size > 10 * 1024 * 1024 for info in z.infolist()): raise ValueError("EPUB member exceeds size limit")
        if sum(info.file_size for info in z.infolist()) > 50 * 1024 * 1024: raise ValueError("EPUB exceeds uncompressed size limit")
        safe_names = {_safe_member(n) for n in names}
        containers = [n for n in safe_names if n.endswith("META-INF/container.xml")]
        if not containers: raise ValueError("EPUB container is missing")
        try:
            container = ET.fromstring(z.read(containers[0]))
            package = next(x.attrib["full-path"] for x in _local(container, "rootfile"))
            package = _safe_member(package)
            opf_root = ET.fromstring(z.read(package))
        except (KeyError, ET.ParseError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("invalid EPUB package metadata") from exc
        base = PurePosixPath(package).parent
        manifest = {x.attrib["id"]: x.attrib["href"] for x in _local(opf_root, "item") if "id" in x.attrib and "href" in x.attrib}
        spine = [x.attrib["idref"] for x in _local(opf_root, "itemref") if "idref" in x.attrib]
        sections = []
        for item_id in spine:
            href = manifest.get(item_id)
            if not href: continue
            path = _safe_member(str(base / PurePosixPath(unquote(href))))
            if path not in safe_names: raise ValueError("EPUB spine references a missing member")
            parser = TextParser(); parser.feed(z.read(path).decode("utf-8", "replace"))
            found = split_text("".join(parser.parts), False)
            sections.extend(found)
        return sections or [Section("Section 1", "")]


def extract(filename: str, data: bytes, max_pdf_pages: int) -> tuple[str, list[Section]]:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"txt", "md", "pdf", "epub"}: raise ValueError("unsupported format; use .txt, .md, .pdf, or .epub")
    if not data: raise ValueError("uploaded file is empty")
    try:
        if suffix == "pdf":
            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > max_pdf_pages: raise ValueError("PDF exceeds page limit")
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            return filename.rsplit(".", 1)[0], split_text(text, False)
        if suffix == "epub": return filename.rsplit(".", 1)[0], epub_sections(data)
        text = data.decode("utf-8-sig")
        return filename.rsplit(".", 1)[0], split_text(text, suffix == "md")
    except UnicodeDecodeError as exc: raise ValueError("text file must be UTF-8") from exc
    except zipfile.BadZipFile as exc: raise ValueError("invalid EPUB archive") from exc
