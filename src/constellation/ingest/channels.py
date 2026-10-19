"""ISS channel configuration and NASA's ISSLive parameter dictionary."""

import csv
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from constellation.core.config import load_config
from constellation.core.errors import ConfigError

DICTIONARY_RESOURCE = ("constellation.ingest", "data/isslive_dictionary.csv")
DICTIONARY_COLUMNS = (
    "public_pui",
    "discipline",
    "description",
    "units",
    "ops_nom",
    "eng_nom",
    "min",
    "max",
)
_XML_FIELDS = {
    "description": "Description",
    "units": "UNITS",
    "ops_nom": "OPS_NOM",
    "eng_nom": "ENG_NOM",
    "min": "MIN",
    "max": "MAX",
}


@dataclass(frozen=True)
class DictionaryEntry:
    """One symbol of the ISSLive parameter dictionary."""

    public_pui: str
    discipline: str
    description: str
    units: str
    ops_nom: str
    eng_nom: str
    min: str
    max: str


def parse_puilist(xml: bytes) -> list[DictionaryEntry]:
    """Parse NASA's `PUIList.xml` (UTF-16) into dictionary entries."""
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml or b"<\x00!\x00D\x00O\x00C" in xml:
        raise ConfigError("PUIList.xml must not declare a DOCTYPE or entities")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ConfigError(f"cannot parse PUIList.xml: {error}") from error

    entries: list[DictionaryEntry] = []
    for discipline in root.iter("Discipline"):
        for symbol in discipline.iter("Symbol"):
            pui = (symbol.findtext("Public_PUI") or "").strip()
            if not pui:
                continue
            fields = {
                key: " ".join((symbol.findtext(tag) or "").split())
                for key, tag in _XML_FIELDS.items()
            }
            entries.append(
                DictionaryEntry(
                    public_pui=pui,
                    discipline=discipline.get("name", ""),
                    **fields,
                )
            )
    return entries


def write_dictionary(
    entries: Iterable[DictionaryEntry], path: Path, header: str
) -> None:
    """Write entries as CSV, with `header` as leading `#` comment lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        for line in header.strip().splitlines():
            handle.write(f"# {line}\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(DICTIONARY_COLUMNS)
        for entry in entries:
            writer.writerow([getattr(entry, column) for column in DICTIONARY_COLUMNS])


def load_dictionary(path: str | Path | None = None) -> dict[str, DictionaryEntry]:
    """Read the dictionary CSV (the packaged copy by default), keyed by public PUI."""
    if path is None:
        package, name = DICTIONARY_RESOURCE
        text = resources.files(package).joinpath(name).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    rows = csv.DictReader(lines)
    return {row["public_pui"]: DictionaryEntry(**row) for row in rows}


class Channel(BaseModel):
    """One configured ISS channel."""

    id: str
    name: str
    subsystem: str
    kind: Literal["numeric", "state"]
    units: str = ""
    expect_static: bool = False

    @field_validator("name")
    @classmethod
    def _name_is_a_safe_identifier(cls, value: str) -> str:
        if not value or not value.replace("_", "").isalnum() or value != value.lower():
            raise ValueError(f"channel name {value!r} must be lower_snake_case")
        return value


class ChannelConfig(BaseModel):
    """The `configs/ingest/iss.yaml` file."""

    channels: list[Channel]

    @field_validator("channels")
    @classmethod
    def _ids_and_names_are_unique(cls, channels: list[Channel]) -> list[Channel]:
        for attribute in ("id", "name"):
            values = [getattr(c, attribute) for c in channels]
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"duplicate channel {attribute}s: {duplicates}")
        return channels


def validate_against_dictionary(
    config: ChannelConfig, dictionary: dict[str, DictionaryEntry]
) -> None:
    """Raise `ConfigError` if a configured ID is not in the dictionary."""
    unknown = sorted(c.id for c in config.channels if c.id not in dictionary)
    if unknown:
        raise ConfigError(f"channel IDs not in the ISSLive dictionary: {unknown}")


def load_channels(
    path: str | Path,
    dictionary: dict[str, DictionaryEntry] | None = None,
) -> ChannelConfig:
    """Load the channel config and check every ID against the dictionary."""
    config = load_config(path, ChannelConfig)
    validate_against_dictionary(config, dictionary or load_dictionary())
    return config
