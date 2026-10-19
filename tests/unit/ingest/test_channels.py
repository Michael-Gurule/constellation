import pytest

from constellation.core.errors import ConfigError
from constellation.ingest.channels import (
    ChannelConfig,
    DictionaryEntry,
    load_channels,
    load_dictionary,
    parse_puilist,
    validate_against_dictionary,
    write_dictionary,
)

CONFIG = "configs/ingest/iss.yaml"

PUILIST = """<?xml version="1.0" encoding="utf-16"?>
<ISSLivePUIList created="Sep Tue 20 17:22 2011">
  <Discipline name="ADCO">
    <Symbol>
      <PUI>***nOt*aVaiLable***</PUI>
      <Public_PUI>USLAB000009</Public_PUI>
      <Description>Active  CMG Momentum (Nms)</Description>
      <MIN></MIN><MAX>N/A</MAX>
      <OPS_NOM>CMG_MOM</OPS_NOM><ENG_NOM>Momentum</ENG_NOM><UNITS>FT-LB-S</UNITS>
    </Symbol>
    <Symbol><Public_PUI></Public_PUI><Description>no id</Description></Symbol>
  </Discipline>
  <Discipline name="EVA">
    <Symbol>
      <Public_PUI>AIRLOCK000001</Public_PUI>
      <Description>Power, Voltage</Description><UNITS>CNT</UNITS>
    </Symbol>
  </Discipline>
</ISSLivePUIList>
"""


def _channel(**overrides):
    channel = {
        "id": "USLAB000009",
        "name": "cmg_momentum",
        "subsystem": "cmg",
        "kind": "numeric",
        "units": "N*m*s",
    }
    channel.update(overrides)
    return channel


def test_parse_puilist_reads_utf16_symbols():
    entries = parse_puilist(PUILIST.encode("utf-16"))

    assert [e.public_pui for e in entries] == ["USLAB000009", "AIRLOCK000001"]
    first = entries[0]
    assert first.discipline == "ADCO"
    assert first.description == "Active CMG Momentum (Nms)"  # whitespace collapsed
    assert first.units == "FT-LB-S"
    assert first.max == "N/A"
    assert entries[1].ops_nom == ""


@pytest.mark.parametrize(
    "xml",
    [
        b"<root><unclosed>",
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><x/>',
        '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE x><x/>'.encode("utf-16"),
    ],
)
def test_parse_puilist_rejects_broken_or_entity_laden_input(xml):
    with pytest.raises(ConfigError):
        parse_puilist(xml)


def test_dictionary_csv_round_trip_with_comment_header(tmp_path):
    entries = parse_puilist(PUILIST.encode("utf-16"))
    path = tmp_path / "dictionary.csv"

    write_dictionary(entries, path, "Source: example\nRetrieved: 2026-10-19")
    loaded = load_dictionary(path)

    assert path.read_text().startswith("# Source: example\n# Retrieved: 2026-10-19\n")
    assert list(loaded) == ["USLAB000009", "AIRLOCK000001"]
    assert loaded["USLAB000009"] == entries[0]


def test_packaged_dictionary_covers_the_isslive_symbols():
    dictionary = load_dictionary()

    assert len(dictionary) == 298
    assert dictionary["Z1000009"].description.endswith("Wheel Speed (rpm)")
    assert all(isinstance(e, DictionaryEntry) for e in dictionary.values())


def test_shipped_channel_config_is_valid_and_complete():
    config = load_channels(CONFIG)
    by_name = {c.name: c for c in config.channels}

    assert len(config.channels) == 91
    assert by_name["cmg1_wheel_speed"].id == "Z1000009"
    assert by_name["cmg1_spin_bearing_temp"].units == "deg C"
    assert by_name["gnc_mode"].kind == "state"
    assert {c.subsystem for c in config.channels} == {
        "cmg",
        "attitude",
        "power",
        "thermal",
        "comms",
    }
    # The time items are subscribed to separately and are not channels.
    assert not any(c.id.startswith("TIME_") for c in config.channels)


def test_unknown_ids_are_rejected():
    config = ChannelConfig(channels=[_channel(id="NOPE000001")])

    with pytest.raises(ConfigError, match="NOPE000001"):
        validate_against_dictionary(config, load_dictionary())


def test_duplicate_ids_and_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate channel ids"):
        ChannelConfig(channels=[_channel(), _channel(name="other")])
    with pytest.raises(ValueError, match="duplicate channel names"):
        ChannelConfig(channels=[_channel(), _channel(id="USLAB000010")])


@pytest.mark.parametrize("name", ["", "Has-Dash", "UPPER", "with space"])
def test_channel_names_must_be_lower_snake_case(name):
    with pytest.raises(ValueError, match="lower_snake_case"):
        ChannelConfig(channels=[_channel(name=name)])


def test_load_channels_reports_unknown_ids_from_a_file(tmp_path):
    path = tmp_path / "iss.yaml"
    path.write_text(
        "channels:\n  - {id: NOPE000001, name: ghost, subsystem: cmg, kind: numeric}\n"
    )

    with pytest.raises(ConfigError, match="NOPE000001"):
        load_channels(path)
