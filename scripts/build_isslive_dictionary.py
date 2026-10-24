"""Build the packaged ISSLive parameter dictionary from NASA's PUIList.xml.

python scripts/build_isslive_dictionary.py
python scripts/build_isslive_dictionary.py --source-file PUIList.xml
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import httpx

from constellation.ingest.channels import parse_puilist, write_dictionary

SOURCE_URL = (
    "https://raw.githubusercontent.com/Lightstreamer/"
    "Lightstreamer-example-ISSLive-client-javascript/master/src/assets/PUIList.xml"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "src/constellation/ingest/data/isslive_dictionary.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-file", type=Path, help="read a local copy instead")
    args = parser.parse_args()

    if args.source_file is not None:
        xml = args.source_file.read_bytes()
        origin = f"local copy of {SOURCE_URL}"
    else:
        response = httpx.get(SOURCE_URL, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        xml = response.content
        origin = SOURCE_URL

    entries = parse_puilist(xml)
    header = (
        "NASA ISSLive parameter dictionary (PUIList.xml, created 2011).\n"
        f"Source: {origin}\n"
        f"Retrieved: {datetime.now(UTC):%Y-%m-%d}"
    )
    write_dictionary(entries, args.out, header)
    print(f"wrote {len(entries)} symbols to {args.out}")


if __name__ == "__main__":
    main()
