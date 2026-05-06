"""One-time migration: convert v3 sanity stories from <P1>/</P1> XML format
to ## Prior markdown format. Idempotent (skips files already in markdown).

Usage: python scripts/migrate_v3_xml_to_md.py runs/cognitive_v3_sanity
"""
import re
import sys
from pathlib import Path


def parse_xml(text: str) -> dict[str, str] | None:
    blocks: dict[str, str] = {}
    for tag in ("P1", "P2", "P3"):
        m = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE
        )
        if m:
            blocks[tag] = m.group(1).strip()
    return blocks if len(blocks) == 3 else None


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    stories_dir = Path(sys.argv[1]) / "stories"
    converted = 0
    for path in sorted(stories_dir.glob("*.txt")):
        if path.parent != stories_dir:
            continue
        text = path.read_text()
        if "## Prior" in text:
            continue
        head, _, body = text.partition("\n---\n")
        blocks = parse_xml(body)
        if blocks is None:
            print(f"  SKIP {path.name}: cannot parse")
            continue
        new_body = (
            f"## Prior\n{blocks['P1']}\n\n"
            f"## Discovery\n{blocks['P2']}\n\n"
            f"## Reaction\n{blocks['P3']}\n"
        )
        path.write_text(head + "\n---\n" + new_body)
        converted += 1
    print(f"Converted {converted} files.")


if __name__ == "__main__":
    main()
