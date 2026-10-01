"""Remove leaked pair metadata from entity-table sidecars in the isolated campaign."""
from pathlib import Path

count = 0
for root in (Path('data/processed/magellan'), Path('data/processed_en/magellan'), Path('data/processed_cross_language/magellan')):
    for path in sorted(root.rglob('*.metadata')):
        if '_left_' in path.name or '_right_' in path.name:
            if path.read_text() != '#key=mag_id\n':
                path.write_text('#key=mag_id\n')
                count += 1
print('Corrected entity-table metadata sidecars:', count)
