"""Profile any CSV in 3 lines.

Usage:
    python examples/profile_csv.py path/to/your/data.csv
    python examples/profile_csv.py data.tsv --delimiter tab
"""

import sys
import database_whisper as dw

path = sys.argv[1] if len(sys.argv) > 1 else "spotify_tracks.csv"
delimiter = "\t" if "--delimiter" in sys.argv and "tab" in sys.argv else None

print(dw.profile(path, delimiter=delimiter))
