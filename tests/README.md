# tests

`python3 tests/test_logo_finder.py` — plain script, no pytest, non-zero exit on failure.

## Two fixture sets

The harness prefers `tests/private/` (or wherever `PYCF_TEST_FIXTURES` points) and falls
back to the committed `tests/fixtures/`. It prints which it used.

`tests/private/` is **gitignored and must never be published.** It holds the maps at full
fidelity: no text scrub, no `.40` trim, one extra case, and each one's source path. It is
28MB and it is the only place the provenance survives.

`tests/fixtures/` is the publishable set — same cases, scrubbed as described below. Both
carry their own `expected.json`, because the scrub shifts a couple of recorded numbers.

If `tests/private/` is ever lost it can be rebuilt only by re-running the logo search over
the source recordings, and those get deleted over time — one is already gone.

## Why these are data and not video

`logo_finder._analyze` is a pure function of `(logo_sum, fcount, shape)`, so the logo search
can be tested without decoding anything. That is the point: the recordings these came from
get deleted over time, and two of them are the only examples we have of a real on-screen
weather alert.

## fixtures/

`<name>.npz` holds one persistence map — the per-pixel count of "was an edge in this
sampled frame" — plus `fcount` and `shape`, which is exactly what `_analyze` takes. Names
describe what the case exercises; nothing identifies a recording.

`expected.json` records what `_analyze` produced for each, and for eight of them what a
human saw when they watched the video. Those eight are the real assertions. The rest are
regression locks that only say "this did not change".

`logo-handflagged.counts.npz` is the one supervised case: per-frame match counts against
that recording's mask, plus its hand-flagged tags. It is what lets a test assert the logo
is on during true show and off during true commercial rather than merely self-consistent.

### What was removed from the committed maps

- **Everything below `.40 * fcount`.** Nothing under that can reach a logo mask (the `.55`
  gate less the `.15` band) or a stuck core (`.60`), so it cannot change a decision. It was
  verified to leave every output byte-identical, and it took the set from 29MB to 0.3MB.
- **On-screen text inside the alert bands.** A persistence map renders as a legible picture,
  and the two alert bands carried station identification and local place names. Those
  regions are horizontally max-pooled in 24px blocks, which keeps each column's vertical
  profile — so the band still reads as a band, with its edges and connectivity intact — while
  merging glyphs into unreadable blocks. Both still produce the same logo as before.
- **One fixture entirely.** A news channel whose map showed legible headlines, a location
  dateline and on-screen clocks; too much to scrub with confidence. It was the real example
  of a scatter of edges beside a removed banner being returned as a logo. The synthetic
  version of that case is still tested.

Station logos are deliberately left intact, since the logo is what is under test.

## What the checks cover

- the eight human verdicts: a logo is found, or correctly is not
- every fixture still produces its recorded output
- both confirmed alerts bound into one near-full-width region, and their cores clear the
  `_STUCK_CORE_PX` gate by 30x or more
- a logo whose own per-pixel persistence crosses the stuck cutoff is still returned — one
  fixture reaches .909 and is a real, useful logo; without the core gate it is deleted
- that recording scores 87.8% on during true show and 0.3% during true commercial
- synthetic overlays composited onto a real map: a scrolling band with disconnected pieces,
  a band flush with the frame edge, core-size boundaries at 1/49/100/400px, a graded skirt,
  an oversized region, and a scatter of edges beside a removed banner
