"""
Verse-level audio-text alignment for the LAIBUG Bible corpus.

IMPORTANT: Do NOT use Whisper (or any model under evaluation) to generate
timestamps/alignment for this corpus. Doing so would reintroduce the same
anchoring-bias problem already identified for the GRN ground truth (Section
4.7 / methodology doc) — the model being evaluated should never be the
source of its own reference labels.

Preferred alignment sources, in order:

  1. Bible Brain / Digital Bible Platform (DBP v4) API verse-timing data,
     if available for this fileset (~231+ bibleIds had timing data as of
     Jan 2022, growing since — check via the Available Content browser
     after API key approval).
  2. If unavailable: an independent forced-aligner (e.g. aeneas, Montreal
     Forced Aligner) run on chapter-level audio + chapter-level text. These
     tools are text-audio alignment specialists, not ASR models, so they
     don't share failure modes with Whisper/OWSM/MMS.
  3. As a last resort: manual verse-boundary marking via silence/VAD
     detection, spot-checked by a native speaker.

This script implements path 1 (DBP API) with a stub for path 2.
"""

import requests
from pathlib import Path

DBP_API_BASE = "https://4.dbt.io/api"  # confirm current base URL in DBP docs
# TODO: set via environment variable, do not commit the key
API_KEY = None


def get_available_filesets(api_key: str, language_code: str = "bug"):
    """Check whether this language has a Bugis fileset, and whether it
    includes verse-timing data. Confirm response schema against current
    Bible Brain API docs before relying on field names below."""
    resp = requests.get(
        f"{DBP_API_BASE}/bibles",
        params={"key": api_key, "language_code": language_code},
    )
    resp.raise_for_status()
    return resp.json()


def get_verse_timing(api_key: str, fileset_id: str, book: str, chapter: int):
    """Fetch verse start-times for a given chapter, if timing data exists
    for this fileset. Returns None if not available — fall back to
    independent forced alignment (see module docstring)."""
    resp = requests.get(
        f"{DBP_API_BASE}/timestamps/{fileset_id}/{book}/{chapter}",
        params={"key": api_key},
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def segment_audio_by_timing(audio_path: str, timing_data: list[dict], out_dir: str):
    """Cut chapter-level audio into verse-level clips using timing data.
    Requires timing_data as list of {verse, start_time, end_time}."""
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in timing_data:
        start_sample = int(entry["start_time"] * sr)
        end_sample = int(entry["end_time"] * sr)
        clip = audio[start_sample:end_sample]
        sf.write(out_dir / f"verse_{entry['verse']:03d}.wav", clip, sr)


def align_with_forced_aligner(audio_path: str, text_path: str, out_dir: str):
    """Fallback path if DBP has no timing data for this fileset. Requires
    an external forced-aligner tool (aeneas / Montreal Forced Aligner) to
    be installed separately — not a Python-only implementation.

    TODO: implement once it's confirmed the DBP fileset lacks timing data.
    """
    raise NotImplementedError(
        "Install and configure an external forced-aligner (aeneas or MFA) "
        "before using this path. Do not substitute a Whisper-based aligner."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--check-only", action="store_true", help="Just check fileset availability")
    args = parser.parse_args()

    filesets = get_available_filesets(args.api_key)
    print(filesets)
