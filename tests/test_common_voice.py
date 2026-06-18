"""Speaker selection from a Common Voice TSV fixture."""

from __future__ import annotations

import pytest

from duplex_bol.data import count_by_speaker, read_cv_tsv, select_speakers

# spk1: male x5, spk2: female x4, spk3: male x3, spk4: female x1
_TSV = """client_id\tpath\tsentence\tgender\tage
spk1\ta1.mp3\tجملہ ایک\tmale_masculine\ttwenties
spk1\ta2.mp3\tجملہ دو\tmale_masculine\ttwenties
spk1\ta3.mp3\tجملہ تین\tmale_masculine\ttwenties
spk1\ta4.mp3\tجملہ چار\tmale_masculine\ttwenties
spk1\ta5.mp3\tجملہ پانچ\tmale_masculine\ttwenties
spk2\tb1.mp3\tجملہ\tfemale_feminine\tthirties
spk2\tb2.mp3\tجملہ\tfemale_feminine\tthirties
spk2\tb3.mp3\tجملہ\tfemale_feminine\tthirties
spk2\tb4.mp3\tجملہ\tfemale_feminine\tthirties
spk3\tc1.mp3\tجملہ\tmale_masculine\tforties
spk3\tc2.mp3\tجملہ\tmale_masculine\tforties
spk3\tc3.mp3\tجملہ\tmale_masculine\tforties
spk4\td1.mp3\tجملہ\tfemale_feminine\tteens
"""


@pytest.fixture
def cv_tsv(tmp_path):
    path = tmp_path / "validated.tsv"
    path.write_text(_TSV, encoding="utf-8")
    return path


def test_read_parses_columns_and_gender_buckets(cv_tsv):
    clips = read_cv_tsv(cv_tsv)
    assert len(clips) == 13
    assert clips[0].client_id == "spk1"
    assert clips[0].gender_bucket == "male"
    assert clips[5].gender_bucket == "female"


def test_count_by_speaker(cv_tsv):
    counts = count_by_speaker(read_cv_tsv(cv_tsv))
    assert counts == {"spk1": 5, "spk2": 4, "spk3": 3, "spk4": 1}


def test_select_top_n_by_count_without_balancing(cv_tsv):
    chosen = select_speakers(read_cv_tsv(cv_tsv), n=3, balance_gender=False)
    assert set(chosen) == {"spk1", "spk2", "spk3"}  # the three biggest
    assert len(chosen["spk1"]) == 5


def test_balanced_selection_mixes_genders(cv_tsv):
    chosen = select_speakers(read_cv_tsv(cv_tsv), n=2, balance_gender=True)
    buckets = {clips[0].gender_bucket for clips in chosen.values()}
    assert buckets == {"male", "female"}  # one of each, not two of the same


def test_min_clips_filters_thin_speakers(cv_tsv):
    chosen = select_speakers(read_cv_tsv(cv_tsv), n=10, min_clips=2)
    assert "spk4" not in chosen  # only 1 clip


def test_selection_is_deterministic(cv_tsv):
    clips = read_cv_tsv(cv_tsv)
    a = select_speakers(clips, n=3)
    b = select_speakers(clips, n=3)
    assert list(a) == list(b)
