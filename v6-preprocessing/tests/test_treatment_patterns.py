"""
Pytest suite for treatment pattern rules.

One synthetic patient per rule, designed to match that rule exactly.
Each parametrized test asserts:
  - the intended patient matches the rule
  - no other patient in the fixture accidentally matches it

Patient construction notes
--------------------------
All dates are UTC-normalised (midnight) as in real IDEA4RC data.  The fixture
includes every slot column that any patient needs (up to 3 chemo lines), so
count_treatments() correctly returns 1, 2, or 3 per row.

Row → expected rule
 0  only_surgery
 1  only_radio
 2  only_chemo
 3  only_immuno
 4  only_target
 5  concomitant_systemic_radio
 6  surgery_postop_radio
 7  surgery_adj_chemo
 8  surgery_postop_radio_concomi_chemo
 9  radio_adj_chemo
10  concomi_chemo_radio_adj_chemo
11  chemo_immuno
12  chemo_target
13  immuno_target
14  neoadj_chemo_radio
15  neoadj_chemo_surgery
16  neoadj_chemo_concomi_chemo_radio
17  neoadj_chemo_concomi_chemo_radio_adj_chemo
18  neoadj_chemo_radio_adj_chemo
19  other  (no rule 1–19 fires)
"""

import pytest
import pandas as pd

from v6_preprocessing._treatment_patterns.rules import (
    GeneralParams,
    Rule6Params,
    Rule7Params,
    Rule8Params,
    Rule9Params,
    Rule10Params,
    Rule11Params,
    Rule15Params,
    Rule16Params,
    Rule17Params,
    Rule18Params,
    Rule19Params,
    rule_only_surgery,
    rule_only_radio,
    rule_only_chemo,
    rule_only_immuno,
    rule_only_target,
    rule_concomitant_systemic_radio,
    rule_surgery_postop_radio,
    rule_surgery_adj_chemo,
    rule_surgery_postop_radio_concomi_chemo,
    rule_radio_adj_chemo,
    rule_concomi_chemo_radio_adj_chemo,
    rule_chemo_immuno,
    rule_chemo_target,
    rule_immuno_target,
    rule_neoadj_chemo_radio,
    rule_neoadj_chemo_surgery,
    rule_neoadj_chemo_concomi_chemo_radio,
    rule_neoadj_chemo_concomi_chemo_radio_adj_chemo,
    rule_neoadj_chemo_radio_adj_chemo,
    rule_other,
)

TZ = "UTC"
D = "2020-01-01"  # shared diagnosis date


def ts(date_str):
    return pd.Timestamp(date_str, tz=TZ)


NAT = pd.NaT


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _base() -> dict:
    """Empty patient: diagnosis only, all treatment slots NaT."""
    return {
        "diagnosis_date": ts(D),
        "surgery_1_date": NAT,
        "radio_1_start_date": NAT, "radio_1_end_date": NAT,
        "chemo_1_start_date": NAT, "chemo_1_end_date": NAT,
        "chemo_2_start_date": NAT, "chemo_2_end_date": NAT,
        "chemo_3_start_date": NAT, "chemo_3_end_date": NAT,
        "immuno_1_start_date": NAT, "immuno_1_end_date": NAT,
        "targeted_1_start_date": NAT, "targeted_1_end_date": NAT,
    }


@pytest.fixture(scope="module")
def patients() -> pd.DataFrame:
    rows = []

    # 0 — only_surgery: surgery at day +45, no other treatment
    rows.append({**_base(),
        "surgery_1_date": ts("2020-02-15"),
    })

    # 1 — only_radio: radio at day +45–+74, no other treatment
    rows.append({**_base(),
        "radio_1_start_date": ts("2020-02-15"), "radio_1_end_date": ts("2020-03-15"),
    })

    # 2 — only_chemo: 1 chemo line at day +31, no other treatment
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
    })

    # 3 — only_immuno: 1 immuno line at day +31
    rows.append({**_base(),
        "immuno_1_start_date": ts("2020-02-01"), "immuno_1_end_date": ts("2020-04-01"),
    })

    # 4 — only_target: 1 targeted line at day +31
    rows.append({**_base(),
        "targeted_1_start_date": ts("2020-02-01"), "targeted_1_end_date": ts("2020-04-01"),
    })

    # 5 — concomitant_systemic_radio
    # Radio starts first (Feb 5), chemo starts 5 days later (Feb 10) — both overlap.
    # chemo_start > radio_start prevents rule_neoadj_chemo_radio from matching.
    rows.append({**_base(),
        "radio_1_start_date": ts("2020-02-05"), "radio_1_end_date": ts("2020-03-20"),
        "chemo_1_start_date": ts("2020-02-10"), "chemo_1_end_date": ts("2020-03-25"),
    })

    # 6 — surgery_postop_radio: surgery day+31, radio starts 59 days after surgery, no systemic
    rows.append({**_base(),
        "surgery_1_date": ts("2020-02-01"),
        "radio_1_start_date": ts("2020-04-01"), "radio_1_end_date": ts("2020-05-01"),
    })

    # 7 — surgery_adj_chemo: surgery day+31, chemo starts 89 days after surgery, no radio
    rows.append({**_base(),
        "surgery_1_date": ts("2020-02-01"),
        "chemo_1_start_date": ts("2020-05-01"), "chemo_1_end_date": ts("2020-07-01"),
    })

    # 8 — surgery_postop_radio_concomi_chemo: surgery day+31, radio+chemo together 59d later
    rows.append({**_base(),
        "surgery_1_date": ts("2020-02-01"),
        "radio_1_start_date": ts("2020-04-01"), "radio_1_end_date": ts("2020-05-01"),
        "chemo_1_start_date": ts("2020-04-05"), "chemo_1_end_date": ts("2020-04-27"),
    })

    # 9 — radio_adj_chemo: radio day+31–+91, chemo starts 61d after radio end, no surgery
    rows.append({**_base(),
        "radio_1_start_date": ts("2020-02-01"), "radio_1_end_date": ts("2020-04-01"),
        "chemo_1_start_date": ts("2020-06-01"), "chemo_1_end_date": ts("2020-08-01"),
    })

    # 10 — concomi_chemo_radio_adj_chemo
    # chemo1 starts on the same day as radio (not strictly before → rule_neoadj variants don't fire)
    # chemo2 starts 28d after max(chemo1_end, radio_end)=Apr3
    rows.append({**_base(),
        "radio_1_start_date": ts("2020-02-05"), "radio_1_end_date": ts("2020-04-03"),
        "chemo_1_start_date": ts("2020-02-05"), "chemo_1_end_date": ts("2020-04-01"),
        "chemo_2_start_date": ts("2020-05-01"), "chemo_2_end_date": ts("2020-07-01"),
    })

    # 11 — chemo_immuno: both start within 90d of diagnosis, no surgery/radio/targeted
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "immuno_1_start_date": ts("2020-02-15"), "immuno_1_end_date": ts("2020-05-01"),
    })

    # 12 — chemo_target
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "targeted_1_start_date": ts("2020-02-15"), "targeted_1_end_date": ts("2020-05-01"),
    })

    # 13 — immuno_target: no chemo
    rows.append({**_base(),
        "immuno_1_start_date": ts("2020-02-01"), "immuno_1_end_date": ts("2020-04-01"),
        "targeted_1_start_date": ts("2020-02-15"), "targeted_1_end_date": ts("2020-05-01"),
    })

    # 14 — neoadj_chemo_radio: chemo ends, then radio starts 30d later (< 90d threshold)
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "radio_1_start_date": ts("2020-05-01"), "radio_1_end_date": ts("2020-06-01"),
    })

    # 15 — neoadj_chemo_surgery: chemo ends, surgery 30d later
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "surgery_1_date": ts("2020-05-01"),
    })

    # 16 — neoadj_chemo_concomi_chemo_radio
    # chemo1 (neoadj), then chemo2+radio together (chemo2 starts 4d after radio_start)
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "radio_1_start_date": ts("2020-05-01"), "radio_1_end_date": ts("2020-07-01"),
        "chemo_2_start_date": ts("2020-05-05"), "chemo_2_end_date": ts("2020-06-27"),
    })

    # 17 — neoadj_chemo_concomi_chemo_radio_adj_chemo
    # Same as row 16, plus chemo3 starting 31d after max(chemo2_end=Jun27, radio_end=Jul1)=Jul1
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "radio_1_start_date": ts("2020-05-01"), "radio_1_end_date": ts("2020-07-01"),
        "chemo_2_start_date": ts("2020-05-05"), "chemo_2_end_date": ts("2020-06-27"),
        "chemo_3_start_date": ts("2020-08-01"), "chemo_3_end_date": ts("2020-10-01"),
    })

    # 18 — neoadj_chemo_radio_adj_chemo
    # chemo1 ends Apr1, radio starts May1 (30d < 90), radio ends Jun15, chemo2 starts Jul15 (30d < 90)
    # chemo2 starts AFTER radio_end → adjuvant (no overlap with radio)
    rows.append({**_base(),
        "chemo_1_start_date": ts("2020-02-01"), "chemo_1_end_date": ts("2020-04-01"),
        "radio_1_start_date": ts("2020-05-01"), "radio_1_end_date": ts("2020-06-15"),
        "chemo_2_start_date": ts("2020-07-15"), "chemo_2_end_date": ts("2020-09-15"),
    })

    # 19 — other: surgery + chemo both outside the 90-day diagnosis window
    rows.append({**_base(),
        "surgery_1_date": ts("2020-08-01"),          # +212d > 90
        "chemo_1_start_date": ts("2020-09-01"),       # +244d > 90
        "chemo_1_end_date": ts("2020-11-01"),
    })

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Compute all rule results once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rule_results(patients) -> dict[str, pd.Series]:
    gp = GeneralParams()
    results_list = [
        rule_only_surgery(patients, gp),
        rule_only_radio(patients, gp),
        rule_only_chemo(patients, gp),
        rule_only_immuno(patients, gp),
        rule_only_target(patients, gp),
        rule_concomitant_systemic_radio(patients, Rule6Params()),
        rule_surgery_postop_radio(patients, Rule7Params()),
        rule_surgery_adj_chemo(patients, Rule8Params()),
        rule_surgery_postop_radio_concomi_chemo(patients, Rule9Params()),
        rule_radio_adj_chemo(patients, Rule10Params()),
        rule_concomi_chemo_radio_adj_chemo(patients, Rule11Params()),
        rule_chemo_immuno(patients, gp),
        rule_chemo_target(patients, gp),
        rule_immuno_target(patients, gp),
        rule_neoadj_chemo_radio(patients, Rule15Params()),
        rule_neoadj_chemo_surgery(patients, Rule16Params()),
        rule_neoadj_chemo_concomi_chemo_radio(patients, Rule17Params()),
        rule_neoadj_chemo_concomi_chemo_radio_adj_chemo(patients, Rule18Params()),
        rule_neoadj_chemo_radio_adj_chemo(patients, Rule19Params()),
    ]
    results_list.append(rule_other(results_list))

    names = [
        "only_surgery",
        "only_radio",
        "only_chemo",
        "only_immuno",
        "only_target",
        "concomitant_systemic_radio",
        "surgery_postop_radio",
        "surgery_adj_chemo",
        "surgery_postop_radio_concomi_chemo",
        "radio_adj_chemo",
        "concomi_chemo_radio_adj_chemo",
        "chemo_immuno",
        "chemo_target",
        "immuno_target",
        "neoadj_chemo_radio",
        "neoadj_chemo_surgery",
        "neoadj_chemo_concomi_chemo_radio",
        "neoadj_chemo_concomi_chemo_radio_adj_chemo",
        "neoadj_chemo_radio_adj_chemo",
        "other",
    ]
    return dict(zip(names, results_list))


# ---------------------------------------------------------------------------
# Parametrized test: one patient per rule
# ---------------------------------------------------------------------------

RULE_TO_ROW = [
    ("only_surgery",                              0),
    ("only_radio",                                1),
    ("only_chemo",                                2),
    ("only_immuno",                               3),
    ("only_target",                               4),
    ("concomitant_systemic_radio",                5),
    ("surgery_postop_radio",                      6),
    ("surgery_adj_chemo",                         7),
    ("surgery_postop_radio_concomi_chemo",         8),
    ("radio_adj_chemo",                           9),
    ("concomi_chemo_radio_adj_chemo",            10),
    ("chemo_immuno",                             11),
    ("chemo_target",                             12),
    ("immuno_target",                            13),
    ("neoadj_chemo_radio",                       14),
    ("neoadj_chemo_surgery",                     15),
    ("neoadj_chemo_concomi_chemo_radio",         16),
    ("neoadj_chemo_concomi_chemo_radio_adj_chemo", 17),
    ("neoadj_chemo_radio_adj_chemo",             18),
    ("other",                                    19),
]


@pytest.mark.parametrize("rule_name,expected_row", RULE_TO_ROW)
def test_rule_matches_intended_patient(rule_results, rule_name, expected_row):
    series = rule_results[rule_name]
    assert bool(series.iloc[expected_row]), (
        f"Rule '{rule_name}' did not match row {expected_row} (the patient designed for it)"
    )


@pytest.mark.parametrize("rule_name,expected_row", RULE_TO_ROW)
def test_rule_has_no_false_positives(rule_results, rule_name, expected_row):
    series = rule_results[rule_name]
    false_positives = [i for i, v in enumerate(series) if bool(v) and i != expected_row]
    assert false_positives == [], (
        f"Rule '{rule_name}' unexpectedly matched rows: {false_positives}"
    )
