from dataclasses import dataclass, field

import pandas as pd

from .helpers import (
    has_treatment,
    has_any_systemic,
    count_treatments,
    count_systemic,
    first_start,
    last_end,
    first_systemic_start,
    last_systemic_end,
    nth_start,
    nth_end,
    within_days_after,
    within_symmetric_gap,
    dates_overlap,
)


# ---------------------------------------------------------------------------
# Per-rule parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GeneralParams:
    general_rule_days: int = 90


@dataclass
class Rule6Params:
    general_rule_days: int = 90
    concomitant_start_gap: int = 14
    concomitant_end_gap: int = 14


@dataclass
class Rule7Params:
    general_rule_days: int = 90
    surgery_postop_radio_days: int = 120


@dataclass
class Rule8Params:
    general_rule_days: int = 90
    surgery_adjuvant_chemo_days: int = 120


@dataclass
class Rule9Params:
    general_rule_days: int = 90
    surgery_postop_radio_days: int = 120
    postop_radio_concomi_start_gap: int = 14
    postop_radio_concomi_end_gap: int = 14


@dataclass
class Rule10Params:
    general_rule_days: int = 90
    radio_adjuvant_chemo_days: int = 120


@dataclass
class Rule11Params:
    general_rule_days: int = 90
    concomi_radio_adj_start_gap: int = 14
    concomi_radio_adj_end_gap: int = 14
    concomi_radio_adj_to_next: int = 90


@dataclass
class Rule15Params:
    general_rule_days: int = 90
    neoadj_chemo_to_radio: int = 90


@dataclass
class Rule16Params:
    general_rule_days: int = 90
    neoadj_chemo_to_surgery: int = 90


@dataclass
class Rule17Params:
    general_rule_days: int = 90
    neoadj_concomi_to_phase: int = 90
    neoadj_concomi_chemo2_start_gap: int = 14
    neoadj_concomi_chemo2_end_gap: int = 14


@dataclass
class Rule18Params:
    general_rule_days: int = 90
    neoadj_concomi_to_phase: int = 90
    neoadj_concomi_chemo2_start_gap: int = 14
    neoadj_concomi_chemo2_end_gap: int = 14
    neoadj_concomi_adj_to_next: int = 90


@dataclass
class Rule19Params:
    general_rule_days: int = 90
    neoadj_radio_adj_chemo1_to_radio: int = 90
    neoadj_radio_adj_chemo2_to_chemo: int = 90


@dataclass
class Rule20Params:
    general_rule_days: int = 90
    neoadj_chemo_to_surgery: int = 90
    surgery_postop_radio_days: int = 120


@dataclass
class Rule21Params:
    general_rule_days: int = 90
    surgery_adjuvant_chemo_days: int = 120
    adj_chemo_to_concomi_chemo: int = 90
    concomitant_start_gap: int = 14
    concomitant_end_gap: int = 14


@dataclass
class Rule22Params:
    general_rule_days: int = 90
    neoadj_chemo_to_surgery: int = 90
    surgery_adjuvant_chemo_days: int = 120
    concomitant_start_gap: int = 14
    concomitant_end_gap: int = 14


@dataclass
class Rule23Params:
    general_rule_days: int = 90
    neoadj_chemo_to_radio: int = 90
    radio_to_surgery: int = 90


@dataclass
class Rule24Params:
    general_rule_days: int = 90
    surgery_adjuvant_chemo_days: int = 120
    adj_chemo_to_radio: int = 90


# ---------------------------------------------------------------------------
# Rules 1–5: single modality only
# ---------------------------------------------------------------------------

def rule_only_surgery(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & ~has_any_systemic(df)
        & within_days_after(first_start(df, "surgery"), diag, p.general_rule_days)
    ).astype("boolean")


def rule_only_radio(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        has_treatment(df, "radio")
        & ~has_treatment(df, "surgery")
        & ~has_any_systemic(df)
        & within_days_after(first_start(df, "radio"), diag, p.general_rule_days)
    ).astype("boolean")


def rule_only_chemo(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(first_start(df, "chemo"), diag, p.general_rule_days)
    ).astype("boolean")


def rule_only_immuno(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        (count_treatments(df, "immuno") == 1)
        & ~has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & ~has_treatment(df, "chemo")
        & ~has_treatment(df, "targeted")
        & within_days_after(first_start(df, "immuno"), diag, p.general_rule_days)
    ).astype("boolean")


def rule_only_target(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        (count_treatments(df, "targeted") == 1)
        & ~has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & ~has_treatment(df, "chemo")
        & ~has_treatment(df, "immuno")
        & within_days_after(first_start(df, "targeted"), diag, p.general_rule_days)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 6: concomitant systemic + radio, no surgery
# ---------------------------------------------------------------------------

def rule_concomitant_systemic_radio(df: pd.DataFrame, p: Rule6Params) -> pd.Series:
    diag = df["diagnosis_date"]
    sys_start = first_systemic_start(df)
    sys_end = last_systemic_end(df)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_systemic(df) == 1)
        & dates_overlap(sys_start, sys_end, radio_start, radio_end)
        & within_symmetric_gap(sys_start, radio_start, p.concomitant_start_gap)
        & within_symmetric_gap(sys_end, radio_end, p.concomitant_end_gap)
        & (
            within_days_after(sys_start, diag, p.general_rule_days)
            | within_days_after(radio_start, diag, p.general_rule_days)
        )
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 7: surgery + post-op radio, no systemic
# ---------------------------------------------------------------------------

def rule_surgery_postop_radio(df: pd.DataFrame, p: Rule7Params) -> pd.Series:
    diag = df["diagnosis_date"]
    surg = first_start(df, "surgery")
    radio_start = first_start(df, "radio")

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & ~has_any_systemic(df)
        & within_days_after(surg, diag, p.general_rule_days)
        & within_days_after(radio_start, surg, p.surgery_postop_radio_days)
        & radio_start.notna() & surg.notna() & (radio_start > surg)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 8: surgery + adjuvant chemo, no radio
# ---------------------------------------------------------------------------

def rule_surgery_adj_chemo(df: pd.DataFrame, p: Rule8Params) -> pd.Series:
    diag = df["diagnosis_date"]
    surg = first_start(df, "surgery")
    chemo_start = first_start(df, "chemo")

    return (
        has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(surg, diag, p.general_rule_days)
        & within_days_after(chemo_start, surg, p.surgery_adjuvant_chemo_days)
        & chemo_start.notna() & surg.notna() & (chemo_start > surg)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 9: surgery + post-op radio + concomitant chemo
# ---------------------------------------------------------------------------

def rule_surgery_postop_radio_concomi_chemo(df: pd.DataFrame, p: Rule9Params) -> pd.Series:
    diag = df["diagnosis_date"]
    surg = first_start(df, "surgery")
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")
    chemo_start = first_start(df, "chemo")
    chemo_end = last_end(df, "chemo")

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(surg, diag, p.general_rule_days)
        & (
            within_days_after(chemo_start, surg, p.surgery_postop_radio_days)
            | within_days_after(radio_start, surg, p.surgery_postop_radio_days)
        )
        & within_symmetric_gap(chemo_start, radio_start, p.postop_radio_concomi_start_gap)
        & within_symmetric_gap(chemo_end, radio_end, p.postop_radio_concomi_end_gap)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 10: radio + adjuvant chemo, no surgery
# ---------------------------------------------------------------------------

def rule_radio_adj_chemo(df: pd.DataFrame, p: Rule10Params) -> pd.Series:
    diag = df["diagnosis_date"]
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")
    chemo_start = first_start(df, "chemo")

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(radio_start, diag, p.general_rule_days)
        & within_days_after(chemo_start, radio_end, p.radio_adjuvant_chemo_days)
        & chemo_start.notna() & radio_end.notna() & (chemo_start > radio_end)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 11: concomitant chemo-radio + adjuvant chemo (2 chemo lines)
# ---------------------------------------------------------------------------

def rule_concomi_chemo_radio_adj_chemo(df: pd.DataFrame, p: Rule11Params) -> pd.Series:
    diag = df["diagnosis_date"]
    chemo1_start = nth_start(df, "chemo", 1)
    chemo1_end = nth_end(df, "chemo", 1)
    chemo2_start = nth_start(df, "chemo", 2)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")

    # chemo2 must start after max(chemo1_end, radio_end)
    ref_end = pd.concat([chemo1_end.rename("c1e"), radio_end.rename("re")], axis=1).max(axis=1)

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 2)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & dates_overlap(chemo1_start, chemo1_end, radio_start, radio_end)
        & (
            within_days_after(chemo1_start, diag, p.general_rule_days)
            | within_days_after(radio_start, diag, p.general_rule_days)
        )
        & within_symmetric_gap(chemo1_start, radio_start, p.concomi_radio_adj_start_gap)
        & within_symmetric_gap(chemo1_end, radio_end, p.concomi_radio_adj_end_gap)
        & within_days_after(chemo2_start, ref_end, p.concomi_radio_adj_to_next)
        & chemo2_start.notna() & ref_end.notna() & (chemo2_start > ref_end)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rules 12–14: multi-systemic (no surgery, no radio)
# ---------------------------------------------------------------------------

def rule_chemo_immuno(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        ~has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & has_treatment(df, "chemo")
        & has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(first_start(df, "chemo"), diag, p.general_rule_days)
        & within_days_after(first_start(df, "immuno"), diag, p.general_rule_days)
    ).astype("boolean")


def rule_chemo_target(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        ~has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & has_treatment(df, "chemo")
        & ~has_treatment(df, "immuno")
        & has_treatment(df, "targeted")
        & within_days_after(first_start(df, "chemo"), diag, p.general_rule_days)
        & within_days_after(first_start(df, "targeted"), diag, p.general_rule_days)
    ).astype("boolean")


def rule_immuno_target(df: pd.DataFrame, p: GeneralParams) -> pd.Series:
    diag = df["diagnosis_date"]
    return (
        ~has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & ~has_treatment(df, "chemo")
        & has_treatment(df, "immuno")
        & has_treatment(df, "targeted")
        & within_days_after(first_start(df, "immuno"), diag, p.general_rule_days)
        & within_days_after(first_start(df, "targeted"), diag, p.general_rule_days)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rules 15–19: neoadjuvant patterns
# ---------------------------------------------------------------------------

def rule_neoadj_chemo_radio(df: pd.DataFrame, p: Rule15Params) -> pd.Series:
    """Neoadjuvant chemo followed by radiotherapy."""
    diag = df["diagnosis_date"]
    chemo_start = first_start(df, "chemo")
    chemo_end = last_end(df, "chemo")
    radio_start = first_start(df, "radio")

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(chemo_start, diag, p.general_rule_days)
        & within_days_after(radio_start, chemo_end, p.neoadj_chemo_to_radio)
        & chemo_start.notna() & radio_start.notna() & (chemo_start < radio_start)
    ).astype("boolean")


def rule_neoadj_chemo_surgery(df: pd.DataFrame, p: Rule16Params) -> pd.Series:
    """Neoadjuvant chemo followed by surgery."""
    diag = df["diagnosis_date"]
    chemo_start = first_start(df, "chemo")
    chemo_end = last_end(df, "chemo")
    surg = first_start(df, "surgery")

    return (
        has_treatment(df, "surgery")
        & ~has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(chemo_start, diag, p.general_rule_days)
        & within_days_after(surg, chemo_end, p.neoadj_chemo_to_surgery)
        & chemo_start.notna() & surg.notna() & (chemo_start < surg)
    ).astype("boolean")


def rule_neoadj_chemo_concomi_chemo_radio(df: pd.DataFrame, p: Rule17Params) -> pd.Series:
    """Neoadjuvant chemo (line 1) → concomitant chemo-radio (chemo line 2)."""
    diag = df["diagnosis_date"]
    chemo1_start = nth_start(df, "chemo", 1)
    chemo1_end = nth_end(df, "chemo", 1)
    chemo2_start = nth_start(df, "chemo", 2)
    chemo2_end = nth_end(df, "chemo", 2)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 2)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        # chemo1 before radio
        & chemo1_start.notna() & radio_start.notna() & (chemo1_start < radio_start)
        # chemo2 overlaps radio
        & dates_overlap(chemo2_start, chemo2_end, radio_start, radio_end)
        & within_days_after(chemo1_start, diag, p.general_rule_days)
        & (
            within_days_after(chemo2_start, chemo1_end, p.neoadj_concomi_to_phase)
            | within_days_after(radio_start, chemo1_end, p.neoadj_concomi_to_phase)
        )
        & within_symmetric_gap(chemo2_start, radio_start, p.neoadj_concomi_chemo2_start_gap)
        & within_symmetric_gap(chemo2_end, radio_end, p.neoadj_concomi_chemo2_end_gap)
    ).astype("boolean")


def rule_neoadj_chemo_concomi_chemo_radio_adj_chemo(df: pd.DataFrame, p: Rule18Params) -> pd.Series:
    """Neoadjuvant chemo → concomitant chemo-radio → adjuvant chemo (3 chemo lines)."""
    diag = df["diagnosis_date"]
    chemo1_start = nth_start(df, "chemo", 1)
    chemo1_end = nth_end(df, "chemo", 1)
    chemo2_start = nth_start(df, "chemo", 2)
    chemo2_end = nth_end(df, "chemo", 2)
    chemo3_start = nth_start(df, "chemo", 3)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")
    ref_end = pd.concat([chemo2_end.rename("c2e"), radio_end.rename("re")], axis=1).max(axis=1)

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 3)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & chemo1_start.notna() & radio_start.notna() & (chemo1_start < radio_start)
        & dates_overlap(chemo2_start, chemo2_end, radio_start, radio_end)
        & within_days_after(chemo1_start, diag, p.general_rule_days)
        & (
            within_days_after(chemo2_start, chemo1_end, p.neoadj_concomi_to_phase)
            | within_days_after(radio_start, chemo1_end, p.neoadj_concomi_to_phase)
        )
        & within_symmetric_gap(chemo2_start, radio_start, p.neoadj_concomi_chemo2_start_gap)
        & within_symmetric_gap(chemo2_end, radio_end, p.neoadj_concomi_chemo2_end_gap)
        & within_days_after(chemo3_start, ref_end, p.neoadj_concomi_adj_to_next)
        & chemo3_start.notna() & ref_end.notna() & (chemo3_start > ref_end)
    ).astype("boolean")


def rule_neoadj_chemo_radio_adj_chemo(df: pd.DataFrame, p: Rule19Params) -> pd.Series:
    """Neoadjuvant chemo → radio → adjuvant chemo (2 chemo lines)."""
    diag = df["diagnosis_date"]
    chemo1_start = nth_start(df, "chemo", 1)
    chemo1_end = nth_end(df, "chemo", 1)
    chemo2_start = nth_start(df, "chemo", 2)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")

    return (
        ~has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 2)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & chemo1_start.notna() & radio_start.notna() & (chemo1_start < radio_start)
        # chemo2 does NOT overlap radio (adjuvant = starts after radio)
        & chemo2_start.notna() & radio_end.notna() & (chemo2_start > radio_end)
        & within_days_after(chemo1_start, diag, p.general_rule_days)
        & within_days_after(radio_start, chemo1_end, p.neoadj_radio_adj_chemo1_to_radio)
        & within_days_after(chemo2_start, radio_end, p.neoadj_radio_adj_chemo2_to_chemo)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rules 20–24: surgery + radio + (neo)adjuvant chemo sequences
# ---------------------------------------------------------------------------

def rule_neoadj_chemo_surgery_radio(df: pd.DataFrame, p: Rule20Params) -> pd.Series:
    """Neoadjuvant chemo → surgery → post-op radio (1 chemo line)."""
    diag = df["diagnosis_date"]
    chemo_start = first_start(df, "chemo")
    chemo_end = last_end(df, "chemo")
    surg = first_start(df, "surgery")
    radio_start = first_start(df, "radio")

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(chemo_start, diag, p.general_rule_days)
        & chemo_start.notna() & surg.notna() & (chemo_start < surg)
        & within_days_after(surg, chemo_end, p.neoadj_chemo_to_surgery)
        & radio_start.notna() & (radio_start > surg)
        & within_days_after(radio_start, surg, p.surgery_postop_radio_days)
    ).astype("boolean")


def rule_surgery_adj_chemo_concomi_chemo_radio(df: pd.DataFrame, p: Rule21Params) -> pd.Series:
    """Surgery → adjuvant chemo (line 1) → concomitant chemo-radio (chemo line 2)."""
    diag = df["diagnosis_date"]
    surg = first_start(df, "surgery")
    chemo1_start = nth_start(df, "chemo", 1)
    chemo1_end = nth_end(df, "chemo", 1)
    chemo2_start = nth_start(df, "chemo", 2)
    chemo2_end = nth_end(df, "chemo", 2)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")

    # concomitant phase starts at the earlier of the chemo2 / radio start dates
    phase2_start = pd.concat(
        [chemo2_start.rename("c2s"), radio_start.rename("rs")], axis=1
    ).min(axis=1)

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 2)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(surg, diag, p.general_rule_days)
        & chemo1_start.notna() & surg.notna() & (chemo1_start > surg)
        & within_days_after(chemo1_start, surg, p.surgery_adjuvant_chemo_days)
        & dates_overlap(chemo2_start, chemo2_end, radio_start, radio_end)
        & within_symmetric_gap(chemo2_start, radio_start, p.concomitant_start_gap)
        & within_symmetric_gap(chemo2_end, radio_end, p.concomitant_end_gap)
        & phase2_start.notna() & chemo1_end.notna() & (phase2_start > chemo1_end)
        & within_days_after(phase2_start, chemo1_end, p.adj_chemo_to_concomi_chemo)
    ).astype("boolean")


def rule_neoadj_chemo_surgery_concomi_chemo_radio(df: pd.DataFrame, p: Rule22Params) -> pd.Series:
    """Neoadjuvant chemo (line 1) → surgery → concomitant chemo-radio (chemo line 2)."""
    diag = df["diagnosis_date"]
    surg = first_start(df, "surgery")
    chemo1_start = nth_start(df, "chemo", 1)
    chemo1_end = nth_end(df, "chemo", 1)
    chemo2_start = nth_start(df, "chemo", 2)
    chemo2_end = nth_end(df, "chemo", 2)
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")

    # concomitant phase starts at the earlier of the chemo2 / radio start dates
    phase2_start = pd.concat(
        [chemo2_start.rename("c2s"), radio_start.rename("rs")], axis=1
    ).min(axis=1)

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 2)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(chemo1_start, diag, p.general_rule_days)
        & chemo1_start.notna() & surg.notna() & (chemo1_start < surg)
        & within_days_after(surg, chemo1_end, p.neoadj_chemo_to_surgery)
        & dates_overlap(chemo2_start, chemo2_end, radio_start, radio_end)
        & within_symmetric_gap(chemo2_start, radio_start, p.concomitant_start_gap)
        & within_symmetric_gap(chemo2_end, radio_end, p.concomitant_end_gap)
        & phase2_start.notna() & (phase2_start > surg)
        & within_days_after(phase2_start, surg, p.surgery_adjuvant_chemo_days)
    ).astype("boolean")


def rule_neoadj_chemo_radio_surgery(df: pd.DataFrame, p: Rule23Params) -> pd.Series:
    """Neoadjuvant chemo → radio → surgery (1 chemo line)."""
    diag = df["diagnosis_date"]
    chemo_start = first_start(df, "chemo")
    chemo_end = last_end(df, "chemo")
    radio_start = first_start(df, "radio")
    radio_end = last_end(df, "radio")
    surg = first_start(df, "surgery")

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(chemo_start, diag, p.general_rule_days)
        & chemo_start.notna() & radio_start.notna() & (chemo_start < radio_start)
        & within_days_after(radio_start, chemo_end, p.neoadj_chemo_to_radio)
        & surg.notna() & radio_end.notna() & (surg > radio_end)
        & within_days_after(surg, radio_end, p.radio_to_surgery)
    ).astype("boolean")


def rule_surgery_adj_chemo_radio(df: pd.DataFrame, p: Rule24Params) -> pd.Series:
    """Surgery → adjuvant chemo → radio (1 chemo line)."""
    diag = df["diagnosis_date"]
    surg = first_start(df, "surgery")
    chemo_start = first_start(df, "chemo")
    chemo_end = last_end(df, "chemo")
    radio_start = first_start(df, "radio")

    return (
        has_treatment(df, "surgery")
        & has_treatment(df, "radio")
        & (count_treatments(df, "chemo") == 1)
        & ~has_treatment(df, "immuno")
        & ~has_treatment(df, "targeted")
        & within_days_after(surg, diag, p.general_rule_days)
        & chemo_start.notna() & surg.notna() & (chemo_start > surg)
        & within_days_after(chemo_start, surg, p.surgery_adjuvant_chemo_days)
        & radio_start.notna() & chemo_end.notna() & (radio_start > chemo_end)
        & within_days_after(radio_start, chemo_end, p.adj_chemo_to_radio)
    ).astype("boolean")


# ---------------------------------------------------------------------------
# Rule 25: other (none of rules 1–24)
# ---------------------------------------------------------------------------

def rule_other(rule_results: list[pd.Series]) -> pd.Series:
    combined = rule_results[0].copy()
    for r in rule_results[1:]:
        combined = combined | r
    return (~combined).astype("boolean")
