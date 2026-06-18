import pandas as pd

from vantage6.algorithm.decorator.action import preprocessing
from vantage6.algorithm.tools.util import info, error

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

_SUFFIXES = [
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


@preprocessing
def annotate_treatment_patterns(
    df: pd.DataFrame,
    prefix: str = "trt_pattern_",
    general_rule_days: int = 90,
    concomitant_start_gap: int = 14,
    concomitant_end_gap: int = 14,
    surgery_postop_radio_days: int = 120,
    surgery_adjuvant_chemo_days: int = 120,
    postop_radio_concomi_start_gap: int = 14,
    postop_radio_concomi_end_gap: int = 14,
    radio_adjuvant_chemo_days: int = 120,
    concomi_radio_adj_start_gap: int = 14,
    concomi_radio_adj_end_gap: int = 14,
    concomi_radio_adj_to_next: int = 90,
    chemo_immuno_days: int = 180,
    neoadj_chemo_to_radio: int = 90,
    neoadj_chemo_to_surgery: int = 90,
    neoadj_concomi_to_phase: int = 90,
    neoadj_concomi_chemo2_start_gap: int = 14,
    neoadj_concomi_chemo2_end_gap: int = 14,
    neoadj_concomi_adj_to_next: int = 90,
    neoadj_radio_adj_chemo1_to_radio: int = 90,
    neoadj_radio_adj_chemo2_to_chemo: int = 90,
) -> pd.DataFrame:

    old_df = df.copy()

    try:
        gp = GeneralParams(general_rule_days=general_rule_days)

        results = [
            rule_only_surgery(df, gp),
            rule_only_radio(df, gp),
            rule_only_chemo(df, gp),
            rule_only_immuno(df, gp),
            rule_only_target(df, gp),
            rule_concomitant_systemic_radio(df, Rule6Params(
                general_rule_days=general_rule_days,
                concomitant_start_gap=concomitant_start_gap,
                concomitant_end_gap=concomitant_end_gap,
            )),
            rule_surgery_postop_radio(df, Rule7Params(
                general_rule_days=general_rule_days,
                surgery_postop_radio_days=surgery_postop_radio_days,
            )),
            rule_surgery_adj_chemo(df, Rule8Params(
                general_rule_days=general_rule_days,
                surgery_adjuvant_chemo_days=surgery_adjuvant_chemo_days,
            )),
            rule_surgery_postop_radio_concomi_chemo(df, Rule9Params(
                general_rule_days=general_rule_days,
                surgery_postop_radio_days=surgery_postop_radio_days,
                postop_radio_concomi_start_gap=postop_radio_concomi_start_gap,
                postop_radio_concomi_end_gap=postop_radio_concomi_end_gap,
            )),
            rule_radio_adj_chemo(df, Rule10Params(
                general_rule_days=general_rule_days,
                radio_adjuvant_chemo_days=radio_adjuvant_chemo_days,
            )),
            rule_concomi_chemo_radio_adj_chemo(df, Rule11Params(
                general_rule_days=general_rule_days,
                concomi_radio_adj_start_gap=concomi_radio_adj_start_gap,
                concomi_radio_adj_end_gap=concomi_radio_adj_end_gap,
                concomi_radio_adj_to_next=concomi_radio_adj_to_next,
            )),
            rule_chemo_immuno(df, GeneralParams(general_rule_days=general_rule_days)),
            rule_chemo_target(df, GeneralParams(general_rule_days=general_rule_days)),
            rule_immuno_target(df, GeneralParams(general_rule_days=general_rule_days)),
            rule_neoadj_chemo_radio(df, Rule15Params(
                general_rule_days=general_rule_days,
                neoadj_chemo_to_radio=neoadj_chemo_to_radio,
            )),
            rule_neoadj_chemo_surgery(df, Rule16Params(
                general_rule_days=general_rule_days,
                neoadj_chemo_to_surgery=neoadj_chemo_to_surgery,
            )),
            rule_neoadj_chemo_concomi_chemo_radio(df, Rule17Params(
                general_rule_days=general_rule_days,
                neoadj_concomi_to_phase=neoadj_concomi_to_phase,
                neoadj_concomi_chemo2_start_gap=neoadj_concomi_chemo2_start_gap,
                neoadj_concomi_chemo2_end_gap=neoadj_concomi_chemo2_end_gap,
            )),
            rule_neoadj_chemo_concomi_chemo_radio_adj_chemo(df, Rule18Params(
                general_rule_days=general_rule_days,
                neoadj_concomi_to_phase=neoadj_concomi_to_phase,
                neoadj_concomi_chemo2_start_gap=neoadj_concomi_chemo2_start_gap,
                neoadj_concomi_chemo2_end_gap=neoadj_concomi_chemo2_end_gap,
                neoadj_concomi_adj_to_next=neoadj_concomi_adj_to_next,
            )),
            rule_neoadj_chemo_radio_adj_chemo(df, Rule19Params(
                general_rule_days=general_rule_days,
                neoadj_radio_adj_chemo1_to_radio=neoadj_radio_adj_chemo1_to_radio,
                neoadj_radio_adj_chemo2_to_chemo=neoadj_radio_adj_chemo2_to_chemo,
            )),
        ]

        results.append(rule_other(results))

        for suffix, series in zip(_SUFFIXES, results):
            col = f"{prefix}{suffix}"
            df[col] = series
            n = int(series.sum())
            info(f"{col}: {n} patients match")

        return df

    except Exception as exc:
        error(f"Failed to annotate treatment patterns: {exc}")
        return old_df
