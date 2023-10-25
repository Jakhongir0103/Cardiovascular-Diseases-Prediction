# lists of features to use
from typing import Dict, List, Tuple, Union
from enum import Enum


class FeatureType(Enum):
    BOOL = 5  #
    NUMERIC = 6
    FLAG = 7  # 0, 1 valid data
    RANGE = 8


class Feature:
    def __init__(
        self,
        feature_name: str,
        feature_type: FeatureType,
        max_value: int = None,
        nan_aliases: List[int] = None,
        map_values: Dict[int, int] = None,
    ):

        self.feature_name = feature_name
        self.feature_type = feature_type

        assert feature_type != FeatureType.RANGE or max_value is not None
        self.max_value = max_value

        if nan_aliases is None:
            nan_aliases = []
        if map_values is None:
            map_values = {}

        self.map_values = map_values
        self.nan_aliases: List[int]
        if feature_type == FeatureType.BOOL:
            self.nan_aliases = [7, 9] + nan_aliases
        else:
            self.nan_aliases = nan_aliases

    def __repr__(self):
        return self.feature_name

    def isnan(self, v):
        is_valid: bool = v not in self.nan_aliases

        if self.feature_type == FeatureType.BOOL:
            return not (v in {1, 2} and is_valid)
        elif self.feature_type == FeatureType.RANGE:
            return not (v <= self.max_value and is_valid)
        elif self.feature_type == FeatureType.NUMERIC:
            return not is_valid


####### Features Static Info #######

HEALTH_FEATURES: List[Feature] = [
    Feature("GENHLTH", FeatureType.RANGE, max_value=5),
    Feature(
        "PHYSHLTH", FeatureType.RANGE, max_value=30, map_values={88: 0}
    ),  # 88 means 0
    Feature(
        "MENTHLTH", FeatureType.RANGE, max_value=30, map_values={88: 0}
    ),  # 88 means 0
    Feature("POORHLTH", FeatureType.RANGE, max_value=30, map_values={88: 0}),
]  # 88 means 0

HEALTHCARE_FEATURES: List[Feature] = [
    Feature("PERSDOC2", FeatureType.RANGE, max_value=3),
    Feature("MEDCOST", FeatureType.BOOL),
    Feature(
        "CHECKUP1", FeatureType.RANGE, max_value=5, map_values={8: 5}
    ),  # 8 means never so put to 5
    Feature("_HCVU651", FeatureType.BOOL),
]

HYPERTENSION_FEATURES: List[Feature] = [
    Feature("_CHOLCHK", FeatureType.RANGE, max_value=3),
    Feature("_RFCHOL", FeatureType.BOOL),
]

CHRONIC_FEATURES: List[Feature] = [
    Feature("CVDSTRK3", FeatureType.BOOL),
    Feature("_ASTHMS1", FeatureType.RANGE, max_value=3),
    Feature("CHCSCNCR", FeatureType.BOOL),
    Feature("CHCOCNCR", FeatureType.BOOL),
    Feature("CHCCOPD1", FeatureType.BOOL),
    Feature("_DRDXAR1", FeatureType.BOOL),
    Feature("ADDEPEV2", FeatureType.BOOL),
    Feature("CHCKIDNY", FeatureType.BOOL),
    # inverted 3 and 4 (pre-diabetes worse than no diabetes)
    Feature("DIABETE3", FeatureType.RANGE, max_value=5, map_values={4: 3, 3: 4}),
    Feature("DIABAGE2", FeatureType.RANGE, max_value=97),
]

DEMOGRAPHICS_FEATURES: List[Feature] = [
    Feature("SEX", FeatureType.BOOL),
    Feature("_AGE80", FeatureType.NUMERIC),
    # changed the order to: married, couple, unmarried, separated, divorced, widowed
    Feature(
        "MARITAL", FeatureType.RANGE, max_value=6, map_values={6: 2, 5: 3, 2: 5, 3: 6}
    ),
    Feature("_CHLDCNT", FeatureType.RANGE, max_value=6),
    Feature("_EDUCAG", FeatureType.RANGE, max_value=4),
    Feature("_INCOMG", FeatureType.RANGE, max_value=5),
    Feature("PREGNANT", FeatureType.BOOL),
    Feature("QLACTLM2", FeatureType.BOOL),
    Feature("USEEQUIP", FeatureType.BOOL),
    Feature("DECIDE", FeatureType.BOOL),
    Feature("DIFFWALK", FeatureType.BOOL),
    Feature("DIFFDRES", FeatureType.BOOL),
    Feature("DIFFALON", FeatureType.BOOL),
    Feature("HTM4", FeatureType.NUMERIC),
    Feature("WTKG3", FeatureType.NUMERIC, nan_aliases=[99999]),
    Feature("_BMI5", FeatureType.NUMERIC),
]

TOBACCO_FEATURES: List[Feature] = [
    Feature("_SMOKER3", FeatureType.RANGE, max_value=4),
    Feature("USENOW3", FeatureType.RANGE, max_value=3),
]

ALCOHOL_FEATURES: List[Feature] = [
    Feature("DRNKANY5", FeatureType.BOOL),
    Feature("DROCDY3_", FeatureType.NUMERIC, nan_aliases=[900]),
    Feature("_RFBING5", FeatureType.BOOL),
    Feature("_DRNKWEK", FeatureType.NUMERIC, nan_aliases=[99900]),
    Feature("_RFDRHV5", FeatureType.BOOL),
]

FRUIT_FEATURES: List[Feature] = [
    Feature("FTJUDA1_", FeatureType.NUMERIC),
    Feature("FRUTDA1_", FeatureType.NUMERIC),
    Feature("BEANDAY_", FeatureType.NUMERIC),
    Feature("GRENDAY_", FeatureType.NUMERIC),
    Feature("ORNGDAY_", FeatureType.NUMERIC),
    Feature("VEGEDA1_", FeatureType.NUMERIC),
    Feature("_MISFRTN", FeatureType.RANGE, max_value=2),  # do we need this?
    Feature("_MISVEGN", FeatureType.RANGE, max_value=4),  # do we need this?
    Feature("_FRUTSUM", FeatureType.NUMERIC),
    Feature("_VEGESUM", FeatureType.NUMERIC),
    # Feature("_FRTLT1", FeatureType.BOOL), # commented in NAN_REPL_FRUIT
    # Feature("_VEGLT1", FeatureType.BOOL), # commented in NAN_REPL_FRUIT
    Feature("_FRT16", FeatureType.FLAG),  # do we need this?
    Feature("_VEG23", FeatureType.FLAG),
]  # do we need this?

EXERCISE_FEATURES: List[Feature] = [
    Feature("_TOTINDA", FeatureType.BOOL),
    Feature("METVL11_", FeatureType.NUMERIC),
    Feature("METVL21_", FeatureType.NUMERIC),
    Feature("MAXVO2_", FeatureType.NUMERIC, nan_aliases=[99900]),
    Feature("ACTIN11_", FeatureType.RANGE, max_value=2),
    Feature("ACTIN21_", FeatureType.RANGE, max_value=2),
    Feature("PADUR1_", FeatureType.NUMERIC),
    Feature("PADUR2_", FeatureType.NUMERIC),
    Feature("PAFREQ1_", FeatureType.NUMERIC, nan_aliases=[99900]),
    Feature("PAFREQ2_", FeatureType.NUMERIC, nan_aliases=[99900]),
    Feature("_MINAC11", FeatureType.NUMERIC),
    Feature("_MINAC21", FeatureType.NUMERIC),
    Feature("STRFREQ_", FeatureType.NUMERIC, nan_aliases=[99900]),
    Feature("PA1MIN_", FeatureType.NUMERIC),
    Feature("PAVIG11_", FeatureType.NUMERIC),
    Feature("PAVIG21_", FeatureType.NUMERIC),
    Feature("PA1VIGM_", FeatureType.NUMERIC),
    Feature("_PACAT1", FeatureType.RANGE, max_value=4),
    Feature("_PAINDX1", FeatureType.BOOL),
    Feature("_PA150R2", FeatureType.RANGE, max_value=3),
    Feature("_PA300R2", FeatureType.RANGE, max_value=3),
    Feature("_PA30021", FeatureType.BOOL),
    Feature("_PASTRNG", FeatureType.BOOL),
]

FEATURES_DICT: Dict[str, Feature] = {
    f.feature_name: f
    for f in HEALTH_FEATURES
    + HEALTHCARE_FEATURES
    + HYPERTENSION_FEATURES
    + CHRONIC_FEATURES
    + DEMOGRAPHICS_FEATURES
    + TOBACCO_FEATURES
    + ALCOHOL_FEATURES
    + FRUIT_FEATURES
    + EXERCISE_FEATURES
}

FEATURES_BY_CATEGORY = {
    "HEALTH_FEATURES": HEALTH_FEATURES,
    "HEALTHCARE_FEATURES": HEALTHCARE_FEATURES,
    "HYPERTENSION_FEATURES": HYPERTENSION_FEATURES,
    "CHRONIC_FEATURES": CHRONIC_FEATURES,
    "DEMOGRAPHICS_FEATURES": DEMOGRAPHICS_FEATURES,
    "TOBACCO_FEATURES": TOBACCO_FEATURES,
    "ALCOHOL_FEATURES": ALCOHOL_FEATURES,
    "FRUIT_FEATURES": FRUIT_FEATURES,
    "EXERCISE_FEATURES": EXERCISE_FEATURES,
}

####### Features NaN Replacement #######

NAN_REPL_HEALTH: Dict = {"GENHLTH": "mean", "PHYSHLTH": 0, "MENTHLTH": 0, "POORHLTH": 0}

NAN_REPL_HEALTHCARE: Dict = {"PERSDOC2": 3, "MEDCOST": 2, "CHECKUP1": 5, "_HCVU651": 0}

NAN_REPL_HYPERTENSION: Dict = {"_CHOLCHK": 0, "_RFCHOL": 0}  # Note: 13% of NaNs

NAN_REPL_CHRONIC: Dict = {
    "CVDSTRK3": 0,
    "_ASTHMS1": 2,
    "CHCSCNCR": 2,
    "CHCOCNCR": 2,
    "CHCCOPD1": 2,
    "_DRDXAR1": 2,
    "ADDEPEV2": 2,
    "CHCKIDNY": 2,
    "DIABETE3": 2,
    "DIABAGE2": "mean",
}

NAN_REPL_DEMOGRAPHICS: Dict = {  # 'SEX': ,
    # '_AGE80': ,
    "MARITAL": "mean",
    "_CHLDCNT": "mean",
    "_EDUCAG": "mean",
    "_INCOMG": "mean",
    "PREGNANT": 2,
    "QLACTLM2": 2,
    "USEEQUIP": 2,
    "DECIDE": 2,
    "DIFFWALK": 2,
    "DIFFDRES": 2,
    "DIFFALON": 2,
    "HTM4": "mean",
    "WTKG3": "mean",
    "_BMI5": "mean",
}

NAN_REPL_TOBACCO: Dict = {"_SMOKER3": 4, "USENOW3": 3}

NAN_REPL_ALCOHOL: Dict = {
    "DRNKANY5": "mean",
    "DROCDY3_": "mean",
    "_RFBING5": 1,
    "_DRNKWEK": "mean",
    "_RFDRHV5": 1,
}

NAN_REPL_FRUIT: Dict = {
    "FTJUDA1_": "mean",
    "FRUTDA1_": "mean",
    "BEANDAY_": "mean",
    "GRENDAY_": "mean",
    "ORNGDAY_": "mean",
    "VEGEDA1_": "mean",
    # '_MISFRTN': ,
    # '_MISVEGN': ,
    "_FRUTSUM": "mean",
    "_VEGESUM": "mean",
    # '_FRTLT1': ,
    # '_VEGLT1': ,
    # '_FRT16': ,
    # '_VEG23':
}

NAN_REPL_EXERCISE: Dict = {
    "_TOTINDA": 1,
    "METVL11_": "mean",
    "METVL21_": "median",  # TODO: put mean again, this is just a test
    "MAXVO2_": "mean",
    "ACTIN11_": 1,
    "ACTIN21_": 1,
    "PADUR1_": "mean",  # maybe median is better
    "PADUR2_": "mean",  # maybe median is better
    "PAFREQ1_": "mean",  # maybe median is better
    "PAFREQ2_": "mean",  # maybe median is better
    "_MINAC11": "mean",  # maybe median is better
    "_MINAC21": "mean",  # maybe median is better
    "STRFREQ_": "mean",  # maybe median is better
    "PA1MIN_": "mean",  # maybe median is better
    "PAVIG11_": "mean",  # maybe median is better
    "PAVIG21_": "mean",  # maybe median is better
    "PA1VIGM_": "mean",  # maybe median is better
    "_PACAT1": 4,
    "_PAINDX1": 2,
    "_PA150R2": 3,
    "_PA300R2": 2,
    "_PA30021": 2,
    "_PASTRNG": 2,
}

REPLACEMENT_DICT: Dict = {
    **NAN_REPL_HEALTH,
    **NAN_REPL_HEALTHCARE,
    **NAN_REPL_HYPERTENSION,
    **NAN_REPL_CHRONIC,
    **NAN_REPL_DEMOGRAPHICS,
    **NAN_REPL_TOBACCO,
    **NAN_REPL_ALCOHOL,
    **NAN_REPL_FRUIT,
    **NAN_REPL_EXERCISE,
}

REPLACEMENT_LIST: List[Tuple[List[str], Union[str, float]]] = [
    ([key for key in REPLACEMENT_DICT if REPLACEMENT_DICT[key] == v], v)
    for v in set(REPLACEMENT_DICT.values())
]
