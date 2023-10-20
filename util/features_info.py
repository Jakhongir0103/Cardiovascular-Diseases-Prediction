# lists of features to use
from typing import Dict, List
from enum import Enum


class FeatureType(Enum):

    BOOL = 5  #
    NUMERIC = 6
    FLAG = 7  # 0, 1 valid data
    RANGE = 8


class Feature:
    def __init__(self,
                 feature_name: str,
                 feature_type: FeatureType,
                 max_value: int = None,
                 nan_aliases: List[int] = None):

        self.feature_name = feature_name
        self.feature_type = feature_type

        assert feature_type != FeatureType.RANGE or max_value is not None
        self.max_value = max_value

        if nan_aliases is None:
            nan_aliases = []

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



HEALTH_FEATURES: List[Feature] = [Feature("GENHLTH", FeatureType.RANGE, max_value=5),   # ok
                                  Feature("PHYSHLTH", FeatureType.RANGE, max_value=30), # 88 is 0
                                  Feature("MENTHLTH", FeatureType.RANGE, max_value=30), # 88 is 0
                                  Feature("POORHLTH", FeatureType.RANGE, max_value=30)] # 88 is 0

HEALTHCARE_FEATURES: List[Feature] = [Feature("PERSDOC2", FeatureType.RANGE, max_value=3),  # ok
                                      Feature("MEDCOST", FeatureType.BOOL),                 # ok
                                      Feature("CHECKUP1", FeatureType.RANGE, max_value=5),  # 8 should mean infinite (put to 5 or bigger)
                                      Feature("_HCVU651", FeatureType.BOOL)]                # ok

HYPERTENSION_FEATURES: List[Feature] = [Feature("_CHOLCHK", FeatureType.RANGE, max_value=3),    # ok
                                        Feature("_RFCHOL", FeatureType.BOOL)]                   # ok

CHRONIC_FEATURES: List[Feature] = [Feature("CVDSTRK3", FeatureType.BOOL),   # ok
                                   Feature("_ASTHMS1", FeatureType.RANGE, max_value=3), # ok
                                   Feature("CHCSCNCR", FeatureType.BOOL),   # ok
                                   Feature("CHCOCNCR", FeatureType.BOOL),   # ok
                                   Feature("CHCCOPD1", FeatureType.BOOL),   # ok
                                   Feature("_DRDXAR1", FeatureType.BOOL),   # ok
                                   Feature("ADDEPEV2", FeatureType.BOOL),   # ok
                                   Feature("CHCKIDNY", FeatureType.BOOL),   # ok
                                   Feature("DIABETE3", FeatureType.RANGE, max_value=5),     # maybe 4 should go before 3
                                   Feature("DIABAGE2", FeatureType.RANGE, max_value=97)]    # ok

DEMOGRAPHICS_FEATURES: List[Feature] = [Feature("SEX", FeatureType.BOOL),       # ok
                                        Feature("_AGE80", FeatureType.NUMERIC), # maybe should be range
                                        Feature("MARITAL", FeatureType.RANGE, max_value=6),     # think about the order
                                        Feature("_CHLDCNT", FeatureType.RANGE, max_value=6),    # ok
                                        Feature("_EDUCAG", FeatureType.RANGE, max_value=4),     # ok
                                        Feature("_INCOMG", FeatureType.RANGE, max_value=5),     # ok
                                        Feature("PREGNANT", FeatureType.BOOL),  # ok
                                        Feature("QLACTLM2", FeatureType.BOOL),  # ok
                                        Feature("USEEQUIP", FeatureType.BOOL),  # ok
                                        Feature("DECIDE", FeatureType.BOOL),    # ok  
                                        Feature("DIFFWALK", FeatureType.BOOL),  # ok
                                        Feature("DIFFDRES", FeatureType.BOOL),  # ok
                                        Feature("DIFFALON", FeatureType.BOOL),  # ok
                                        Feature("HTM4", FeatureType.NUMERIC),   # ok
                                        Feature("WTKG3", FeatureType.NUMERIC, nan_aliases=[99999]), # ok
                                        Feature("_BMI5", FeatureType.NUMERIC)]  # ok

TOBACCO_FEATURES: List[Feature] = [Feature("_SMOKER3", FeatureType.RANGE, max_value=4), # ok
                                   Feature("USENOW3", FeatureType.RANGE, max_value=3)]  # ok

ALCOHOL_FEATURES: List[Feature] = [Feature("DRNKANY5", FeatureType.BOOL),   # ok
                                   Feature("DROCDY3_", FeatureType.NUMERIC, nan_aliases=[900]), # ok
                                   Feature("_RFBING5", FeatureType.BOOL),   # ok
                                   Feature("_DRNKWEK", FeatureType.NUMERIC, nan_aliases=[99900]), # ok
                                   Feature("_RFDRHV5", FeatureType.BOOL)]   # ok

FRUIT_FEATURES: List[Feature] = [Feature("FTJUDA1_", FeatureType.NUMERIC),  # ok
                                 Feature("FRUTDA1_", FeatureType.NUMERIC),  # ok
                                 Feature("BEANDAY_", FeatureType.NUMERIC),  # ok
                                 Feature("GRENDAY_", FeatureType.NUMERIC),  # ok
                                 Feature("ORNGDAY_", FeatureType.NUMERIC),  # ok
                                 Feature("VEGEDA1_", FeatureType.NUMERIC),  # ok
                                 Feature("_MISFRTN", FeatureType.RANGE, max_value=2), # do we need this?
                                 Feature("_MISVEGN", FeatureType.RANGE, max_value=4), # do we need this?
                                 Feature("_FRUTSUM", FeatureType.NUMERIC),  # ok
                                 Feature("_VEGESUM", FeatureType.NUMERIC),  # ok
                                 Feature("_FRTLT1", FeatureType.BOOL),  # ok
                                 Feature("_VEGLT1", FeatureType.BOOL),  # ok
                                 Feature("_FRT16",  FeatureType.FLAG),  # do we need this?
                                 Feature("_VEG23", FeatureType.FLAG)]   # do we need this?

EXERCISE_FEATURES: List[Feature] = [Feature("_TOTINDA", FeatureType.BOOL),  # ok
                                    Feature("METVL11_", FeatureType.NUMERIC),   # ok
                                    Feature("METVL21_", FeatureType.NUMERIC),   # ok
                                    Feature("MAXVO2_", FeatureType.NUMERIC, nan_aliases=[99900]),   # ok
                                    Feature("ACTIN11_", FeatureType.RANGE, max_value=2),   # ok
                                    Feature("ACTIN21_", FeatureType.RANGE, max_value=2),   # ok
                                    Feature("PADUR1_", FeatureType.NUMERIC),    # ok
                                    Feature("PADUR2_", FeatureType.NUMERIC),    # ok
                                    Feature("PAFREQ1_", FeatureType.NUMERIC, nan_aliases=[99900]),  # ok
                                    Feature("PAFREQ2_", FeatureType.NUMERIC, nan_aliases=[99900]),  # ok   
                                    Feature("_MINAC11", FeatureType.NUMERIC),   # ok
                                    Feature("_MINAC21", FeatureType.NUMERIC),   # ok
                                    Feature("STRFREQ_", FeatureType.NUMERIC, nan_aliases=[99900]),  # ok
                                    Feature("PA1MIN_", FeatureType.NUMERIC),    # ok
                                    Feature("PAVIG11_", FeatureType.NUMERIC),   # ok
                                    Feature("PAVIG21_", FeatureType.NUMERIC),   # ok
                                    Feature("PA1VIGM_", FeatureType.NUMERIC),   # ok
                                    Feature("_PACAT1",  FeatureType.RANGE, max_value=4),    # ok
                                    Feature("_PAINDX1", FeatureType.BOOL),  # ok
                                    Feature("_PA150R2", FeatureType.RANGE, max_value=3),    # ok
                                    Feature("_PA300R2", FeatureType.RANGE, max_value=3),    # ok
                                    Feature("_PA30021", FeatureType.BOOL),  # ok
                                    Feature("_PASTRNG", FeatureType.BOOL)]  # ok

FEATURES_DICT: Dict[str, Feature] = {f.feature_name: f for f in
                                     HEALTH_FEATURES + HEALTHCARE_FEATURES + HYPERTENSION_FEATURES + CHRONIC_FEATURES +
                                     DEMOGRAPHICS_FEATURES + TOBACCO_FEATURES + ALCOHOL_FEATURES + FRUIT_FEATURES +
                                     EXERCISE_FEATURES}

#HEALTH_FEATURES: Dict[str, FeatureType] = {"GENHLTH": FeatureType.RANGE1_5,
#                                           "PHYSHLTH": FeatureType.RANGE1_30,
#                                           "MENTHLTH": FeatureType.RANGE1_30,
#                                           "POORHLTH": FeatureType.RANGE1_30}

#HEALTHCARE_FEATURES: Dict[str, FeatureType] = {"PERSDOC2": FeatureType.RANGE1_3,  # Look
#                                               "MEDCOST": FeatureType.BOOL,
#                                               "CHECKUP1": FeatureType.RANGE1_5,  # Look
#                                               "_HCVU651": FeatureType.BOOL}

#HYPERTENSION_FEATURES: Dict[str, FeatureType] = {"_CHOLCHK": FeatureType.RANGE1_3,
#                                                 "_RFCHOL": FeatureType.BOOL}

#CHRONIC_FEATURES: Dict[str, FeatureType] = {"CVDSTRK3": FeatureType.BOOL,
#                                            "_ASTHMS1": FeatureType.RANGE1_3,
#                                            "CHCSCNCR": FeatureType.BOOL,
#                                            "CHCOCNCR": FeatureType.BOOL,
#                                            "CHCCOPD1": FeatureType.BOOL,
#                                            "_DRDXAR1": FeatureType.BOOL,
#                                            "ADDEPEV2": FeatureType.BOOL,
#                                            "CHCKIDNY": FeatureType.BOOL,
#                                            "DIABETE3": FeatureType.RANGE1_5,
#                                            "DIABAGE2": FeatureType.RANGE1_97}

#DEMOGRAPHICS_FEATURES: Dict[str, FeatureType] = {"SEX": FeatureType.BOOL,
#                                                 "_AGE80": FeatureType.NUMERIC,
#                                                 "MARITAL": FeatureType.RANGE1_6,
#                                                 "_CHLDCNT": FeatureType.RANGE1_97,
#                                                 "_EDUCAG": FeatureType.RANGE1_5,
#                                                 "_INCOMG": FeatureType.RANGE1_5,
#                                                 "PREGNANT": FeatureType.BOOL,
#                                                 "QLACTLM2": FeatureType.BOOL,
#                                                 "USEEQUIP": FeatureType.BOOL,
#                                                 "DECIDE": FeatureType.BOOL,
#                                                 "DIFFWALK": FeatureType.BOOL,
#                                                 "DIFFDRES": FeatureType.BOOL,
#                                                 "DIFFALON": FeatureType.BOOL,
#                                                 "HTM4": FeatureType.NUMERIC,
#                                                 "WTKG3": FeatureType.NUMERIC,  # WARN: 99999 is NAN
#                                                 "_BMI5": FeatureType.NUMERIC}

#TOBACCO_FEATURES: Dict[str, FeatureType] = {"_SMOKER3": FeatureType.RANGE1_5,
#                                            "USENOW3": FeatureType.RANGE1_3}

#ALCOHOL_FEATURES: Dict[str, FeatureType] = {"DRNKANY5": FeatureType.BOOL,
#                                            "DROCDY3_": FeatureType.NUMERIC, # WARN 900 is nan
#                                            "_RFBING5": FeatureType.BOOL,
#                                            "_DRNKWEK": FeatureType.NUMERIC, # WARN 99900 is nan
#                                            "_RFDRHV5": FeatureType.BOOL}

#FRUIT_FEATURES: Dict[str, FeatureType] = {"FTJUDA1_": FeatureType.NUMERIC,
#                                          "FRUTDA1_": FeatureType.NUMERIC,
#                                          "BEANDAY_": FeatureType.NUMERIC,
#                                          "GRENDAY_": FeatureType.NUMERIC,
#                                          "ORNGDAY_": FeatureType.NUMERIC,
#                                          "VEGEDA1_": FeatureType.NUMERIC,
#                                          "_MISFRTN": FeatureType.NUMERIC,
#                                          "_MISVEGN": FeatureType.NUMERIC,
#                                          "_FRUTSUM": FeatureType.NUMERIC,
#                                          "_VEGESUM": FeatureType.NUMERIC,
#                                          "_FRTLT1": FeatureType.NUMERIC,
#                                          "_VEGLT1": FeatureType.BOOL,
#                                          "_FRT16":  FeatureType.FLAG,
#                                          "_VEG23": FeatureType.FLAG}

#EXERCISE_FEATURES: Dict[str, FeatureType] = {"_TOTINDA": FeatureType.BOOL,
#                                             "METVL11_": FeatureType.NUMERIC,
#                                             "METVL21_": FeatureType.NUMERIC,
#                                             "MAXVO2_": FeatureType.NUMERIC,  # warn 99900 is nan
#                                             "ACTIN11_": FeatureType.NUMERIC,
#                                             "ACTIN21_": FeatureType.NUMERIC,
#                                             "PADUR1_": FeatureType.NUMERIC,
#                                             "PADUR2_": FeatureType.NUMERIC,
#                                             "PAFREQ1_": FeatureType.NUMERIC, # warn 99900 is nan
#                                             "PAFREQ2_": FeatureType.NUMERIC, # warn 99900 is nan
#                                             "_MINAC11": FeatureType.NUMERIC,
#                                             "_MINAC21": FeatureType.NUMERIC,
#                                             "STRFREQ_": FeatureType.NUMERIC, # warn 99900 is nan
#                                             "PA1MIN_": FeatureType.NUMERIC,
#                                             "PAVIG11_": FeatureType.NUMERIC,
#                                             "PAVIG21_": FeatureType.NUMERIC,
#                                             "PA1VIGM_": FeatureType.NUMERIC,
#                                             "_PACAT1": FeatureType.RANGE1_5,
#                                             "_PAINDX1": FeatureType.BOOL,
#                                             "_PA150R2": FeatureType.RANGE1_3,
#                                             "_PA300R2": FeatureType.RANGE1_3,
#                                             "_PA30021": FeatureType.BOOL,
#                                             "_PASTRNG": FeatureType.BOOL}
