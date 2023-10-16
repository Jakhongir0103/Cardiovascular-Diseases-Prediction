# lists of features to use
from typing import Dict
from enum import Enum


class FeatureType(Enum):
    RANGE1_3 = 0
    RANGE1_5 = 1  # everything out of the range will be put to nan
    RANGE1_6 = 2
    RANGE1_30 = 3  # same here
    RANGE1_97 = 4
    BOOL = 5 # everything not in {1, 2} will be put to nan
    NUMERIC = 6
    FLAG = 7 # 0, 1 valid data


HEALTH_FEATURES: Dict[str, FeatureType] = {"GENHLTH": FeatureType.RANGE1_5,
                                           "PHYSHLTH": FeatureType.RANGE1_30,
                                           "MENTHLTH": FeatureType.RANGE1_30,
                                           "POORHLTH": FeatureType.RANGE1_30}

HEALTHCARE_FEATURES: Dict[str, FeatureType] = {"PERSDOC2": FeatureType.RANGE1_3,  # Look
                                               "MEDCOST": FeatureType.BOOL,
                                               "CHECKUP1": FeatureType.RANGE1_5,  # Look
                                               "_HCVU651": FeatureType.BOOL}

HYPERTENSION_FEATURES: Dict[str, FeatureType] = {"_CHOLCHK": FeatureType.RANGE1_3,
                                                 "_RFCHOL": FeatureType.BOOL}

CHRONIC_FEATURES: Dict[str, FeatureType] = {"CVDSTRK3": FeatureType.BOOL,
                                            "_ASTHMS1": FeatureType.RANGE1_3,
                                            "CHCSCNCR": FeatureType.BOOL,
                                            "CHCOCNCR": FeatureType.BOOL,
                                            "CHCCOPD1": FeatureType.BOOL,
                                            "_DRDXAR1": FeatureType.BOOL,
                                            "ADDEPEV2": FeatureType.BOOL,
                                            "CHCKIDNY": FeatureType.BOOL,
                                            "DIABETE3": FeatureType.RANGE1_5,
                                            "DIABAGE2": FeatureType.RANGE1_97}

DEMOGRAPHICS_FEATURES: Dict[str, FeatureType] = {"SEX": FeatureType.BOOL,
                                                 "_AGE80": FeatureType.NUMERIC,
                                                 "MARITAL": FeatureType.RANGE1_6,
                                                 "_CHLDCNT": FeatureType.RANGE1_97,
                                                 "_EDUCAG": FeatureType.RANGE1_5,
                                                 "_INCOMG": FeatureType.RANGE1_5,
                                                 "PREGNANT": FeatureType.BOOL,
                                                 "QLACTLM2": FeatureType.BOOL,
                                                 "USEEQUIP": FeatureType.BOOL,
                                                 "DECIDE": FeatureType.BOOL,
                                                 "DIFFWALK": FeatureType.BOOL,
                                                 "DIFFDRES": FeatureType.BOOL,
                                                 "DIFFALON": FeatureType.BOOL,
                                                 "HTM4": FeatureType.NUMERIC,
                                                 "WTKG3": FeatureType.NUMERIC,  # WARN: 99999 is NAN
                                                 "_BMI5": FeatureType.NUMERIC}

TOBACCO_FEATURES: Dict[str, FeatureType] = {"_SMOKER3": FeatureType.RANGE1_5,
                                            "USENOW3": FeatureType.RANGE1_3}

ALCOHOL_FEATURES: Dict[str, FeatureType] = {"DRNKANY5": FeatureType.BOOL,
                                            "DROCDY3_": FeatureType.NUMERIC, # WARN 900 is nan
                                            "_RFBING5": FeatureType.BOOL,
                                            "_DRNKWEK": FeatureType.NUMERIC, # WARN 99900 is nan
                                            "_RFDRHV5": FeatureType.BOOL}

FRUIT_FEATURES: Dict[str, FeatureType] = {"FTJUDA1_": FeatureType.NUMERIC,
                                          "FRUTDA1_": FeatureType.NUMERIC,
                                          "BEANDAY_": FeatureType.NUMERIC,
                                          "GRENDAY_": FeatureType.NUMERIC,
                                          "ORNGDAY_": FeatureType.NUMERIC,
                                          "VEGEDA1_": FeatureType.NUMERIC,
                                          "_MISFRTN": FeatureType.NUMERIC,
                                          "_MISVEGN": FeatureType.NUMERIC,
                                          "_FRUTSUM": FeatureType.NUMERIC,
                                          "_VEGESUM": FeatureType.NUMERIC,
                                          "_FRTLT1": FeatureType.NUMERIC,
                                          "_VEGLT1": FeatureType.BOOL,
                                          "_FRT16":  FeatureType.FLAG,
                                          "_VEG23": FeatureType.FLAG}

EXERCISE_FEATURES: Dict[str, FeatureType] = {"_TOTINDA": FeatureType.BOOL,
                                             "METVL11_": FeatureType.NUMERIC,
                                             "METVL21_": FeatureType.NUMERIC,
                                             "MAXVO2_": FeatureType.NUMERIC,  # warn 99900 is nan
                                             "ACTIN11_": FeatureType.NUMERIC,
                                             "ACTIN21_": FeatureType.NUMERIC,
                                             "PADUR1_": FeatureType.NUMERIC,
                                             "PADUR2_": FeatureType.NUMERIC,
                                             "PAFREQ1_": FeatureType.NUMERIC, # warn 99900 is nan
                                             "PAFREQ2_": FeatureType.NUMERIC, # warn 99900 is nan
                                             "_MINAC11": FeatureType.NUMERIC,
                                             "_MINAC21": FeatureType.NUMERIC,
                                             "STRFREQ_": FeatureType.NUMERIC, # warn 99900 is nan
                                             "PA1MIN_": FeatureType.NUMERIC,
                                             "PAVIG11_": FeatureType.NUMERIC,
                                             "PAVIG21_": FeatureType.NUMERIC,
                                             "PA1VIGM_": FeatureType.NUMERIC,
                                             "_PACAT1": FeatureType.RANGE1_5,
                                             "_PAINDX1": FeatureType.BOOL,
                                             "_PA150R2": FeatureType.RANGE1_3,
                                             "_PA300R2": FeatureType.RANGE1_3,
                                             "_PA30021": FeatureType.BOOL,
                                             "_PASTRNG": FeatureType.BOOL}
