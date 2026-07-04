"""Feature engineering for house-prices, applied to the combined train+test frame.

build_features(df) is the single entry point called by lib.common.load_data
before one-hot encoding and median imputation. It must be leak-free: every
transform is row-local or uses constants, never a target-derived statistic.
"""
import numpy as np
import pandas as pd

ORDINAL = {
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PavedDrive": {"N": 1, "P": 2, "Y": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "PoolQC": {"NA": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
}

FILL_NONE = ["Alley", "MasVnrType", "GarageType", "MiscFeature"]

FILL_ZERO = ["MasVnrArea", "GarageCars", "GarageArea", "TotalBsmtSF", "BsmtFinSF1",
             "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"]

CAST_STR = ["MSSubClass", "MoSold", "YrSold"]

LOG_COLS = ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea", "1stFlrSF",
            "MasVnrArea", "OpenPorchSF", "WoodDeckSF", "BsmtFinSF1", "LotFrontage"]


def build_features(df):
    df = df.copy()

    for col, mapping in ORDINAL.items():
        df[col] = df[col].fillna("NA").map(lambda v, m=mapping: m.get(v, 0)).astype(float)

    for col in FILL_NONE:
        df[col] = df[col].fillna("None").astype(str)

    for col in FILL_ZERO:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # GarageYrBlt missing means no garage; a known test typo records 2207.
    df["GarageYrBlt"] = pd.to_numeric(df["GarageYrBlt"], errors="coerce")
    df.loc[df["GarageYrBlt"] > 2010, "GarageYrBlt"] = df["YearBuilt"]
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    df["LotFrontage"] = pd.to_numeric(df["LotFrontage"], errors="coerce")

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalLivingSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBathrooms"] = (df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["TotalBsmtFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

    df["HouseAge"] = (df["YrSold"].astype(float) - df["YearBuilt"]).clip(lower=0)
    df["YearsSinceRemod"] = (df["YrSold"].astype(float) - df["YearRemodAdd"]).clip(lower=0)
    df["GarageAge"] = (df["YrSold"].astype(float) - df["GarageYrBlt"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNew"] = (df["HouseAge"] <= 1).astype(int)

    df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)
    df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)

    df["QualLivingArea"] = df["OverallQual"] * df["GrLivArea"]
    df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["QualSquared"] = df["OverallQual"] ** 2
    df["CondLivingArea"] = df["OverallCond"] * df["GrLivArea"]
    df["BsmtFinRatio"] = df["TotalBsmtFinSF"] / (df["TotalBsmtSF"] + 1.0)
    df["LivingAreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1.0)

    for col in CAST_STR:
        df[col] = df[col].astype(str)

    for col in LOG_COLS:
        df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))

    return df
