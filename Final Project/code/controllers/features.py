"Feature construction Module"

from functools import reduce

from finta import TA
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from toolz import curry

ON_CHAIN_COLS = [
    "BLOCK_SIZE",
    "HASH_RATE",
    "DIFFICULTY",
    "TRANSACTION_RATE",
    "ACTIVE_ADDRESSES",
    "NEW_ADDRESSES",
]

PCT_CHANGE_COLS = [
    "ATR_3",
    "ATR_7",
    "ATR_14",
    "ATR_32",
    "BB_UPPER_5",
    "BB_LOWER_5",
    "BB_MIDDLE_5",
    "BB_UPPER_10",
    "BB_LOWER_10",
    "BB_MIDDLE_10",
    "BB_UPPER_20",
    "BB_LOWER_20",
    "BB_MIDDLE_20",
] + ON_CHAIN_COLS


STANDARD_COLS = [
    "ROC_3",
    "ROC_6",
    "ROC_12",
    "ROC_24",
    "RSI_3",
    "RSI_7",
    "RSI_14",
    "RSI_32",
    "WILLIAMS_3",
    "WILLIAMS_7",
    "WILLIAMS_14",
    "WILLIAMS_32",
    "MI_3",
    "MI_9",
    "MI_18",
    "CCI_5",
    "CCI_10",
    "CCI_20",
    "BASP_BUY_10",
    "BASP_SELL_10",
    "BASP_BUY_20",
    "BASP_SELL_20",
    "BASP_BUY_40",
    "BASP_SELL_40",
    "ER_5",
    "ER_10",
    "ER_15",
    "ER_20",
    "MACD_LOW",
    "MACD_SIGNAL_LOW",
    "MACD_MID",
    "MACD_SIGNAL_MID",
    "MACD_HIGH",
    "MACD_SIGNAL_HIGH",
    "ADX_3",
    "ADX_7",
    "ADX_14",
    "ADX_21",
    "STOCH_3",
    "STOCH_7",
    "STOCH_14",
    "STOCH_21",
    "STOCHRSI_3",
    "STOCHRSI_7",
    "STOCHRSI_14",
    "STOCHRSI_21",
    "BBWIDTH_5",
    "BBWIDTH_10",
    "BBWIDTH_20",
    "PERCENT_B_5",
    "PERCENT_B_10",
    "PERCENT_B_20",
]

ALL_FEATURES = PCT_CHANGE_COLS + STANDARD_COLS

NON_FEATURE_COLS = ["open", "close", "high", "low", "volume"]


def get_scaler(standscaler_cols: list = PCT_CHANGE_COLS, minmaxscaler_cols: list = STANDARD_COLS):
    "Get Column Transformer Scaler for features"
    return ColumnTransformer(
        [("pct_change_cols", StandardScaler(), standscaler_cols), ("min_max_scaler", MinMaxScaler(), minmaxscaler_cols)]
    )


class Features:
    @staticmethod
    @curry
    def roc(df: pd.DataFrame, period: int):
        return TA.ROC(df, period=period).rename(f"ROC_{period}")

    @staticmethod
    @curry
    def rsi(df: pd.DataFrame, period: int):
        return TA.RSI(df, period=period).rename(f"RSI_{period}")

    @staticmethod
    @curry
    def atr(df: pd.DataFrame, period: int):
        return TA.ATR(df, period=period).rename(f"ATR_{period}")

    @staticmethod
    @curry
    def williams(df: pd.DataFrame, period: int):
        return TA.WILLIAMS(df, period=period).rename(f"WILLIAMS_{period}")

    @staticmethod
    @curry
    def mi(df: pd.DataFrame, period: int):
        return TA.MI(df, period=period).rename(f"MI_{period}")

    @staticmethod
    @curry
    def cci(df: pd.DataFrame, period: int):
        return TA.CCI(df, period=period).rename(f"CCI_{period}")

    @staticmethod
    @curry
    def basp(df: pd.DataFrame, period: int):
        return TA.BASP(df, period=period).rename({"Buy.": f"BASP_BUY_{period}", "Sell.": f"BASP_SELL_{period}"}, axis=1)

    @staticmethod
    @curry
    def bbands(df: pd.DataFrame, period: int):
        return TA.BBANDS(df, period=period).rename(
            {"BB_UPPER": f"BB_UPPER_{period}", "BB_LOWER": f"BB_LOWER_{period}", "BB_MIDDLE": f"BB_MIDDLE_{period}"},
            axis=1,
        )

    @staticmethod
    @curry
    def er(df: pd.DataFrame, period: int):
        return TA.ER(df, period=period).rename(f"ER_{period}")

    @staticmethod
    @curry
    def macd(df: pd.DataFrame, period_fast: int, period_slow: int, signal: int, level: str):
        return TA.MACD(df, period_fast=period_fast, period_slow=period_slow, signal=signal).rename(
            {"MACD": f"MACD_{level}", "SIGNAL": f"MACD_SIGNAL_{level}"}, axis=1
        )

    @staticmethod
    @curry
    def adx(df: pd.DataFrame, period: int):
        return TA.ADX(df, period=period).rename(f"ADX_{period}")

    @staticmethod
    @curry
    def stoch(df: pd.DataFrame, period: int):
        return TA.STOCH(df, period=period).rename(f"STOCH_{period}")

    @staticmethod
    @curry
    def stochrsi(df: pd.DataFrame, rsi_period: int, stoch_period: int):
        return TA.STOCHRSI(df, rsi_period, stoch_period).rename(f"STOCHRSI_{rsi_period}")

    @staticmethod
    @curry
    def bbwidth(df: pd.DataFrame, period: int):
        return TA.BBWIDTH(df, period=period).rename(f"BBWIDTH_{period}")

    @staticmethod
    @curry
    def percentb(df: pd.DataFrame, period: int):
        return TA.PERCENT_B(df, period=period).rename(f"PERCENT_B_{period}")

    @classmethod
    def mapping(cls):
        return {
            "ROC_3": cls.roc(period=3),
            "ROC_6": cls.roc(period=6),
            "ROC_12": cls.roc(period=12),
            "ROC_24": cls.roc(period=24),
            "RSI_3": cls.rsi(period=3),
            "RSI_7": cls.rsi(period=7),
            "RSI_14": cls.rsi(period=14),
            "RSI_32": cls.rsi(period=32),
            "ATR_3": cls.atr(period=3),
            "ATR_7": cls.atr(period=7),
            "ATR_14": cls.atr(period=14),
            "ATR_32": cls.atr(period=32),
            "WILLIAMS_3": cls.williams(period=3),
            "WILLIAMS_7": cls.williams(period=7),
            "WILLIAMS_14": cls.williams(period=14),
            "WILLIAMS_32": cls.williams(period=32),
            "MI_3": cls.mi(period=3),
            "MI_9": cls.mi(period=9),
            "MI_18": cls.mi(period=18),
            "CCI_5": cls.cci(period=5),
            "CCI_10": cls.cci(period=10),
            "CCI_20": cls.cci(period=20),
            "BASP_10": cls.basp(period=10),
            "BASP_20": cls.basp(period=20),
            "BASP_40": cls.basp(period=40),
            "BB_5": cls.bbands(period=5),
            "BB_10": cls.bbands(period=10),
            "BB_20": cls.bbands(period=20),
            "ER_5": cls.er(period=5),
            "ER_10": cls.er(period=10),
            "ER_15": cls.er(period=15),
            "ER_20": cls.er(period=20),
            "MACD_LOW": cls.macd(period_fast=6, period_slow=12, signal=4, level="LOW"),
            "MACD_MID": cls.macd(period_fast=12, period_slow=26, signal=9, level="MID"),
            "MACD_HIGH": cls.macd(period_fast=19, period_slow=39, signal=9, level="HIGH"),
            "ADX_3": cls.adx(period=3),
            "ADX_7": cls.adx(period=7),
            "ADX_14": cls.adx(period=14),
            "ADX_21": cls.adx(period=21),
            "STOCH_3": cls.stoch(period=3),
            "STOCH_7": cls.stoch(period=7),
            "STOCH_14": cls.stoch(period=14),
            "STOCH_21": cls.stoch(period=21),
            "STOCHRSI_3": cls.stochrsi(rsi_period=3, stoch_period=3),
            "STOCHRSI_7": cls.stochrsi(rsi_period=7, stoch_period=7),
            "STOCHRSI_14": cls.stochrsi(rsi_period=14, stoch_period=14),
            "STOCHRSI_21": cls.stochrsi(rsi_period=21, stoch_period=21),
            "BBWIDTH_5": cls.bbwidth(period=5),
            "BBWIDTH_10": cls.bbwidth(period=10),
            "BBWIDTH_20": cls.bbwidth(period=20),
            "PERCENT_B_5": cls.percentb(period=5),
            "PERCENT_B_10": cls.percentb(period=10),
            "PERCENT_B_20": cls.percentb(period=20),
        }

    @classmethod
    def multi_col_mapping(cls):
        return {
            "BASP_10": ["BASP_BUY_10", "BASP_SELL_10"],
            "BASP_20": ["BASP_BUY_20", "BASP_SELL_20"],
            "BASP_40": ["BASP_BUY_40", "BASP_SELL_40"],
            "BB_5": ["BB_UPPER_5", "BB_LOWER_5", "BB_MIDDLE_5"],
            "BB_10": ["BB_UPPER_10", "BB_LOWER_10", "BB_MIDDLE_10"],
            "BB_20": ["BB_UPPER_20", "BB_LOWER_20", "BB_MIDDLE_20"],
            "MACD_LOW": ["MACD_LOW", "MACD_SIGNAL_LOW"],
            "MACD_MID": ["MACD_MID", "MACD_SIGNAL_MID"],
            "MACD_HIGH": ["MACD_HIGH", "MACD_SIGNAL_HIGH"],
        }

    @classmethod
    def add_features(cls, ohlcv: pd.DataFrame, features: list = None):
        "Add all feature to the OHLCV data."

        indicators = [ohlcv.copy()]

        mapping_funcs = cls.mapping()

        if features is not None:
            for feature in features:
                if feature not in ON_CHAIN_COLS:
                    indicator = mapping_funcs[feature](df=ohlcv)
                    indicators.append(indicator)

        else:
            for m in mapping_funcs.values():
                indicators.append(m(df=ohlcv))

        df = reduce(lambda left, right: left.join(right), indicators)

        for col in df.columns:
            if col in PCT_CHANGE_COLS:
                df[col] = df[col].pct_change()

        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        col_maps = cls.multi_col_mapping()
        cols = []

        if features is not None:
            for x in features:
                if x in col_maps:
                    cols.extend(col_maps[x])
                else:
                    cols.append(x)
        else:
            cols = ALL_FEATURES

        return df[cols + ["TARGET"]]
