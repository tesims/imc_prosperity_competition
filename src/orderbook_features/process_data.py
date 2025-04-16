import os, glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from .feature_engineering import (
    read_and_clean,
    cast_and_fill,
    add_round_timestamp,
    add_base_product,
    add_volume_proxies,
    add_log_return,
    add_rolling_zscore,
    add_volume_weighted_return,
    add_bid_ask_spread,
    add_order_book_imbalance,
    add_depth_volatility,
    add_macd,
    add_rsi,
    add_cci,
    add_volatility,
)

class Processor:
    def __init__(self):
        self.spark = SparkSession.builder.appName("L2OrderBookFeatures").getOrCreate()

    def run(self, input_folder: str, output_folder: str):
        # 1) Read & clean each file
        files = glob.glob(os.path.join(input_folder, "*.csv"))
        dfs = [read_and_clean(self.spark, path) for path in files]

        # 2) Union into one DataFrame
        union_df = dfs[0]
        for df in dfs[1:]:
            union_df = union_df.union(df)

        # 3) Cast / fill nulls
        df = cast_and_fill(union_df)

        # 4) Derived features
        df = add_round_timestamp(df)
        df = add_base_product(df)
        df = add_volume_proxies(df)
        df = add_log_return(df)
        df = add_rolling_zscore(df)
        df = add_volume_weighted_return(df)
        df = add_bid_ask_spread(df)
        df = add_order_book_imbalance(df)
        df = add_depth_volatility(df)
        df = add_macd(df)
        df = add_rsi(df)
        df = add_cci(df)
        df = add_volatility(df)

        # 5) Split by base_product and write
        base_products = [r.base_product for r in df.select("base_product").distinct().collect()]
        for p in base_products:
            out_df = df.filter(col("base_product") == p)
            out_path = os.path.join(output_folder, p + ".csv")
            out_df.coalesce(1) \
                  .write \
                  .mode("overwrite") \
                  .option("header", "true") \
                  .csv(out_path)

        self.spark.stop()
