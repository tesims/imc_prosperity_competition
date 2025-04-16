from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, when, least, greatest, log, avg, stddev, lag, sum as _sum, abs as _abs
)

# 1) Read & clean single file
def read_and_clean(spark: SparkSession, path: str) -> DataFrame:
    raw = spark.read.option("header","false").option("delimiter",";").csv(path)
    df = raw.filter(~raw["_c0"].startswith("day"))
    prices_feat = [
        "day","timestamp","product",
        "bid_price_1","bid_volume_1",
        "bid_price_2","bid_volume_2",
        "bid_price_3","bid_volume_3",
        "ask_price_1","ask_volume_1",
        "ask_price_2","ask_volume_2",
        "ask_price_3","ask_volume_3",
        "mid_price","profit_and_loss"
    ]
    return df.toDF(*prices_feat)

# 2) Cast numeric & fill nulls
def cast_and_fill(df: DataFrame) -> DataFrame:
    cols = [
        "bid_price_1","bid_volume_1","bid_price_2","bid_volume_2",
        "bid_price_3","bid_volume_3","ask_price_1","ask_volume_1",
        "ask_price_2","ask_volume_2","ask_price_3","ask_volume_3",
        "mid_price","profit_and_loss"
    ]
    for c in cols:
        df = df.withColumn(c, col(c).cast("double"))
    return df.fillna(0, subset=cols)

# 3) round_timestamp
def add_round_timestamp(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "round_timestamp",
        col("timestamp").cast("long") + 1000000 * col("day").cast("long")
    )

# 4) base_product
def add_base_product(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "base_product",
        when(col("product").startswith("VOLCANIC_ROCK_VOUCHER_"),
             "VOLCANIC_ROCK_VOUCHER")
        .otherwise(col("product"))
    )

# 5) volume proxies
def add_volume_proxies(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "trade_volume_best",
        least(col("bid_volume_1"), col("ask_volume_1"))
    ).withColumn(
        "total_bid_volume",
        col("bid_volume_1") + col("bid_volume_2") + col("bid_volume_3")
    ).withColumn(
        "total_ask_volume",
        col("ask_volume_1") + col("ask_volume_2") + col("ask_volume_3")
    ).withColumn(
        "trade_volume_total",
        least(col("total_bid_volume"), col("total_ask_volume"))
    )

# 6) normalized mid-price return
def add_log_return(df: DataFrame) -> DataFrame:
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp")
    return df.withColumn("lag_mid_price", lag("mid_price").over(win)) \
             .withColumn("log_return", log(col("mid_price")/col("lag_mid_price")))

# 7) rolling Z-score
def add_rolling_zscore(df: DataFrame, window_size:int=10) -> DataFrame:
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp") \
                .rowsBetween(-window_size+1,0)
    return df.withColumn("rolling_mean", avg("mid_price").over(win)) \
             .withColumn("rolling_stddev", stddev("mid_price").over(win)) \
             .withColumn("rolling_zscore",
                         (col("mid_price")-col("rolling_mean"))/col("rolling_stddev"))

# 8) volume-weighted return
def add_volume_weighted_return(df: DataFrame, window_size:int=10) -> DataFrame:
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp") \
                .rowsBetween(-window_size+1,0)
    return df.withColumn(
        "volume_weighted_return",
        _sum(col("log_return")*col("trade_volume_best")).over(win)
        / _sum(col("trade_volume_best")).over(win)
    )

# 9) bid-ask spread
def add_bid_ask_spread(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "bid_ask_spread",
        col("ask_price_1") - col("bid_price_1")
    )

# 10) order-book imbalance
def add_order_book_imbalance(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "order_book_imbalance",
        (col("bid_volume_1") - col("ask_volume_1")) /
        (col("bid_volume_1") + col("ask_volume_1"))
    )

# 11) depth volatility
def add_depth_volatility(df: DataFrame, window_size:int=10) -> DataFrame:
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp") \
                .rowsBetween(-window_size+1,0)
    return df.withColumn(
        "depth_volatility",
        stddev(col("bid_volume_1")+col("ask_volume_1")).over(win)
    )

# 12) MACD (SMA proxy)
def add_macd(df: DataFrame) -> DataFrame:
    fast = Window.partitionBy("base_product") \
                 .orderBy("round_timestamp") \
                 .rowsBetween(-11,0)
    slow = Window.partitionBy("base_product") \
                 .orderBy("round_timestamp") \
                 .rowsBetween(-25,0)
    sig  = Window.partitionBy("base_product") \
                 .orderBy("round_timestamp") \
                 .rowsBetween(-8,0)
    return df.withColumn("fast_ma", avg("mid_price").over(fast)) \
             .withColumn("slow_ma", avg("mid_price").over(slow)) \
             .withColumn("macd_line", col("fast_ma")-col("slow_ma")) \
             .withColumn("macd_signal", avg("macd_line").over(sig))

# 13) RSI
def add_rsi(df: DataFrame) -> DataFrame:
    win0 = Window.partitionBy("base_product") \
                 .orderBy("round_timestamp")
    df1 = df.withColumn("prev_mid_price", lag("mid_price").over(win0)) \
            .withColumn("diff", col("mid_price")-col("prev_mid_price")) \
            .withColumn("gain", when(col("diff")>0,col("diff")).otherwise(0)) \
            .withColumn("loss", when(col("diff")<0,-col("diff")).otherwise(0))
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp") \
                .rowsBetween(-13,0)
    return df1.withColumn("avg_gain", avg("gain").over(win)) \
              .withColumn("avg_loss", avg("loss").over(win)) \
              .withColumn("rs", col("avg_gain")/col("avg_loss")) \
              .withColumn("rsi", 100-(100/(1+col("rs"))))

# 14) CCI
def add_cci(df: DataFrame) -> DataFrame:
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp") \
                .rowsBetween(-19,0)
    return df.withColumn("sma", avg("mid_price").over(win)) \
             .withColumn("mad", avg(_abs(col("mid_price")-col("sma"))).over(win)) \
             .withColumn("cci", (col("mid_price")-col("sma"))/(0.015*col("mad")))

# 15) rolling volatility
def add_volatility(df: DataFrame, window_size:int=10) -> DataFrame:
    win = Window.partitionBy("base_product") \
                .orderBy("round_timestamp") \
                .rowsBetween(-window_size+1,0)
    return df.withColumn("rolling_volatility", stddev("log_return").over(win))
