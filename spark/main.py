from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import count
from pyspark.sql.functions import col, hour, when, lit
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession

def main():
    # Создание Spark-сессии
    spark = (
        SparkSession
        .builder
        .appName('job_save_table')
        .getOrCreate()
    )

    trips_df = spark.read.parquet("s3a://hse-storage-spark/*.parquet")
    zones_df = spark.read.csv("s3a://hse-storage-spark/taxi_zone_lookup.csv", header=True, inferSchema=True)
    trips_df = trips_df \
        .withColumn("pickup_datetime", to_timestamp(col("tpep_pickup_datetime"))) \
        .withColumn("dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime")))

    # Фильтрация по датам и другим условиям
    filtered_trips = trips_df \
        .filter(
            (col("pickup_datetime") >= "2025-01-01 00:00:00") &
            (col("pickup_datetime") <= "2025-06-30 23:59:59") &
            (col("dropoff_datetime") >= "2025-01-01 00:00:00") &
            (col("dropoff_datetime") <= "2025-06-30 23:59:59") &
            (col("trip_distance") > 0) &
            (col("passenger_count") > 0)
        )

    trips_with_hours = filtered_trips \
        .withColumn("pickup_hour", hour(col("pickup_datetime"))) \
        .withColumn("dropoff_hour", hour(col("dropoff_datetime")))

    selected_columns = [
        "pickup_datetime",
        "dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",      # идентификатор зоны посадки
        "DOLocationID",      # идентификатор зоны высадки
        "total_amount",      # полная стоимость
        "pickup_hour",
        "dropoff_hour"
    ]

    cleaned_trips = trips_with_hours.select(*selected_columns)

    # Переименуем колонки в zones_df для ясности
    zones_renamed = zones_df.withColumnRenamed("LocationID", "zone_id")

    # Присоединяем названия зон посадки
    trips_with_pickup_zone = cleaned_trips \
        .join(zones_renamed, cleaned_trips["PULocationID"] == zones_renamed["zone_id"], "left") \
        .drop("zone_id") \
        .withColumnRenamed("Zone", "pickup_zone") \
        .withColumnRenamed("Borough", "pickup_borough")

    # Присоединяем названия зон высадки
    trips_with_zones = trips_with_pickup_zone \
        .join(zones_renamed, trips_with_pickup_zone["DOLocationID"] == zones_renamed["zone_id"], "left") \
        .drop("zone_id") \
        .withColumnRenamed("Zone", "dropoff_zone") \
        .withColumnRenamed("Borough", "dropoff_borough")
    
    hourly_counts = trips_with_zones \
        .groupBy("pickup_zone", "pickup_hour") \
        .agg(count("*").alias("trip_count"))


    pivoted = hourly_counts.groupBy("pickup_zone").pivot("pickup_hour", [i for i in range(24)]).agg({"trip_count": "avg"}).fillna(0)

    # Переименуем колонки: 0 -> hour_0, 1 -> hour_1 и т.д.
    for i in range(24):
        pivoted = pivoted.withColumnRenamed(str(i), f"hour_{i}")

    pivoted.write \
        .mode("overwrite") \
        .parquet("s3a://hse-storage-spark/result/result_table.parquet")
    
    #spark.stop()

if __name__ == '__main__':
    main()
