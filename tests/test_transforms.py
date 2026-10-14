import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("PySparkUnitTests") \
        .master("local[2]") \
        .getOrCreate()

def test_enriquecimiento_estaciones(spark):
    """Verifica que la asignación de estaciones sea correcta según el mes."""
    data = [(1, "January"), (4, "April"), (7, "July"), (10, "October")]
    schema = StructType([
        StructField("month", IntegerType(), False),
        StructField("month_name", StringType(), False)
    ])
    df = spark.createDataFrame(data, schema)

    df_result = df.withColumn(
        "seasons",
        F.when(F.col("month").isin(12, 1, 2), "Winter")
        .when(F.col("month").isin(3, 4, 5), "Spring")
        .when(F.col("month").isin(6, 7, 8), "Summer")
        .otherwise("Fall")
    )

    estaciones = {row["month"]: row["seasons"] for row in df_result.collect()}
    assert estaciones[1] == "Winter"
    assert estaciones[4] == "Spring"
    assert estaciones[7] == "Summer"
    assert estaciones[10] == "Fall"

def test_reglas_calidad_limites(spark):
    """Verifica que el filtro descarte valores físicamente imposibles."""
    data = [(-30.0, 50), (20.0, 120), (15.0, 65)]  #row 1 invalid temp, row 2 invalid humidity
    schema = StructType([
        StructField("temperature_celsius", DoubleType(), False),
        StructField("relative_humidity_pct", IntegerType(), False)
    ])
    df = spark.createDataFrame(data, schema)

    df_valid = df.filter(
        (F.col("temperature_celsius") >= -25.0) & (F.col("temperature_celsius") <= 50.0) &
        (F.col("relative_humidity_pct") >= 0) & (F.col("relative_humidity_pct") <= 100)
    )

    assert df_valid.count() == 1
    assert df_valid.first()["temperature_celsius"] == 15.0
