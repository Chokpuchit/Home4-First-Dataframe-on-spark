# Databricks notebook source
# MAGIC %md
# MAGIC # Install Spark + import lib + start spark session

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting data from Kaggle

# COMMAND ----------

# MAGIC %md
# MAGIC You may download data from Kaggle mannually or using auto pipeline like this one.

# COMMAND ----------

# MAGIC %pip install opendatasets --upgrade

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/datasets/claudiodavi/superhero-set/data","/Volumes/workspace/default/test/")

# COMMAND ----------

# MAGIC %sh ls ./

# COMMAND ----------

# MAGIC %sh unzip /dbfs/FileStore/mypath/superhero-set.zip

# COMMAND ----------

#important "file:/"

file_path = "/Volumes/workspace/default/test/superhero-set/heroes_information.csv"


df_hero_indi = spark.read.options(header="true",inferschema = "true").csv(file_path)

df_hero_indi.show()

# COMMAND ----------

display(df_hero_indi)

# COMMAND ----------

file_path2 = "/Volumes/workspace/default/test/superhero-set/super_hero_powers.csv"

df_hero_power = spark.read.options(header="true",inferschema = "true").csv(file_path2)

#df_hero_power.show()
display(df_hero_power)

# COMMAND ----------

# MAGIC %md
# MAGIC # Querying + Stat test

# COMMAND ----------

df_hero_indi.count()

# COMMAND ----------

display(df_hero_indi.select("Race"))

# COMMAND ----------

df_hero_indi.select("Race").distinct().show()

# COMMAND ----------

display(df_hero_indi.filter(col("Race")=="Cyborg"))

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct

df_hero_indi.agg(countDistinct(col("Race"))).show()

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct

display(df_hero_indi.agg(*(countDistinct(col(c)).alias(c) for c in df_hero_indi.columns)))

# COMMAND ----------

from pyspark.sql import functions as F

df_hero_indi.agg(F.min(col("Weight"))\
              ,F.max(col("Weight"))\
              ,F.avg(col("Weight"))\
              ,F.sum(col("Weight"))\
              ,F.stddev(col("Weight")))\
              .show()

# COMMAND ----------

display(
df_hero_indi.groupBy(col("Race")).agg(F.min(col("Weight"))\
              ,F.max(col("Weight"))\
              ,F.avg(col("Weight"))\
              ,F.sum(col("Weight"))\
              ,F.stddev(col("Weight")))\
)
              

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finding Median
# MAGIC http://infolab.stanford.edu/~datar/courses/cs361a/papers/quantiles.pdf

# COMMAND ----------

df_hero_indi.approxQuantile("weight", [0.5], 0.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## "Null" **checking**

# COMMAND ----------

df_hero_indi.filter(col("Weight").isNull()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Group by

# COMMAND ----------

df_hero_indi.groupBy("Race").count().show()

# COMMAND ----------

from pyspark.sql import functions as F

df_hero_indi.groupby(col("Gender")).agg(F.min(col("Weight"))\
              ,F.max(col("Weight"))\
              ,F.avg(col("Weight"))\
              ,F.sum(col("Weight"))\
              ,F.stddev(col("Weight")))\
              .show()

# COMMAND ----------

df_hero_weight = df_hero_indi.filter(col("Weight")!=-99).select("weight")
df_hero_weight.show()

# COMMAND ----------

df_hero_weight.agg(F.min(col("Weight"))\
              ,F.max(col("Weight"))\
              ,F.avg(col("Weight"))\
              ,F.sum(col("Weight"))\
              ,F.stddev(col("Weight")))\
              .show()

# COMMAND ----------

display(df_hero_weight)


# COMMAND ----------

import pandas as pd

def compute_histogram(pdf_iter):
    for pdf in pdf_iter:
        hist, bin_edges = pd.cut(pdf['Weight'], bins=11, retbins=True)
        hist_counts = hist.value_counts().sort_index()
        yield pd.DataFrame({'histogram': hist_counts.values})

weight_histogram = df_hero_weight.mapInPandas(compute_histogram, schema="histogram array<double>")

display(weight_histogram)

# COMMAND ----------

# MAGIC %md
# MAGIC (Using Pandas and plot for showing graph)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joinning (yes, same as join in SQL)

# COMMAND ----------

#Rename to match
df_power = df_hero_power.withColumnRenamed("hero_names","name")

df_power.show()

# COMMAND ----------

df_joined = df_hero_indi.join(df_power, on="name",how="left")
display(df_joined)

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ## New conditional column

# COMMAND ----------

# MAGIC %md
# MAGIC Due to WORM (write once read many) so normally we will not alter df, we would add with new column

# COMMAND ----------

from pyspark.sql.functions import col, expr, when

new_column = F.when(col("Race")=="-","null").otherwise(col("Race"))

df_test_nc = df_hero_indi.withColumn("clean_Race",new_column)
df_test_nc.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Apply same concept to clean null

# COMMAND ----------

from pyspark.sql.functions import col, expr, when

new_column = F.when(col("weight").isNull(),-99).otherwise(col("weight"))

df_test_nc = df_hero_indi.withColumn("clean_weight1",new_column)
df_test_nc.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF: User defined function(s)
# MAGIC Spark does not support direct calculation to each cell values so there is some reway to do calculation, in distribution mode.

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import udf

def lbs2kg(lbs):
    return lbs*0.4536

lbs2kg_udf = udf(lbs2kg, FloatType())
df_test = df_hero_indi.withColumn('weight_in_kg',lbs2kg_udf(df_hero_indi["weight"]))
df_test.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Binarizer

# COMMAND ----------

from pyspark.ml.feature import Binarizer

binarizer = Binarizer().setThreshold(112.25).setInputCol("Weight").setOutputCol("binarized_weight")
binarizedDataFrame = binarizer.transform(df_hero_indi)
binarizedDataFrame.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quatile / Percentile

# COMMAND ----------

bounds = {
    c: dict(
        zip(["q1", "q3"], df_hero_weight.approxQuantile(c, [0.25, 0.75], 0))
    )
    for c in df_hero_weight.columns
}

print(bounds)

# COMMAND ----------

# MAGIC %md
# MAGIC Using quatile to check outlier

# COMMAND ----------

for c in bounds:
    iqr = bounds[c]['q3'] - bounds[c]['q1']
    bounds[c]['lower'] = bounds[c]['q1'] - (iqr * 1.5)
    bounds[c]['upper'] = bounds[c]['q3'] + (iqr * 1.5)
print(bounds)

# COMMAND ----------

import pyspark.sql.functions as f
df_hero_weight.select(
    "*",
    *[
        f.when(
            f.col(c).between(bounds[c]['lower'], bounds[c]['upper']),
            0
        ).otherwise(1).alias(c+"_out") 
        for c in df_hero_weight.columns
    ]
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Advanced solution for percentile / quatile

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrameStatFunctions as statFunc
import numpy as np

from pyspark.sql import Column
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import ast


class Discretize:
    @staticmethod
    def threshold_index(col_val, threshold: Column, threshold_str: bool = False):
        if threshold_str:
            # convert list that represent as string to normal list
            threshold = ast.literal_eval(threshold)
        for i, val_i in enumerate(threshold):
            current = threshold[i]
            if i > 0:
                previous = threshold[i - 1]
                if col_val > previous and col_val <= current:
                    result = int(i)
                elif col_val > previous and col_val > current:
                    # for threshold cutoff (extend positive limit bound)
                    result = int(i) + 1
            if i == 0 and col_val <= current:
                result = int(i)
        return result

    @staticmethod
    def human_score(x, y):
        return (int(y) - int(x))

    @staticmethod
    def indexer(df_in, columnname, x, output_name, invert: bool = True):

        threshold_index_udf = udf(Discretize.threshold_index, IntegerType())
        human_score_udf = udf(Discretize.human_score, IntegerType())

        index = list(np.linspace(1. / x, 1, x))
        pthvalue = statFunc(df_in).approxQuantile(columnname, index, 0.0)  # get list of cutoff nth //
        df_out = df_in.withColumn("pth", array([lit(df_in) for df_in in pthvalue]))
        df_out = df_out.withColumn('ranking', threshold_index_udf(columnname, "pth"))
        if invert:
            df_out = df_out.withColumn("maxpth", lit(x))
            df_out = df_out.withColumn(output_name, col("maxpth") - col("ranking")).drop("maxpth")
        else:
            df_out = df_out.withColumn(output_name, lit("ranking"))
        df_out = df_out.drop("pth")
        return df_out

# COMMAND ----------

output_data=Discretize.indexer(df_hero_weight,"Weight",100,"Percnetile_weight")
output_data.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Numerical to categorical

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

lookup = spark.createDataFrame(
    [(-100.0,0.000,"NA"),
     (0.001,50.00,"0-50 Lbs"),
     (50.00,100.00,"51-100 Lbs"),
     (100.00,200.00,"101-200 Lbs"),
     (200.00,300.00,"201-300 Lbs"),
     (300.00,400.00,"301-400 Lbs"),
     (400.00,500.00,"401-500 Lbs"),
     (500.00,600.00,"501-600 Lbs"),
     (600.00,1000.00,"600+ Lbs")],
    ("b","t","weight_grp"))
    
df_test_grp = df_hero_indi\
    .join(lookup,[F.col("weight")>=F.col("b"),F.col("weight") < F.col("t")],"leftouter")
  
df_test_grp.groupby("weight_grp").count().orderBy("weight_grp").show()

# COMMAND ----------

df_test_grp2 = df_test_nc\
    .join(lookup,[F.col("clean_weight1")>=F.col("b"),F.col("clean_weight1") < F.col("t")],"leftouter")
  
df_test_grp2.groupby("weight_grp").count().orderBy("weight_grp").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardization

# COMMAND ----------

from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="Weight", outputCol="scaled_weight",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(df_hero_indi)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(df_hero_indi)
scaledData.show()

# COMMAND ----------

# MAGIC %md
# MAGIC More reading: https://spark.apache.org/docs/1.4.1/ml-features.html

# COMMAND ----------

# MAGIC %md
# MAGIC #Lab

# COMMAND ----------

# MAGIC %md
# MAGIC Try to utilize spark as much as possible

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest data

# COMMAND ----------

# MAGIC %md
# MAGIC Data set: [here](https://www.kaggle.com/mashlyn/online-retail-ii-uci)

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci","/Volumes/workspace/default/test/")


# COMMAND ----------

# MAGIC %sh ls ./

# COMMAND ----------

file_path = "/Volumes/workspace/default/test/online-retail-ii-uci/online_retail_II.csv"


df_online_retail_II = spark.read.options(header="true",inferschema = "true").csv(file_path)

df_online_retail_II.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Description
# MAGIC
# MAGIC This Online Retail II data set contains all the transactions occurring for a UK-based and registered, 
# MAGIC non-store online retail between 01/12/2009 and 09/12/2011. The company mainly sells unique all-occasion gift-ware. 
# MAGIC Many customers of the company are wholesalers.
# MAGIC
# MAGIC Attribute Information:
# MAGIC
# MAGIC - InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
# MAGIC - StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
# MAGIC - Description: Product (item) name. Nominal.
# MAGIC - Quantity: The quantities of each product (item) per transaction. Numeric.
# MAGIC - InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
# MAGIC - UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
# MAGIC - CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
# MAGIC - Country: Country name. Nominal. The name of the country where a customer resides.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Explore the Data: Check NULL values, Check for outliers, and highlight

# COMMAND ----------

#code here
df_online_retail_II.select([count(when(isnull(c), c)).alias(c) for c in df_online_retail_II.columns]).show()

# COMMAND ----------

print("ตัวอย่างรายการที่ถูกยกเลิก (Quantity <= 0):")
df_online_retail_II.filter(col("Quantity") <= 0).show()


# COMMAND ----------

print("ตัวอย่างรายการที่ราคาเป็นศูนย์ (UnitPrice <= 0):")
df_online_retail_II.filter(col("Price") <= 0).show()

# COMMAND ----------

original_count = df_online_retail_II.count()
print(f"จำนวนข้อมูลเริ่มต้น: {original_count:,}")

# COMMAND ----------

cleaned_df = df_online_retail_II.dropna(subset=['Customer ID']) \
                                .filter(col('Quantity') > 0) \
                                .filter(col('Price') > 0)

# นับจำนวนข้อมูลหลังทำความสะอาด
cleaned_count = cleaned_df.count()
print(f"จำนวนข้อมูลหลังทำความสะอาด: {cleaned_count:,}")
print(f"ข้อมูลที่ถูกลบออกไป: {original_count - cleaned_count:,} แถว")

# แสดงผล DataFrame ที่สะอาดแล้ว
print("\n--- ข้อมูลพร้อมสำหรับการวิเคราะห์ ---")
cleaned_df.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## For all questions, assume the current date is 10/12/2011

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Find an average basket size of customer in each country in the year 2010
# MAGIC
# MAGIC #### Basket size = Total Sales Amount / Total Number of Invoices
# MAGIC
# MAGIC Hint: df.select(to_date(df.STRING_COLUMN).alias('new_date')).show()

# COMMAND ----------

df_2010 = cleaned_df.withColumn("TotalAmount", F.col("Quantity") * F.col("Price")) \
                    .withColumn("InvoiceDate", F.to_date(F.col("InvoiceDate"), "M/d/yyyy H:m")) \
                    .filter(F.year("InvoiceDate") == 2010)

# 3. & 4. จัดกลุ่ม, คำนวณ และหาขนาดตะกร้า
result_2010 = df_2010.groupBy("Country") \
    .agg(
        F.sum("TotalAmount").alias("TotalSales"),
        F.countDistinct("Invoice").alias("TotalInvoices")
    ) \
    .withColumn(
        "BasketSize", F.col("TotalSales") / F.col("TotalInvoices")
    )

# แสดงผลลัพธ์ โดยเรียงจากประเทศที่มีขนาดตะกร้าใหญ่ที่สุด
print("ขนาดตะกร้าเฉลี่ยของแต่ละประเทศในปี 2010:")
result_2010.orderBy(F.desc("BasketSize")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###  3. Does the basket size in each country change over time? Which country has the highest growth in terms of both sales amount and basket size in the past 6 months?

# COMMAND ----------


df_prepared = cleaned_df.withColumn("TotalAmount", F.col("Quantity") * F.col("Price")) \
                        .withColumn("InvoiceDate", F.to_date(F.col("InvoiceDate"), "M/d/yyyy H:m"))

# 2. สร้างฟังก์ชันสำหรับคำนวณ (ใช้ 'Invoice' ในการนับใบเสร็จ)
def calculate_metrics_for_period(df, start_date, end_date):
    """ฟังก์ชันสำหรับกรองข้อมูลตามช่วงเวลาและคำนวณ Metrics"""
    period_df = df.filter((F.col("InvoiceDate") >= start_date) & (F.col("InvoiceDate") <= end_date))
    
    metrics = period_df.groupBy("Country") \
        .agg(
            F.sum("TotalAmount").alias("TotalSales"),
            F.countDistinct("Invoice").alias("TotalInvoices") # <-- แก้ไขตรงนี้
        ) \
        .withColumn("BasketSize", F.col("TotalSales") / F.col("TotalInvoices"))
    return metrics

# 3. กำหนดช่วงเวลาและคำนวณ
# ช่วง 6 เดือนล่าสุด
current_start = "2011-06-10"
current_end = "2011-12-10"
current_metrics = calculate_metrics_for_period(df_prepared, current_start, current_end) \
                    .withColumnRenamed("TotalSales", "CurrentSales") \
                    .withColumnRenamed("BasketSize", "CurrentBasketSize")

# ช่วง 6 เดือนก่อนหน้า
previous_start = "2010-12-10"
previous_end = "2011-06-09"
previous_metrics = calculate_metrics_for_period(df_prepared, previous_start, previous_end) \
                    .withColumnRenamed("TotalSales", "PreviousSales") \
                    .withColumnRenamed("BasketSize", "PreviousBasketSize")

# 4. Join ข้อมูลและคำนวณอัตราการเติบโต
growth_df = previous_metrics.join(current_metrics, "Country", "inner")

final_growth_df = growth_df.withColumn(
    "SalesGrowth", (F.col("CurrentSales") - F.col("PreviousSales")) / F.col("PreviousSales")
).withColumn(
    "BasketSizeGrowth", (F.col("CurrentBasketSize") - F.col("PreviousBasketSize")) / F.col("PreviousBasketSize")
).filter(F.col("PreviousSales") > 0) # ป้องกันการหารด้วยศูนย์

# 5. แสดงผลลัพธ์
print("--- เรียงตามอัตราการเติบโตของยอดขาย (Sales Growth) ---")
final_growth_df.select("Country", "SalesGrowth", "BasketSizeGrowth") \
               .orderBy(F.desc("SalesGrowth")).show()

print("\n--- เรียงตามอัตราการเติบโตของขนาดตะกร้า (Basket Size Growth) ---")
final_growth_df.select("Country", "SalesGrowth", "BasketSizeGrowth") \
               .orderBy(F.desc("BasketSizeGrowth")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Monitor weekly sales and visit by country, Past 1 week, Past 2 weeks, Past 4 weeks, Year-to-date
# MAGIC #### Create a report that includes the following columns:
# MAGIC - Country
# MAGIC - Number of Customers in past 1 week
# MAGIC - Number of Customers in past 2 weeks
# MAGIC - Number of Customers in past 4 weeks
# MAGIC - Number of Customers accumulated since 01/01/2011
# MAGIC - Sales amount in past 1 week
# MAGIC - Sales amount in past 2 weeks
# MAGIC - Sales amount in past 4 weeks
# MAGIC - Sales amount since 01/01/2011
# MAGIC - Number of Invoices in past 1 week
# MAGIC - Number of Invoices in past 2 weeks
# MAGIC - Number of Invoices in past 4 weeks
# MAGIC - Number of Invoices since 01/01/2011

# COMMAND ----------



# ใช้ cleaned_df ที่สร้างจากขั้นตอนข้างบน
snapshot_date = F.to_date(F.lit("2011-12-10"))
date_1w_start = snapshot_date - F.expr("INTERVAL 6 DAYS")
date_2w_start = snapshot_date - F.expr("INTERVAL 13 DAYS")
date_4w_start = snapshot_date - F.expr("INTERVAL 27 DAYS")
date_ytd_start = F.to_date(F.lit("2011-01-01"))

df_prepared = cleaned_df.withColumn("TotalAmount", F.col("Quantity") * F.col("Price")) \
                        .withColumn("InvoiceDate", F.to_date(F.col("InvoiceDate"), "M/d/yyyy H:m"))

cond_1w = (F.col("InvoiceDate") >= date_1w_start) & (F.col("InvoiceDate") <= snapshot_date)
cond_2w = (F.col("InvoiceDate") >= date_2w_start) & (F.col("InvoiceDate") <= snapshot_date)
cond_4w = (F.col("InvoiceDate") >= date_4w_start) & (F.col("InvoiceDate") <= snapshot_date)
cond_ytd = (F.col("InvoiceDate") >= date_ytd_start) & (F.col("InvoiceDate") <= snapshot_date)

monitoring_report = df_prepared.groupBy("Country").agg(
    F.countDistinct(F.when(cond_1w, F.col("Customer ID"))).alias("Customers_1W"),
    F.countDistinct(F.when(cond_2w, F.col("Customer ID"))).alias("Customers_2W"),
    F.countDistinct(F.when(cond_4w, F.col("Customer ID"))).alias("Customers_4W"),
    F.countDistinct(F.when(cond_ytd, F.col("Customer ID"))).alias("Customers_YTD"),
    F.sum(F.when(cond_1w, F.col("TotalAmount"))).alias("Sales_1W"),
    F.sum(F.when(cond_2w, F.col("TotalAmount"))).alias("Sales_2W"),
    F.sum(F.when(cond_4w, F.col("TotalAmount"))).alias("Sales_4W"),
    F.sum(F.when(cond_ytd, F.col("TotalAmount"))).alias("Sales_YTD"),
    F.countDistinct(F.when(cond_1w, F.col("Invoice"))).alias("Invoices_1W"),
    F.countDistinct(F.when(cond_2w, F.col("Invoice"))).alias("Invoices_2W"),
    F.countDistinct(F.when(cond_4w, F.col("Invoice"))).alias("Invoices_4W"),
    F.countDistinct(F.when(cond_ytd, F.col("Invoice"))).alias("Invoices_YTD")
).na.fill(0)

final_report = monitoring_report.select(
    "Country",
    "Customers_1W", "Customers_2W", "Customers_4W", "Customers_YTD",
    "Sales_1W", "Sales_2W", "Sales_4W", "Sales_YTD",
    "Invoices_1W", "Invoices_2W", "Invoices_4W", "Invoices_YTD"
).orderBy(F.desc("Sales_YTD"))

print("รายงานสรุปยอดขายและลูกค้า (แก้ไขสมบูรณ์):")
final_report.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Find the average number of days since last visit of the customer in each country

# COMMAND ----------

# 1. กำหนดวันที่ปัจจุบัน (Snapshot Date)
snapshot_date = F.to_date(F.lit("2011-12-10"))

# 2. เตรียมข้อมูล (ใช้ cleaned_df ที่มีชื่อคอลัมน์ถูกต้อง)
df_prepared = cleaned_df.withColumn("InvoiceDate", F.to_date(F.col("InvoiceDate"), "M/d/yyyy H:m"))

# 3. หารวันซื้อล่าสุดและคำนวณ Recency ของลูกค้าแต่ละคน
#    เราจะgroupBy ทั้ง CustomerID และ Country เพื่อให้รู้ว่าลูกค้าคนนั้นอยู่ประเทศไหน
recency_per_customer = df_prepared.groupBy("Customer ID", "Country").agg(
    F.max("InvoiceDate").alias("LastPurchaseDate")
).withColumn(
    "Recency", F.datediff(snapshot_date, F.col("LastPurchaseDate"))
)

# 4. นำค่า Recency ของลูกค้าทุกคนมาหาค่าเฉลี่ยของแต่ละประเทศ
avg_recency_by_country = recency_per_customer.groupBy("Country").agg(
    F.avg("Recency").alias("AvgDaysSinceLastVisit")
)

# 5. แสดงผลลัพธ์ โดยเรียงจากประเทศที่ลูกค้ากลับมาซื้อเร็วที่สุด (ค่าน้อย)
print("จำนวนวันเฉลี่ยที่ลูกค้าแต่ละประเทศห่างหายไปจากการซื้อครั้งล่าสุด:")
avg_recency_by_country.orderBy("AvgDaysSinceLastVisit").show()