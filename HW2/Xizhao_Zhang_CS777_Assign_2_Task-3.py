from __future__ import print_function

import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm


from operator import add
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount  <wikiCategoryLinks> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="Assignment-2")
    spark = SparkSession(sc)
    

    wikiCategoryLinks=sc.textFile(sys.argv[1])

    wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))
    # create df and count Category
    DF = spark.createDataFrame(wikiCats, ["Page", "Category"])
    counts = DF.groupBy("Page").agg(func.count("Category").alias("count"))

    ###Task3.1
    # register the DataFrame for sql
    counts.createOrReplaceTempView("table")

    medianDF = spark.sql("SELECT percentile_approx(count, 0.5) as median FROM table")
    median_value = medianDF.collect()[0]["median"]

    output = counts.agg(
        func.max("count").alias("Max"),
        func.mean("count").alias("Average"),
        func.stddev("count").alias("StdDev")
    ).collect()[0]
    

    print(f"Max: {output['Max']}, Average: {output['Average']}, StdDev: {output['StdDev']}, Median: {median_value}")
    
    ###Task3.2
    top10 = (DF.groupBy("Category")
                        .agg(func.count("Page").alias("Count"))
                        .sort("Count", ascending=False)).show(10)

    sc.stop()
  



    