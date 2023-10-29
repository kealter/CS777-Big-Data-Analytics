from __future__ import print_function

import os
import sys
import requests
from operator import add

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
import time




#Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
 
    except:
         return False

#Function - Cleaning
#For example, remove lines if they don’t have 16 values and 
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 0.1 miles, 
# fare amount and total amount are more than 0.1 dollars
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11]) and isfloat(p[15])):
            if(float(p[4])> 60 and float(p[5])>0 and float(p[11])> 1 
               and float(p[16])> 0 and float(p[11])< 600 ):
                return p

#Main
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="Assignment-3")
    sqlContext = SQLContext(sc)
    df= sqlContext.read.format('csv').options(header='false', inferSchema='true',  sep =",").load(sys.argv[1])
    rdd0 = df.rdd.map(tuple)
    rdd1 = rdd0.filter(correctRows)

    #Task1
    xy = rdd1.map(lambda row: (row[5], row[11]))

    start_time = time.time()
    n = xy.count()
    sum_x, sum_y, sum_xy, sum_x2 = xy.aggregate(
    (0, 0, 0, 0),
    lambda acc, xy: (acc[0] + xy[0], acc[1] + xy[1], acc[2] + xy[0] * xy[1], acc[3] + xy[0] ** 2),
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1], acc1[2] + acc2[2], acc1[3] + acc2[3])
    )

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_x2 * sum_y - sum_x * sum_xy) / (n * sum_x2 - sum_x ** 2)

    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b}")

    results_1 = sc.parallelize([("Slope (m)", m), ("Intercept (b)", b)])
    # Results_1 should have m and b parameters from the calculations
    results_1.coalesce(1).saveAsTextFile(sys.argv[2])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time for the task: {elapsed_time} seconds")
    
    sc.stop()

    