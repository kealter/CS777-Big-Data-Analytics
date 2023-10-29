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
# checking if the trip duration is more than a minute, trip distance is more than 0 miles, 
# fare amount and total amount are more than 0 dollars
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[4])> 60 and float(p[5])>0 and float(p[11])> 0 and float(p[16])> 0):
                return p

#Main
if __name__ == "__main__":
    if len(sys.argv) <4 :
        print("Usage: main_task1 <file> <output> <output2>", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="Assignment-1")
    sqlContext = SQLContext(sc)
    df= sqlContext.read.format('csv').options(header='false', inferSchema='true',  sep =",").load(sys.argv[1])
    rdd0 = df.rdd.map(tuple)
    rdd1 = rdd0.filter(correctRows)

    

    #Task 1
    x = rdd1.map(lambda x: (x[0], x[1]))
    count = x.distinct().countByKey()
    top10_1 = sorted(count.items(), key=lambda x: -x[1])[:10]

    results_1 = sc.parallelize(top10_1)
    results_1.coalesce(1).saveAsTextFile(sys.argv[2])


    #Task 2
    # create calculated field for earned money per minute and a count field
    def calculated(rdd):
        new_field =  rdd[16]/(rdd[4]/60)
        return rdd + (new_field,) + (1,)
    newrdd = rdd1.map(calculated)
    
    sum = newrdd.map(lambda x: (x[1], (x[17], x[18]))).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    average = sum.map(lambda x: (x[0], x[1][0] / x[1][1]))
    top10_2 = average.top(10, key=lambda x: x[1])

    results_2 = sc.parallelize(top10_2)
    #savings output to argument
    results_2.coalesce(1).saveAsTextFile(sys.argv[3])

    sc.stop()