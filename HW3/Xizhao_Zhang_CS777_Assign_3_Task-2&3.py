from __future__ import print_function

import os
import sys
import requests
from operator import add
import time
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import *


import numpy as np


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
    spark = SparkSession.builder.getOrCreate()
    sqlContext = SQLContext(sc)
    df= sqlContext.read.format('csv').options(header='false', inferSchema='true',  sep =",").load(sys.argv[1])
    #print(df.columns)
    rdd = df.rdd.map(tuple).filter(correctRows)
    

    #Task 2
    df1 = spark.createDataFrame(rdd, df.schema)

    #addtion clean steps
    df1 = df1.filter(
        (F.col('_c5').isNotNull()) & 
        (F.col('_c11').isNotNull()) & 
        (F.col('_c4').isNotNull()) & 
        (F.col('_c15').isNotNull()) & 
        (F.col('_c16').isNotNull())
    )

    stats5 = df1.select(F.mean('_c5').alias('mean5'), F.stddev('_c5').alias('std5')).first()

    df1 = df1.filter(
        (F.col('_c5') >= stats5['mean5'] - 3 * stats5['std5']) & 
        (F.col('_c5') <= stats5['mean5'] + 3 * stats5['std5'])
    )



    df1.cache()

    xy = df1.select(df1["_c5"].alias("x"), df1["_c11"].alias("y"))

    def gradient(df, initial_m=0, initial_b=0, 
                 learning_rate=0.0001, num_iterations=80):
        start_time = time.time()
        m, b = initial_m, initial_b
        n = df.count()

        for i in range(num_iterations):
            # y-(mx+b)
            error = df.withColumn("error", F.col("y") - (m * F.col("x") + b))
            
            cost = error.select(F.sum(F.col("error")**2)).collect()[0][0]
            
            # gradients for m and b
            gradients = error.select(F.sum(F.col("x") * F.col("error")), F.sum(F.col("error"))).collect()[0]
            gradient_m = -2 / n * gradients[0]
            gradient_b = -2 / n * gradients[1]
            
            m -= learning_rate * gradient_m
            b -= learning_rate * gradient_b
            
            #learning_rate *= 1.05

            print(f"Iteration {i+1}, Cost: {cost}, m: {m}, b: {b}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The function took {elapsed_time} seconds")

        return cost, m, b

    cost, m, b = gradient(xy)
    # print the cost, intercept, and the slope for each iteration

    # Results_2 should have m and b parameters from the gradient Descent Calculations
    #results_2.coalesce(1).saveAsTextFile(sys.argv[3])




    #Task 3 
    df2 = df1.select(df1["_c4"].alias("x1"), df1["_c5"].alias("x2"), df1["_c11"].alias("x3"), 
                     df1["_c15"].alias("x4"), df1["_c16"].alias("y"))
    
    def gradient2(df, initial_m=np.zeros(4), initial_b=0, learning_rate=0.0001, num_iterations=70):
        start_time = time.time()

        m, b = initial_m, initial_b
        n = df.count()
        prev_cost = float('inf')

        for i in range(num_iterations):
            #y-(mx+b)
            error = df.withColumn("error", F.col("y") - (m[0] * F.col("x1") + m[1] * F.col("x2") + m[2] * F.col("x3") + m[3] * F.col("x4") + b))
            
            cost = error.select(F.sum(F.col("error")**2)).collect()[0][0]
            # gradients for m and b
            gradients = error.select(F.sum(F.col("x1") * F.col("error")),
                                     F.sum(F.col("x2") * F.col("error")),
                                     F.sum(F.col("x3") * F.col("error")),
                                     F.sum(F.col("x4") * F.col("error")),
                                     F.sum(F.col("error"))).collect()[0]

            gradient_m = -2.0 / n * np.array(gradients[:-1])
            gradient_b = -2.0 / n * gradients[-1]
      
            m -= learning_rate * gradient_m
            b -= learning_rate * gradient_b

            if cost < prev_cost:
                learning_rate *= 1.05
            else:
                learning_rate *= 0.5

            print(f"Iteration {i+1}, Cost: {cost}, m: {m}, b: {b}, Learning Rate: {learning_rate}")
            prev_cost = cost

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The function took {elapsed_time} seconds")

        return cost, m, b
        

    cost, m, b = gradient2(df2)


    # print the cost, intercept, the slopes (m1,m2,m3,m4), and learning rate for each iteration

    # Results_3 should have b, m1, m2, m3, and m4 parameters from the gradient Descent Calculations
    #results_3.coalesce(1).saveAsTextFile(sys.argv[4])
    

    sc.stop()

    