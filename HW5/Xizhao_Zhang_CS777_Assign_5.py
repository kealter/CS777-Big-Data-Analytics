from __future__ import print_function

import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer,IDF
from pyspark.ml.feature import StopWordsRemover, ChiSqSelector
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
import time

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: main_task1 <file> <file2> ", file=sys.stderr)
        exit(-1)
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.appName("Assignment-5").getOrCreate()

	# Use this code to read the data
    ##TRAIN
    # Use this code to reade the data
    corpus = sc.textFile(sys.argv[1], 1)
    keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])).map(lambda x: (x[0], int(x[0].startswith("AU")),x[1]))   
    # Spark DataFrame to be used wiht MLlib 
    df = spark.createDataFrame(keyAndText).toDF("id","label","text").cache()

    ##TEST
    #Use this code to read the data
    corpust = sc.textFile(sys.argv[2], 1)
    keyAndTextt = corpust.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])).map(lambda x: (x[0], int(x[0].startswith("AU")),x[1]))   
    # Spark DataFrame to be used wiht MLlib 
    test = spark.createDataFrame(keyAndTextt).toDF("id","label","text").cache()


    # Text preprocessin pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    
    start_vectorize_time = time.time()
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="features_c", vocabSize=5000,minDF=10, maxDF=5000)
    
    end_vectorize_time = time.time()
    vectorize_time = end_vectorize_time - start_vectorize_time
    print(vectorize_time)
    
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="features")
    pipeline = Pipeline(stages=[tokenizer,remover, countVectorizer,idf])

  # Fitting the Pipeline Model
    data_model = pipeline.fit(df)
    
    # Print the vaocabulary
    vocabulary = data_model.stages[2].vocabulary
    print(vocabulary[:10])

    ### Task 2
    ### Build your learning model using Logistic Regression

    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20)
    pipeline = Pipeline(stages=[tokenizer, remover, countVectorizer, idf, lr])

    #training
    start_train_time = time.time()

    model = pipeline.fit(df)

    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    #testing
    start_test_time = time.time()

    predictions = model.transform(test)

    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    #evaluating
    start_eval_time = time.time()

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    f1 = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)

    end_eval_time = time.time()
    

    print("\nPerformance Metrics: Logisitic Regression")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    confusion_matrix = predictions.groupBy("label").pivot("prediction", [0, 1]).count().na.fill(0).orderBy("label")
    confusion_matrix.show()

    eval_time = end_eval_time - start_eval_time
    total_time = train_time + test_time + eval_time
    print('The total time needed to train the model: {} secs\nEvaluate the model: {} secs\nTest the model: {} secs\nTotal Time: {} secs'.format(train_time,eval_time,test_time,total_time))


    ### Task 3
    ### Build your learning model using SVM
    
    svm = LinearSVC(labelCol="label", featuresCol="features", maxIter=20)
    pipeline = Pipeline(stages=[tokenizer, remover, countVectorizer, idf, svm])

    #training
    start_train_time = time.time()

    model = pipeline.fit(df)

    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    #testing
    start_test_time = time.time()

    predictions = model.transform(test)

    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    #evaluating
    start_eval_time = time.time()

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    f1 = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)

    end_eval_time = time.time()
    
    print("\nPerformance Metrics: SVM")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    confusion_matrix = predictions.groupBy("label").pivot("prediction", [0, 1]).count().na.fill(0).orderBy("label")
    confusion_matrix.show()
    
    eval_time = end_eval_time - start_eval_time
    total_time = train_time + test_time + eval_time
    print('The total time needed to train the model: {} secs\nEvaluate the model: {} secs\nTest the model: {} secs\nTotal Time: {} secs'.format(train_time,eval_time,test_time,total_time))

    ### Task 4
    ### Rebuild your learning models using 200 words instead of 5000

    chisq_selector = ChiSqSelector(numTopFeatures=200, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    
    
    svm = LinearSVC(labelCol="label", featuresCol="selectedFeatures", maxIter=20)
    lr = LogisticRegression(labelCol="label", featuresCol="selectedFeatures", maxIter=20)
    
    
    pipeline = Pipeline(stages=[tokenizer, remover, countVectorizer, idf, chisq_selector, lr])

    #training
    start_train_time = time.time()

    model = pipeline.fit(df)

    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    #testing
    start_test_time = time.time()

    predictions = model.transform(test)

    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    #evaluating
    start_eval_time = time.time()

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    f1 = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)

    end_eval_time = time.time()
    

    print("\nPerformance Metrics: Logisitic Regression")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    confusion_matrix = predictions.groupBy("label").pivot("prediction", [0, 1]).count().na.fill(0).orderBy("label")
    confusion_matrix.show()

    eval_time = end_eval_time - start_eval_time
    total_time = train_time + test_time + eval_time
    print('The total time needed to train the model: {} secs\nEvaluate the model: {} secs\nTest the model: {} secs\nTotal Time: {} secs'.format(train_time,eval_time,test_time,total_time))

    #SVM

    pipeline = Pipeline(stages=[tokenizer, remover, countVectorizer, idf, chisq_selector, svm])

    #training
    start_train_time = time.time()

    model = pipeline.fit(df)

    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    df.unpersist()

    #testing
    start_test_time = time.time()

    predictions = model.transform(test)

    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    test.unpersist()

    #evaluating
    start_eval_time = time.time()

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    f1 = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)

    end_eval_time = time.time()
    
    print("\nPerformance Metrics: SVM")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    confusion_matrix = predictions.groupBy("label").pivot("prediction", [0, 1]).count().na.fill(0).orderBy("label")
    confusion_matrix.show()
    
    eval_time = end_eval_time - start_eval_time
    total_time = train_time + test_time + eval_time
    print('The total time needed to train the model: {} secs\nEvaluate the model: {} secs\nTest the model: {} secs\nTotal Time: {} secs'.format(train_time,eval_time,test_time,total_time))

    sc.stop()

    
