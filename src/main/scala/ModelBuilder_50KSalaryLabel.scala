import java.lang.System
import org.apache.log4j._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{StringIndexer,VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression,LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics}
import org.apache.spark.ml.linalg.{Vectors,DenseVector}
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.{Pipeline, PipelineModel}

 import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

//https://www.cetic.be/IMG/pdf/mlwithspark.pdf

object ModelBuilder_50KSalaryLabel {
  def main (args: Array[String]){
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
                .builder()
                .appName("Appls-Model Builder 50K Salary Label Logistic Regression")
                .config("spark.master","local")
                .getOrCreate()
  
    val mySchema = StructType(Array(
                                    StructField("age",DoubleType,true),
                                    StructField("workclass" ,StringType,true),
                                    StructField("fnlwgt", DoubleType,true),
                                    StructField("education" ,StringType,true),
                                    StructField("education_num", DoubleType,true),
                                    StructField("marital_status" ,StringType,true),
                                    StructField("occupation" ,StringType,true),
                                    StructField("relationship" ,StringType,true),
                                    StructField("race" ,StringType,true),
                                    StructField("sex" ,StringType,true),
                                    StructField("capital_gain", DoubleType,true),
                                    StructField("capital_loss", DoubleType,true),
                                    StructField("hours_per_week", DoubleType,true),
                                    StructField("native_country" ,StringType,true),
                                    StructField("income" ,StringType,true)))
  
    val fileName="/Users/keeratjohar2305/Downloads/adult.data" 
    case class CC1 (date1: java.sql.Date)
    var rawDF= spark.read
            .format("csv")
            .schema(mySchema)
            //.option("header","true")
            //.option("inferSchema","true")
            //.schema(Encoders.product[CC1).option("dateFormat","yyyyDDmm")
            .option("Delimiter",",")
            .load(fileName)
      
    println("raw dataframe")
    rawDF.show()
    
    //println("featrure column")
    val Stringfeatures= Array("workclass","education","marital_status","occupation","relationship","race","sex","native_country")
    
    val featureCols_VectNames= Stringfeatures.map(_ + "vector")
    //println("label column")
    val labelCols="income"
    
    
    // Vector Assemble to convert categorical feature column to numerical column 
    val featureIndexers= Stringfeatures.map(col=> new StringIndexer().setInputCol(col).setOutputCol(col + "Indexer"))
     
    val OHEEnconders= Stringfeatures.map(col=> new OneHotEncoderEstimator().setInputCols(Array(col + "Indexer")).setOutputCols(Array(col + "vector")))
  
    val numericfeatures = Array("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")
    val myassembler =  new VectorAssembler().setInputCols(numericfeatures ++ featureCols_VectNames).setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("income").setOutputCol("label")
   
    
    // Array of Stages
    val Stages = featureIndexers ++ OHEEnconders ++ Array(myassembler)  ++ Array(labelIndexer)
    Stages.foreach(println)
    
    //Set the pipelines
    val partialPipeline = new Pipeline().setStages(Stages)
    
    
    
    
    val Inputcols = rawDF.columns ++ Array("features" , "label")
    print(Inputcols.toList)
   
    val pipedDataDF = partialPipeline.fit(rawDF).transform(rawDF).select(Inputcols.head,Inputcols.tail:_*)
    pipedDataDF.show(false) 
    
    
    val Array(trainingData, testData) = pipedDataDF.randomSplit(Array(0.9, 0.1),seed=999999999999999999L)
    println("Print traning data")
    trainingData.show()
    println("Print test data")
    testData.show()
  
    var lr = new LogisticRegression()
    var lrModel= lr.fit(trainingData)
    
    // Summary Data
    val trainingSummary = lrModel.binarySummary
   
    // 3.1 Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))
  
    // 3.2 Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = trainingSummary.roc
    roc.show()
    println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")
    
    
    // Lets do prodiction
    val predictions = lrModel.transform(testData) 
    predictions.select("label","rawPrediction","probability","prediction").show(false)

    // Lets start calculatring confusion matrix
    val lp = predictions.select( "label", "prediction").createOrReplaceTempView("prediction_view")
    var counttotal = spark.sql("select count(1) from prediction_view").collect()(0).mkString
    val correct = spark.sql("select count(1) from prediction_view where label == prediction").collect()(0).mkString
    val wrong = spark.sql("select count(1) from prediction_view where label != prediction").collect()(0).mkString
    
    val truep = spark.sql("select count(1) from prediction_view where label == prediction and prediction = 1.0 and label=1.0").collect()(0).mkString
    val falseP = spark.sql("select count(1) from prediction_view where label != prediction and prediction = 1.0 and label=0.0").collect()(0).mkString
    val falseN = spark.sql("select count(1) from prediction_view where label == prediction and prediction = 0.0 and label=0.0").collect()(0).mkString
    val trueN = spark.sql("select count(1) from prediction_view where label != prediction and prediction = 0.0 and label=1.0").collect()(0).mkString
    
    print ("\ncounttotal:", counttotal)
    print("\ncorrect:",correct)
    print("\nwrong:",wrong)
    
    print("\ntruep:",truep)
    print("\nfalseP:",falseP)
    print("\nfalseN:",falseN)
    print("\ntrueN:",trueN)
    
    
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble
    
    print("ratioWrong =", ratioWrong)
    print("ratioCorrect =", ratioCorrect)
    
   
    // There are two of calculating precision - recall (PR) and reciever operating characterstics
    //Using rdd
    val  predictionAndLabels =predictions.select("rawPrediction", "label").rdd.map(x=>  (x(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector](1) , x(1).asInstanceOf[Double]))
    //predictionAndLabels.collect().foreach(println)
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println("area under the precision-recall curve (areaUnderPR): " + metrics.areaUnderPR)
    println("area under the receiver operating characteristic (areaUnderROC) curve : " + metrics.areaUnderROC)
  
    
    // Using DataFrame itself
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
    val accuracy = evaluator.evaluate(predictions)
    print(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderROC): $accuracy")
    
    // Using DataFrame itself
    val evaluator1 = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
    val accuracy1 = evaluator.evaluate(predictions)
    print(s"Accuracy  Per BinaryClassificationEvaluator (areaUnderPR): $accuracy1")
    
   /*
     *  Calculating Confussion matrix
     * 
   */

 
  
    //var lrTuning = new LogisticRegression() //.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) //.setMaxIter(value)
   val paramGrid = new ParamGridBuilder()
                  .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
                  .addGrid(lr.regParam, Array(0.01, 0.5, 2.0))
                  .addGrid(lr.maxIter,Array(1, 5, 10))
                  .build()
    
    println(lr.explainParams())
    val crossVal = new CrossValidator().setNumFolds(5).setEstimator(lr).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
    val CrossValModel = crossVal.fit(trainingData)//.bestModel
    val crossbestModel= CrossValModel.bestModel
  
    val prediction_with_CV = CrossValModel.transform(testData)
    val accuracy_tunedModelROC = evaluator.evaluate(prediction_with_CV)
    print(s"Accuracy  Per BinaryClassificationEvaluator (areaUnderPR): accuracy_tunedModel")
     
    val accuracy_tunedModelPR = evaluator1.evaluate(prediction_with_CV)
    print(s"Accuracy  Per BinaryClassificationEvaluator (areaUnderPR): accuracy_tunedModelPR")
   
    //crossbestModel.transform(dataset)
    
    //print('Model Intercept: ', cvModel.bestModel.intercept)
    //2) we can check the coefficients and intercept also
    //println(s"Coefficients: ${CrossValModel.bestModel} Intercept: ${CrossValModel.bestModel)}")
    prediction_with_CV.select("label", "prediction", "probability", "age", "occupation").show()
      
    
  }
}