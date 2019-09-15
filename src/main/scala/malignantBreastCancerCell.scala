import java.lang.System
import org.apache.log4j._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.{LogisticRegressionModel,LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{DenseVector}
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.split

/*
1\. id - ID numberß
2\. diagnosis - The diagnosis of breast tissues (M = malignant, B = benign)
3\. radius_mean-	mean of distances from center to points on the perimeter
4\. texture_mean-	standard deviation of gray-scale values
5\. perimeter_mean-	mean size of the core tumor

6\. area_mean
7\. smoothness_mean-	mean of local variation in radius lengthså
8\. compactness_mean-	mean of perimeter^2 / area - 1.0
9\. concavity_mean-	mean of severity of concave portions of the contour
10\. concave points_mean-	mean for number of concave portions of the contour
11\. symmetry_mean
12\. fractal_dimension_mean-	mean for "coastline approximation" - 1
13\. radius_se-	standard error for the mean of distances from center to points on the perimeter
14\. texture_se-	standard error for standard deviation of gray-scale values
15\. perimeter_se
16\. area_se
17\. smoothness_se-	standard error for local variation in radius lengths
18\. compactness_se-	standard error for perimeter^2 / area - 1.0
19\. concavity_se-	standard error for severity of concave portions of the contour
20\. concave points_se-	standard error for number of concave portions of the contour
21\. symmetry_se
22\. fractal_dimension_se-	standard error for "coastline approximation" - 1
23\. radius_worst-	"worst" or largest mean value for mean of distances from center to points on the perimeter
24\. texture_worst-	"worst" or largest mean value for standard deviation of gray-scale values
25\. perimeter_worst
27\. area_worst
28\. smoothness_worst-	"worst" or largest mean value for local variation in radius lengths
29\. compactness_worst-	"worst" or largest mean value for perimeter^2 / area - 1.0
30-\. concavity_worst-	"worst" or largest mean value for severity of concave portions of the contour
31\. concave points_worst-	"worst" or largest mean value for number of concave portions of the contour
32\. symmetry_worst
33\. fractal_dimension_worst-	"worst" or largest mean value for "coastline approximation" - 1
   */

/* DATA SET 1
 * 
 * https://mapr.com/blog/predicting-breast-cancer-using-apache-spark-machine-learning-logistic-regression/
 * 1\. Sample code number: id number
2\. Clump Thickness: 1 - 10
3\. Uniformity of Cell Size: 1 - 10
4\. Uniformity of Cell Shape: 1 - 10
5\. Marginal Adhesion: 1 - 10
6\. Single Epithelial Cell Size: 1 - 10
7\. Bare Nuclei: 1 - 10
8\. Bland Chromatin: 1 - 10
9\. Normal Nucleoli: 1 - 10
10\. Mitoses: 1 - 10
11\. Class: (2 for benign, 4 for malignant)
 */
import ML_scalaAdvanceMethods._
object malignantBreastCancerCell {

def main(args: Array[String]){ 
  Logger.getLogger("org").setLevel(Level.ERROR)
  var spark = SparkSession
              .builder()
              .appName("Cancers cells are melignant")
              .config("spark.master","local")
              .getOrCreate()

   /*
    * Two Example are processing in these example , i will create two model
    * 
    */
              
   //Example 1           
   val fileName= "/Users/keeratjohar2305/Downloads/breast-cancer-wisconsin-data.csv"
   var rawDF = spark.read.format("csv")
               .option("header","true")
               .option("inferSchema","true")
               .option("delimiter",",")
               .load(fileName)
               
  rawDF.show()
  rawDF.printSchema
  rawDF.columns
 // val labelCol =
  //val featureCol = 
  
    
    //Example 1
  val fileName2= "/Users/keeratjohar2305/Downloads/Cancer_Observation.csv"
  var my_schema = StructType(Array(
           StructField("Id_Number", IntegerType,true),
           StructField("Clump_Thickness", IntegerType,true),
           StructField("Uniformity_of_Cell_Size",  IntegerType,true),
           StructField("Uniformity_of_Cell_Shape",  IntegerType,true),
           StructField("Marginal_Adhesion",  IntegerType,true),
           StructField("Single_Epithelial_Cell_Size",  IntegerType,true),
           StructField("Bare_Nuclei", IntegerType,true),
           StructField("Bland_Chromatin",  IntegerType,true),
           StructField("Normal_Nucleoli",  IntegerType,true),
           StructField("Mitoses", IntegerType,true),
           StructField("class", IntegerType,true)
           )
           )
  val rawDF2 = spark.read.format("csv")
              //.option("header","true")
              //.option("inferSchema","true")
              .option("delimiter",",")
              .schema(my_schema)
              .load(fileName2)
  val labelCol2 = "class"
  
  // Access Raw data and prepare a dataFrame
  val featureCol2 = Array("Clump_Thickness","Uniformity_of_Cell_Size","Uniformity_of_Cell_Shape","Marginal_Adhesion","Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses")
  print("raw Df printing")
  rawDF2.show()
  rawDF2.printSchema
  rawDF2.columns
  summaryCustomized(rawDF2).show()
  
  
  
  print("assembler to convert feature coloumn to feature vectors")
  val assembler = new VectorAssembler().setInputCols(featureCol2).setOutputCol("features").setHandleInvalid("skip")
  val df2 = assembler.transform(rawDF2)
  df2.show(false)
  
  
  print("label indexer to Add label columns")
  val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("label").setHandleInvalid("skip")
  val df3 = labelIndexer.fit(df2).transform(df2)
  df3.drop()
  df3.show(false)
  df3.select("class").distinct().show()
  
  //Split the Raw data into Training data and test data 
  print("Split Data into training data and test data (70:30 ratio)")
  val splitSeed = 5043
  val Array(trainingData, testData) = df3.randomSplit(Array(0.9, 0.1),seed = 12345)
  println("Print traning data")
  trainingData.show()
  println("Print test data")
  testData.show()
  
  
  // Create a LogisticRegression instance. This instance is an Estimator.
  //and Also set parameters using setter methods
  print("create the mode with three paramters setMaxIter(10), setRegParam(0.3), setElasticNetParam(0.8), you can also save the model")
  val lr2 = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
  
  
  //Learn the model and fit the logistic regression estimator to training data
  val model2 = lr2.fit(trainingData)
  
  
  //3. Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
  /*   
   *   model is created ,let perform
   * 
   *     BEFORE PREDICTION  MODEL SUMMARY
   * 
   * 
   * 
   */
  
  //1) we can view the parameters Estimator used during fit() into training data to generate model or transformer
  println("Model Summary and exploration 1) Model fitted using parameters: " + model2.parent.extractParamMap)
  
  //2) we can check the coefficients and intercept also
  println(s"Model has Coefficients: ${model2.coefficients} And Intercept: ${model2.intercept}")
  
  val modelFile2 = "/Users/keeratjohar2305/Downloads/ML_MODEL/MAPR_CancerMalignacny_detectionModel" 
  //model2.write.overwrite().save(modelFile2)
  //val model2_loaded = LogisticRegressionModel.load(modelFile2)
  //val predictions = model2_loaded.transform(testData) 
  
  val trainingSummary = model2.binarySummary
 
  // 3.1 Obtain the objective per iteration.
   val objectiveHistory = trainingSummary.objectiveHistory
   println("Model objectiveHistory Details:")
   objectiveHistory.foreach(loss => println(loss))
  
   // 3.2 Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
  val roc = trainingSummary.roc
  roc.show()
  println(s"Model areaUnderROC: ${trainingSummary.areaUnderROC}")

  // Set the model threshold to maximize F-Measure
  //  val fMeasure = trainingSummary.fMeasureByThreshold
  //  val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
  //  val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
  //  .select("threshold").head().getDouble(0)
  //   model2.setThreshold(bestThreshold) 
  
  // xxx).   Make predictions on test data using the Transformer.transform() method on the test data
  // LogisticRegression.transform will only use the 'features' column.
    
  // Note that model2.transform() outputs three columns  'prediction', 'probability'   and 'rawPrediction'
  
  println
  println
  println("Lets make prediction on testing data")
  
  val predictions = model2.transform(testData) 
  predictions.select("label","rawPrediction","probability","prediction").show(false)
   
  
  println(""" 
 /*   Now Model is created, Lets do some Matrix calculations
 
  *
  *   FINDING ACCURACY
  * 
  *  A common metric used for logistic regression is area under the ROC(receiver operating characteristic) curve AUC (Area under C). 
    We can use the BinaryClasssificationEvaluator to obtain the AUC
    create an Evaluator for binary classification, which expects two input columns: rawPrediction and label.**

  * Evaluates predictions and returns a scalar metric areaUnderROC(larger is better).**
 */ """)
  
val evaluator = new BinaryClassificationEvaluator().
                              setLabelCol("label").
                              setRawPredictionCol("rawPrediction").
                              setMetricName("areaUnderROC")
val accuracy = evaluator.evaluate(predictions)
println(" Model accuracy is : " + accuracy)


  /*
   *     areaUnderPR   VS   areaUnderROC    
   *     
   *     METHOD 2
  *A Precision-Recall curve plots (precision, recall) points for different threshold values, while a receiver operating characteristic, or ROC, 
  curve plots (recall, false positive rate) points. The closer  the area Under ROC is to 1, the better the model is making predictions.**

  *use MLlib to evaluate, convert DF to RDD**
 */

val  predictionAndLabels =predictions.
                          select("rawPrediction", "label").
                          rdd.
                          map(x=>  (x(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector](1) , x(1).asInstanceOf[Double]))
     predictionAndLabels.collect().foreach(println)
     
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
println("area under the precision-recall curve: " + metrics.areaUnderPR)
println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)

println("""
          /* *******************************
           *  Calculating Confussion matrix
           * 
           ********************************/ """)

val lp = predictions.select( "label", "prediction").createOrReplaceTempView("prediction_view")
var counttotal = spark.sql("select count(1) from prediction_view").collect()(0).mkString
val correct = spark.sql("select count(1) from prediction_view where label == prediction").collect()(0).mkString
val wrong = spark.sql("select count(1) from prediction_view where label != prediction").collect()(0).mkString
val truep = spark.sql("select count(1) from prediction_view where label == prediction and prediction = 1.0 and label=1.0").collect()(0).mkString
val falseP = spark.sql("select count(1) from prediction_view where label != prediction and prediction = 1.0 and label=0.0").collect()(0).mkString
val falseN = spark.sql("select count(1) from prediction_view where label == prediction and prediction = 0.0 and label=0.0").collect()(0).mkString
val trueN = spark.sql("select count(1) from prediction_view where label != prediction and prediction = 0.0 and label=1.0").collect()(0).mkString

val ratioWrong=wrong.toDouble/counttotal.toDouble
val ratioCorrect=correct.toDouble/counttotal.toDouble

println("counttotal: " +  counttotal)
println("correct: " + correct)
println("wrong: " + wrong)

println("truep: " + truep)
println("falseP: "+ falseP)
println("falseN: " + falseN)
println("trueN: " + trueN)

println("ratioWrong: " +  ratioWrong)
println("ratioCorrect: "+  ratioCorrect)


}
 
}