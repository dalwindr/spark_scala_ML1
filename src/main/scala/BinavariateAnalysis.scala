

import java.lang.System
import org.apache.log4j._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel,LogisticRegression,LogisticRegressionModel}
import org.apache.spark.mllib.linalg.{Vectors,DenseVector}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.linalg.Matrices

 


object BinavariateAnalysis {
def main(args: Array[String]){
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  val spark = SparkSession.builder().appName("Random Forest").config("spark.master","local").getOrCreate()
  var trainDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").option("inferSchema","true")
  .load("/Users/keeratjohar2305/Downloads/Dataset/Titanic_train.csv")
  var testDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").option("inferSchema","true")
  .load("/Users/keeratjohar2305/Downloads/Dataset/Titanic_test.csv").withColumn("Survived", lit(0))
 
       def variable_identification(df: org.apache.spark.sql.DataFrame ) ={
           val columns_cnt= df.columns.length
           val row_cnt= df.count()
          
           print(s"Shape(rows =$row_cnt and colummns = $columns_cnt) \n ")
           df.summary().show()
        }
           
        def EDA_univariate( df: org.apache.spark.sql.DataFrame)  = {
           println("Exploratory Data Analysis-Univariate")
           println("_________________________________________________________________________________________________________________________")
            val NumericCol = Array("Fare")
           
           trainDF.columns.foreach{x =>
             
             
             trainDF.groupBy(x).count.show()
             if ( NumericCol contains x ) {
                val df11 = df.select(x)
                // 1) finding mean 
                val fareObservations = df11.rdd.map{row => Vectors.dense(row.getDouble(0))}
                val summary_Fare:MultivariateStatisticalSummary = Statistics.colStats(fareObservations)                   
                println("Mean of Fare: "+summary_Fare.mean)
                
                // 2) finding mean  again
                val meanValue = df11.rdd.map(_.getDouble(0)).mean()
                println("Mean Value of Fare From RDD: "+meanValue)
                // 3.)  Median of the variable Fare
               val countOfFare = df11.count()
               val sortedFare_Rdd = df11.rdd.map(_.getDouble(0)).sortBy(fareVal => fareVal )
               val sortedFareRdd_WithIndex = sortedFare_Rdd.zipWithIndex()
               val median_Fare = if(countOfFare%2 ==1)
                                            sortedFareRdd_WithIndex.filter{case(fareVal:Double, index:Long) => index == (countOfFare-1)/2}.first._1
                                 else{
                                            val elementAtFirstIndex = sortedFareRdd_WithIndex
                                                                          .filter{case(fareVal:Double,index:Long) => index == (countOfFare/2)-1}.first._1
                                            val elementAtSecondIndex = sortedFareRdd_WithIndex
                                                                          .filter{case(fareVal:Double,index:Long)  => index == (countOfFare/2)}.first._1
                                            (elementAtFirstIndex+elementAtSecondIndex)/2.0
                                     }
                println("Median of Fare variable is: "+median_Fare) 
                
                   //4.) Mode of the variable Fare
                val fareDetails_WithCount = df11.groupBy("Fare").count()
                val maxOccurrence_CountsDf = fareDetails_WithCount.select(max("count")).alias("MaxCount")
                val maxOccurrence = maxOccurrence_CountsDf.first().getLong(0)
                val fares_AtMaxOccurrence = fareDetails_WithCount.filter(fareDetails_WithCount("count") === maxOccurrence)
                if(fares_AtMaxOccurrence.count() == 1)
                  println ("Mode of Fare variable is: "+fares_AtMaxOccurrence.first().getDouble(0))

                else {
                      val modeValues = fares_AtMaxOccurrence.collect().map{row =>row.getDouble(0)}
                      println("Fare variable has more 1 mode: ")
                      modeValues.foreach(println)
                     }
                      //Spread of the variable
                      println("Variance is: "+summary_Fare.variance)

                
                 //4. Univariate analysis for Categorical data
             
                
                }
                if ( "Pclass" contains x ) {
                      val class_Details_Df = df.select("Pclass")
                      val count_OfRows = class_Details_Df.count()
                      println("Count of Pclass rows: "+count_OfRows)
                      val classDetails_GroupedCount = class_Details_Df.groupBy("Pclass").count()
                      val classDetails_PercentageDist = classDetails_GroupedCount.withColumn("PercentDistribution",classDetails_GroupedCount("count")/count_OfRows)
                      classDetails_PercentageDist.show()
               
                
             }
           }
          
        }
        
        def EDA_Binvariate( df: org.apache.spark.sql.DataFrame)  = {
           println("Exploratory Data Analysis - Binvariate")
            println("_________________________________________________________________________________________________________________________")
           //1. Correlation and Covariance 
           println( "\n correlation Survived vs Fare = " + df.stat.corr("Survived","Fare") + 
           "\n correlation Survived vs Age = " + df.stat.corr("Survived","Age") + 
           "\n correlation Survived vs Pclass = " +  df.stat.corr("Survived","Pclass") +
            "\n correlation Age vs Fare = " +   df.stat.corr("Age","Fare") +
            "\n correlation Pclass vs Fare = " +   df.stat.corr("Pclass","Fare") +
           
           "\n Covariance Survived vs Fare = " + df.stat.cov("Survived","Fare") +
           "\n Covariance Survived vs Age = " + df.stat.cov("Survived","Age") +
           "\n Covariance Survived vs Age = " + df.stat.cov("Survived","Pclass") +           
           "\n Covariance Age vs Fare = " +  df.stat.cov("Age","Fare") + 
           "\n Covariance Pclass vs Fare = " +  df.stat.cov("Pclass","Fare") 
           )
          //2.  Creating two-way table between Pclass and Sex variables
           println("\nFrequency distribution of Pclass against variable Sex:")
           val  twoWayTable_PclassSex = df.stat.crosstab("Pclass", "Sex")
           twoWayTable_PclassSex.show()
          
           //3
           println("\nFrequency distribution of Sex variable against Embarked:")
           df.stat.crosstab("Sex","Embarked").show()

          val PclassSex_Array = twoWayTable_PclassSex.drop("Pclass_Sex")
                                .collect().map{row =>
                                    val female = row.getLong(0).toDouble;
                                    val male = row.getLong(1).toDouble; (female,male)}
          val femaleValues = PclassSex_Array.map{case(female, male) => female}
          val maleValues = PclassSex_Array.map{case(female, male) => male}
           
          //.3  Creating two-way table between Pclass and Sex variables
          val goodnessOfFitTestResult = Statistics.chiSqTest(
                        Matrices.dense(twoWayTable_PclassSex.count().toInt,twoWayTable_PclassSex.columns.length-1, femaleValues ++ maleValues ))
          println("Chi Square Test Value: "+goodnessOfFitTestResult)

           // Analysis between categorical and continuous variables
          df.groupBy("Pclass").agg(sum("Fare"), count("Fare"),max("Fare"), min("Fare"), stddev("Fare") ).show()
          df.show()   
          System.exit(1)
        }
      
       def CleanDataframe_imputation ( df: org.apache.spark.sql.DataFrame)  = {
         val average_fare = df.agg(avg("fare")).collect()(0).mkString
         val average_age= df.agg(avg(col("age"))).collect()(0).mkString
         val fill_null = Map("fare"-> average_fare, "age" -> average_age, "Embarked"->"S")
         val CleanDF = df.na.fill(fill_null)
         CleanDF 
       }
      
            
       def CategoricalFeatureVectorzing( CatalogicalFeatureList:Seq[String]) = {  
             
             //Array of String indexes of string features
             val stringIndexers = CatalogicalFeatureList.map { colName =>new StringIndexer().setInputCol(colName).setOutputCol(colName + "Indexed")}.toArray
             val OHEncoder = CatalogicalFeatureList.map { colName =>new OneHotEncoderEstimator().setInputCols(Array(colName + "Indexed")).setOutputCols(Array(colName + "Vector"))}.toArray
             //val partial_Stage= 
              stringIndexers ++ OHEncoder
             //partial_Stage
       }
       
       def FeatureAssembler( CatalogicalFeatureList:Seq[String] , numbericalFeatureList:Seq[String] ) = {
             val indexedfeaturesCatColNames = CatalogicalFeatureList.map(_ + "Vector")
             val allIndexedFeaturesColNames = numbericalFeatureList ++ indexedfeaturesCatColNames
       
             //feature Assembler that contains the dense sparse matrix of featured columnns
              val assembler = new VectorAssembler().setInputCols(Array(allIndexedFeaturesColNames: _*)).setOutputCol("features")
             Array(assembler)
       }
      
       def Confusion_matrixCalculation(pridictedDF: org.apache.spark.sql.DataFrame)= {
           val tatal = pridictedDF.count()
           val correct_prediction = pridictedDF.filter(col("label") === col("prediction")).count()
           val wrong_prediction = pridictedDF.filter(col("label") =!= col("prediction")).count()
           println(s"""\n tatal prediction =  $tatal \n correct_prediction = $correct_prediction \n wrong_prediction = $wrong_prediction  """)
           print(pridictedDF.dtypes.toList)
           val trueP = pridictedDF.filter(col("label") === col("prediction") && col("label") === 1.0 && col("prediction") === 1.0 ).count()
           val trueN = pridictedDF.filter(col("label") =!= col("prediction") && col("label") === 1.0 && col("prediction") === 0.0).count()
           
           val falseN = pridictedDF.filter(col("label") === col("prediction") && col("label") === 0.0 && col("prediction") === 0.0 ).count()
           val falseP = pridictedDF.filter(col("label") =!= col("prediction") && col("label") === 0.0 && col("prediction") === 1.0).count()
           
           println(s" \n trueP = $trueP \n trueN = $trueN \n falseN = $falseN \n  falseP = $falseP ")
           
           val ratioWrong=wrong_prediction.toDouble/tatal.toDouble
           val ratioCorrect=correct_prediction.toDouble/tatal.toDouble
              
           println("ratioWrong =", ratioWrong)
           println("ratioCorrect =", ratioCorrect)
       }
       
       def BinaryClassificationEvaluator_ROC (pridictedDF: org.apache.spark.sql.DataFrame) = {
         val evaluatorROC = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
            
         val accuracyROC = evaluatorROC.evaluate(pridictedDF)
         println(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderROC): $accuracyROC")
         println ( "lets train model")
         
       }
       
       def BinaryClassificationEvaluator_PR (pridictedDF: org.apache.spark.sql.DataFrame) = {
         val evaluatorPR = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
         val accuracyPR = evaluatorPR.evaluate(pridictedDF)
         println(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderROC): $accuracyPR")
           
       }
      
      def TuningWithCrossValidate_RF(trainDataFrame: org.apache.spark.sql.DataFrame,testDataFrame: org.apache.spark.sql.DataFrame) ={
       
        println("-------------------------------------------------------------- Tuning RF ------------------------------------------")
        val RandomForestTuning = new RandomForestClassifier() //.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) //.setMaxIter(value)
         
         val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
         val paramGrid = new ParamGridBuilder()
                  .addGrid(RandomForestTuning.maxBins, Array(25, 28, 31))
                  .addGrid(RandomForestTuning.maxDepth, Array(4, 6, 8))
                  .addGrid(RandomForestTuning.impurity,Array("entropy", "gini"))
                  .build()
    
        println(RandomForestTuning.explainParams())
        
        val crossVal = new CrossValidator().setNumFolds(5).setEstimator(RandomForestTuning).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
        val CrossValModel = crossVal.fit(trainDataFrame)//.bestModel
        
        //val crossbestModel= CrossValModel.bestModel
        val prediction_with_CV = CrossValModel.transform(testDataFrame)
       Confusion_matrixCalculation(prediction_with_CV)
       BinaryClassificationEvaluator_ROC(prediction_with_CV)
       BinaryClassificationEvaluator_PR(prediction_with_CV)
      }
     
       def TuningWithCrossValidate_LR(trainDataFrame: org.apache.spark.sql.DataFrame,testDataFrame: org.apache.spark.sql.DataFrame) ={
         
         println("-------------------------------------------------------------- Tuning LR ------------------------------------------")
         val lrTuning = new LogisticRegression() //.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) //.setMaxIter(value)
         
         val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
         val paramGrid = new ParamGridBuilder()
                  .addGrid(lrTuning.elasticNetParam, Array(0.0, 0.5, 1.0))
                  .addGrid(lrTuning.regParam, Array(0.01, 0.5, 2.0))
                  .addGrid(lrTuning.maxIter,Array(1, 5, 10))
                  .build()
    
        println(lrTuning.explainParams())
        
        val crossVal = new CrossValidator().setNumFolds(5).setEstimator(lrTuning).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
        val CrossValModel = crossVal.fit(trainDataFrame)//.bestModel
        
        //val crossbestModel= CrossValModel.bestModel
        val prediction_with_CV = CrossValModel.transform(testDataFrame)
       Confusion_matrixCalculation(prediction_with_CV)
       BinaryClassificationEvaluator_ROC(prediction_with_CV)
       BinaryClassificationEvaluator_PR(prediction_with_CV)
      }
      
      
 
       //Spliting Dataset
       val Array(traningDF,testingDF) = trainDF.randomSplit(Array(0.7,0.3),seed=9999)
                   
       variable_identification(traningDF)  //Step1 
       EDA_univariate(traningDF)   //step2
       EDA_Binvariate(traningDF) //step3 
       val cleanedTrainDF= CleanDataframe_imputation(traningDF) //step4 data cleanring and imputation
       val cleanedTestDF= CleanDataframe_imputation(testingDF)  //step4.1 data cleanring and imputation
       
       // Step5- Seperate out String feature column and numeric features
       val featuresCatColNames = Seq("Pclass", "Sex", "Embarked")
       val featuresNumColNames = Seq("Age", "SibSp", "Parch", "Fare")
      
       // pipelining the stages
       val stages = CategoricalFeatureVectorzing(featuresCatColNames) ++ 
                    FeatureAssembler(featuresCatColNames,featuresNumColNames) ++ 
                    Array(new StringIndexer().setInputCol("Survived").setOutputCol("label"))  
       
       
       // pipelinedStages              
       val pipelinedStages = new Pipeline().setStages(stages)
       
       
       // create piped train DF
       val pipedDF = pipelinedStages.fit(cleanedTrainDF).transform(cleanedTrainDF)
       println("pipedDF training DF")
       pipedDF.show()
       
       // create piped test DF
       val pipedtestDF = pipelinedStages.fit(cleanedTestDF).transform(cleanedTestDF)
       println("pipedtestDF testing DF")
       pipedtestDF.show()
       
       //create random forest model
       println("-------------------------------------------------------------- Genral RF ------------------------------------------")
       val rf_prediction = new RandomForestClassifier().fit(pipedDF).transform(pipedtestDF)
       rf_prediction.show()
       
       //Lets mearsure the accruarcy
       Confusion_matrixCalculation(rf_prediction)
       BinaryClassificationEvaluator_ROC(rf_prediction)
       BinaryClassificationEvaluator_PR(rf_prediction)
       
        //create random forest model
       val lr_prediction = new LogisticRegression().fit(pipedDF).transform(pipedtestDF)
       lr_prediction.show()
       
       //Lets mearsure the accruarcy
       println("-------------------------------------------------------------- Genra LR ------------------------------------------")
       Confusion_matrixCalculation(lr_prediction)
       BinaryClassificationEvaluator_ROC(lr_prediction)
       BinaryClassificationEvaluator_PR(lr_prediction)
       
       
      println("lets execute do further turning")
       
      TuningWithCrossValidate_RF(pipedDF,pipedtestDF)
      TuningWithCrossValidate_LR(pipedDF,pipedtestDF)
}
}