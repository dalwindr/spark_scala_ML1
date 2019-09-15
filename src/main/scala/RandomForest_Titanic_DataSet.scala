import java.lang.System
import org.apache.log4j._

import org.apache.spark.SparkConf

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel,LogisticRegression,LogisticRegressionModel}
import org.apache.spark.mllib.linalg.{Vectors,DenseVector}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}


 
import ML_scalaAdvanceMethods._

object RandomForest_Titanic_DataSet {
def main(args: Array[String]){
  Logger.getLogger("org").setLevel(Level.FATAL)
  
  val spark = SparkSession.builder().appName("Random Forest").config("spark.master","local").getOrCreate()
  
  var trainDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").option("inferSchema","true").
  load("/Users/keeratjohar2305/Downloads/Dataset/Titanic_train.csv")
  
  var testDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").option("inferSchema","true")
  .load("/Users/keeratjohar2305/Downloads/Dataset/Titanic_test.csv").withColumn("Survived", lit(0))
 
 
           
        def EDA_univariate( df: org.apache.spark.sql.DataFrame)  = {
           
            val NumericCol = Array("Fare")
           
            trainDF.columns.intersect(NumericCol).foreach{x =>
             println("Univeriate Exploratory Data Analysis-Univariate for Column : " +x + "\n")
             
             //trainDF.groupBy(x).count.show()
               // val df11 = df.select(x)
//                // 1) finding mean 
//                val fareObservations = df11.rdd.map{row => Vectors.dense(row.getInt(0)* 1.0)}
//                val summary_Fare:MultivariateStatisticalSummary = Statistics.colStats(fareObservations)                   
//                println("Mean of Fare: "+summary_Fare.mean)
//                println("Variance is: "+summary_Fare.variance)
//                
//                // 2) finding mean  again
//                val meanValue = df11.rdd.map(_.getInt(0)).mean()
//                println("Mean Value of Fare From RDD: "+meanValue)
//                
//                
     // 3.  Median of the variable Fare
//               val countOfFare = df11.count()
//               val sortedFare_Rdd = df11.rdd.map(_.getInt(0)).sortBy(fareVal => fareVal )
//               val sortedFareRdd_WithIndex = sortedFare_Rdd.zipWithIndex()
//               val median_Fare = if(countOfFare%2 ==1)
//                                            sortedFareRdd_WithIndex.filter{case(fareVal:Int, index:Long) => index == (countOfFare-1)/2}.first._1
//                                 else{
//                                            val elementAtFirstIndex = sortedFareRdd_WithIndex
//                                                                          .filter{case(fareVal:Int,index:Long) => index == (countOfFare/2)-1}.first._1
//                                            val elementAtSecondIndex = sortedFareRdd_WithIndex
//                                                                          .filter{case(fareVal:Int,index:Long)  => index == (countOfFare/2)}.first._1
//                                            (elementAtFirstIndex+elementAtSecondIndex)/2.0
//                                     }
 //    println("Median of Fare variable is: "+median_Fare) 
                
//           // Mode of the variable Fare
//        val fareDetails_WithCount = df11.groupBy("Fare").count()
//        val maxOccurrence_CountsDf = fareDetails_WithCount.select(max("count")).alias("MaxCount")
//        val maxOccurrence = maxOccurrence_CountsDf.first().getLong(0)
//        val fares_AtMaxOccurrence = fareDetails_WithCount.filter(fareDetails_WithCount("count") === maxOccurrence)
//                if(fares_AtMaxOccurrence.count() == 1)
//                  println ("Mode of Fare variable is: "+fares_AtMaxOccurrence.first().getInt(0))
//
//                else {
//                      val modeValues = fares_AtMaxOccurrence.collect().map{row =>row.getInt(0)}
//                      println("Fare variable has more 1 mode: ")
//                      modeValues.foreach(println)
//                     }
    //Spread of the variable
    

                
//                 //4. Univariate analysis for Categorical data
//                if ( "Pclass" contains x ) {
//                      val class_Details_Df = df.select("Pclass")
//                      val count_OfRows = class_Details_Df.count()
//                      println("Count of Pclass rows: "+count_OfRows)
//                      val classDetails_GroupedCount = class_Details_Df.groupBy("Pclass").count()
//                      val classDetails_PercentageDist = classDetails_GroupedCount.withColumn("PercentDistribution",classDetails_GroupedCount("count")/count_OfRows)
//                      classDetails_PercentageDist.show()
//               
//                
//             }
                
                
           }
          
        }
        
        def EDA_Binvariate( df: org.apache.spark.sql.DataFrame)  = {
           println(s"""Exploratory Data Analysis - Binvariate
         
           df.stat.corr("Survived","Fare"):    ${df.stat.corr("Survived","Fare")}
           df.stat.corr("Survived","Age"):   ${df.stat.corr("Survived","Age")}
           df.stat.corr("Survived","Pclass"):  ${df.stat.corr("Survived","Pclass")}
          
           df.stat.cov("Survived","Fare"):   ${df.stat.cov("Survived","Fare")}
           df.stat.cov("Survived","Age"):  ${df.stat.cov("Survived","Age")}
           df.stat.cov("Survived","Pclass") ${df.stat.cov("Survived","Pclass")}
           """)
           df.show()   
           //System.exit(1)
        }
      
       def CleanDataframe_imputation ( df: org.apache.spark.sql.DataFrame)  = {
         val average_fare = df.agg(avg("fare")).collect()(0).mkString
         val average_age= df.agg(avg(col("age"))).collect()(0).mkString
         val fill_null = Map("fare"-> average_fare, "age" -> average_age, "Embarked"->"S")
         val CleanDF = df.na.fill(fill_null)
         CleanDF 
       }
      
            
//       def CategoricalFeatureVectorzing( CatalogicalFeatureList:Seq[String]) = {  
//             
//             //Array of String indexes of string features
//             val stringIndexers = CatalogicalFeatureList.map { colName =>new StringIndexer().setInputCol(colName).setOutputCol(colName + "Indexed")}.toArray
//             val OHEncoder = CatalogicalFeatureList.map { colName =>new OneHotEncoderEstimator().setInputCols(Array(colName + "Indexed")).setOutputCols(Array(colName + "Vector"))}.toArray
//             //val partial_Stage= 
//              stringIndexers ++ OHEncoder
//             //partial_Stage
//       }
//       
//       def FeatureAssembler( CatalogicalFeatureList:Seq[String] , numbericalFeatureList:Seq[String] ) = {
//             val indexedfeaturesCatColNames = CatalogicalFeatureList.map(_ + "Vector")
//             val allIndexedFeaturesColNames = numbericalFeatureList ++ indexedfeaturesCatColNames
//       
//             //feature Assembler that contains the dense sparse matrix of featured columnns
//              val assembler = new VectorAssembler().setInputCols(Array(allIndexedFeaturesColNames: _*)).setOutputCol("features")
//             Array(assembler)
//       }
      
//       def Confusion_matrixCalculation(pridictedDF: org.apache.spark.sql.DataFrame)= {
//           val tatal = pridictedDF.count()
//           val correct_prediction = pridictedDF.filter(col("label") === col("prediction")).count()
//           val wrong_prediction = pridictedDF.filter(col("label") =!= col("prediction")).count()
//           println(s"""\n tatal prediction =  $tatal \n correct_prediction = $correct_prediction \n wrong_prediction = $wrong_prediction  """)
//           print(pridictedDF.dtypes.toList)
//           val trueP = pridictedDF.filter(col("label") === col("prediction") && col("label") === 1.0 && col("prediction") === 1.0 ).count()
//           val trueN = pridictedDF.filter(col("label") =!= col("prediction") && col("label") === 1.0 && col("prediction") === 0.0).count()
//           
//           val falseN = pridictedDF.filter(col("label") === col("prediction") && col("label") === 0.0 && col("prediction") === 0.0 ).count()
//           val falseP = pridictedDF.filter(col("label") =!= col("prediction") && col("label") === 0.0 && col("prediction") === 1.0).count()
//           
//           println(s" \n trueP = $trueP \n trueN = $trueN \n falseN = $falseN \n  falseP = $falseP ")
//           
//           val ratioWrong=wrong_prediction.toDouble/tatal.toDouble
//           val ratioCorrect=correct_prediction.toDouble/tatal.toDouble
//              
//           println("ratioWrong =", ratioWrong)
//           println("ratioCorrect =", ratioCorrect)
//       }
       
//       def BinaryClassificationEvaluator_ROC (pridictedDF: org.apache.spark.sql.DataFrame) = {
//         val evaluatorROC = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
//            
//         val accuracyROC = evaluatorROC.evaluate(pridictedDF)
//         println(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderROC): $accuracyROC")
//         println ( "lets train model")
//         
//       }
//       
//       def BinaryClassificationEvaluator_PR (pridictedDF: org.apache.spark.sql.DataFrame) = {
//         val evaluatorPR = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
//         val accuracyPR = evaluatorPR.evaluate(pridictedDF)
//         println(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderROC): $accuracyPR")
//           
//       }
//      
      def TuningWithCrossValidate_RFC(trainDataFrame: org.apache.spark.sql.DataFrame,testDataFrame: org.apache.spark.sql.DataFrame) ={
       
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
       Confusion_matrixCalculation(prediction_with_CV,"Optimised Random Forest Classfier")
       BinaryClassificationEvaluator_ROC(prediction_with_CV,"Optimised Random Forest Classfier")
       BinaryClassificationEvaluator_PR(prediction_with_CV,"Optimised Random Forest Classfier")
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
       Confusion_matrixCalculation(prediction_with_CV,"Optimised Logistic Regression")
       BinaryClassificationEvaluator_ROC(prediction_with_CV,"Optimised Logistic Regression")
       BinaryClassificationEvaluator_PR(prediction_with_CV,"Optimised Logistic Regression")
      }
       
//      val nameTitleMap = Map("Mrs"->1,"Lady"-> 1, "Mme"->1, "the Countess"->1,
//                             "Master"-> 2, 
//                             "Miss"-> 3,"Drfemale"-> 3,"Mlle"-> 3,"Ms"-> 3,
//                             "Mr"->4,  "Don"-> 4, "Jonkheer"-> 4, "Sir"-> 4,  "Major"-> 4, "Rev"->4,"Drmale"-> 4,"Col"-> 4, "Capt"-> 4
//                              
//                           )
                           
      //val nameTitleMapBC = spark.sparkContext.broadcast(nameTitleMap)
        
      val readTitleFunc   = udf(( str: String )=>   
        Map("Mrs"->1,"Lady"-> 1, "Mme"->1, "the Countess"->1,
                             "Master"-> 2, 
                             "Miss"-> 3,"Drfemale"-> 3,"Mlle"-> 3,"Ms"-> 3,
                             "Mr"->4,  "Don"-> 4, "Jonkheer"-> 4, "Sir"-> 4,  "Major"-> 4, "Rev"->4,"Drmale"-> 4,"Col"-> 4, "Capt"-> 4
                           ).getOrElse(str, 5)
      
      )
      spark.udf.register("readTitleFunc",readTitleFunc)
      
      //val CabinData = Map( "A" -> 1 ,"B"-> 2 ,"C"-> 3,"D"-> 4,"E"->5,"F"-> 6,"G"->7,"U"-> 7,"" -> 0) 
      //val CabinDataBC = spark.sparkContext.broadcast(CabinData)
      //val CabintoDeckFunc   = udf(( str: String)=>   Map( "A" -> 1 ,"B"-> 2 ,"C"-> 3,"D"-> 4,"E"->5,"F"-> 6,"G"->7,"U"-> 7," " -> "0",""->0).getOrElse( (str(0).toString()), "0".toString() )+ str.drop(1) )
      //spark.udf.register("CabintoDeckFunc",CabintoDeckFunc)
      
      // Lets perform some Feature Engineering + Also Some Data Cleansing
      val FEngDF1 = trainDF.drop("PassengerId").
               //withColumn("Embarked",when(col("Embarked").isNull , lit("C")).otherwise(col("Embarked")) ).
               withColumn("firstName", split(col("name"),"\\," ).getItem(1)).
               withColumn("nameTitle", split(col("firstName"),"\\.").getItem(0)).
              // withColumnRenamed("name", "fullName").
               withColumn("nameTitle",when(trim(col("nameTitle")) === ("Dr"),concat(trim(col("nameTitle")) , trim(col("Sex"))) ).otherwise(trim(col("nameTitle")))).
               withColumn("nameTitle",readTitleFunc(trim(col("nameTitle")))).
               drop(col("firstName")).drop("name").
               withColumn("Sex",when(trim(col("Sex")) === ("male"),lit(1)).otherwise(lit(2))).
               withColumn("Embarked",when(trim(col("Embarked")) === ("C"),lit(1)).
                                     when(trim(col("Embarked")) === ("S"),lit(2)).
                                     otherwise(lit(3))).
               withColumn("FamilySize", col("SibSp") + col("Parch")).drop("SibSp").drop("Parch").drop("Cabin").
               withColumn("FarePerPerson",col("Fare")/ (col("FamilySize") + lit(1))).
               //withColumn("AgeClassFactor",col("Age")*col("Pclass")).
               drop("Ticket").drop("Cabin")
               
               // Ticket column dropped
               // passenger id column dropped
               // drop name column after extracting title
               //  sibling and parch columns removed but retained combined data
               //  drop Cabin
               
       val FEngDF =   missingValFilled(FEngDF1, "Age")
       FEngDF.printSchema()
               
    
//                             expr("""
//                                           case  when(trim(col("nameTitle")).equals("Dr")  then concat(col("nameTitle"),col("Sex")) else col("nameTitle") 
//                                           """)).  
                                            
  
                                           
       FEngDF.show(5,false)
       FEngDF.printSchema()
       summaryCustomized(FEngDF).show()
       
       // Feature Engieering
       
               
       //split(col("value"), "\\|").getItem(0)
       
      
           
                    
      
           println("*********** CATAGORICAL DATA ANALYSYS")
       
       val CatagoricalCol= Seq("Survived","Pclass","Sex","Embarked","nameTitle")
       univariateAnalysis(FEngDF, CatagoricalCol)
       
       println("\n*********** MISSING DATA ANALYSYS")
       val MissingDataCol= Seq("Age","Cabin")
       univariateAnalysis(FEngDF, MissingDataCol)
       EDA_Binvariate(FEngDF)
       
       //val BivariateColCombinationArray = Array( ("Cabin", "Pclass"),("Cabin", "Age"),("Cabin", "SibSp"),("Cabin", "SibSp"),("Cabin", "nameTitle"),(""
       
       //EDA_BivariateAnalysis
       
       //Spliting Dataset
       
       
       
       val Array(traningDF,testingDF) = FEngDF.randomSplit(Array(0.7,0.3),seed=9999)
                   
       dsShape(traningDF)  //Step1 
       println("Final check of null in data " + "\nGet the count after removing null= " + traningDF.na.drop().count() + "\nGet the dataset Count Before Appply removing Null: " + traningDF.count()  )
       //EDA_univariate(traningDF)   //step2
       //EDA_Binvariate(traningDF) //step3 
    
       //val cleanedTrainDF=missingValFilled(traningDF, "Age") //step4 data cleanring and imputation
       //val cleanedTestDF= CleanDataframe_imputation(testingDF)  //step4.1 data cleanring and imputation
       
       // Step5- Seperate out String feature column and numeric features
       val featuresCatColNames = Seq()
       val featuresNumColNames = traningDF.columns.drop(1).toSeq
      
       // pipelining the stages
       val stages = CategoricalFeatureVectorzing(featuresCatColNames) ++ 
                    FeatureAssembler(featuresCatColNames,featuresNumColNames) ++ 
                    Array(new StringIndexer().setInputCol("Survived").setOutputCol("label"))  
       
                 
      //Something missed ...I Did not considered featuresNumColNames
      // Ok doubt is clear   func( FeatureAssembler ) is processing both numerical and cataogiral features         
                
       // pipelinedStages              
       val pipelinedStages = new Pipeline().setStages(stages)
       
       
       // create piped train DF
       val pipedDF = pipelinedStages.fit(traningDF).transform(traningDF)
       
       println("pipedDF training DF")
       pipedDF.show()
       
       // create piped test DF
       val pipedtestDF = pipelinedStages.fit(testingDF).transform(testingDF)
       println("pipedtestDF testing DF")
       pipedtestDF.show()

    import org.apache.spark.ml.classification.NaiveBayes
        println("--------------------------------------------------------------Applying General Default NaiveBayes ------------------------------------------")
                // instantiate the base classifier
        println("Baddest for binary classfication")
    val nv = new NaiveBayes()
                    
       
    val nv_prediction = nv.fit(pipedDF).transform(pipedtestDF)
       nv_prediction.show()
       
     //Lets mearsure the accruarcy
     Confusion_matrixCalculation(nv_prediction,"General Default NaiveBayes")
     BinaryClassificationEvaluator_ROC(nv_prediction,"General Default NaiveBayes")
     BinaryClassificationEvaluator_PR(nv_prediction,"General Default NaiveBayes ")
     // instantiate the base classifier

       
       
       import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
        println("--------------------------------------------------------------Applying General One-vs-Rest classifier (a.k.a. One-vs-All) ------------------------------------------")
                // instantiate the base classifier
      
    val classifier = new LogisticRegression()
                            .setMaxIter(10)
                            .setTol(1E-6)
                            .setFitIntercept(true)

    val ovr = new OneVsRest().setClassifier(classifier)
       
    val ovs_prediction = ovr.fit(pipedDF).transform(pipedtestDF)
       ovs_prediction.show()
       
     //Lets mearsure the accruarcy
     Confusion_matrixCalculation(ovs_prediction,"General One-vs-Rest classifier (a.k.a. One-vs-All)")
     BinaryClassificationEvaluator_ROC(ovs_prediction,"General One-vs-Rest classifier (a.k.a. One-vs-All)")
     BinaryClassificationEvaluator_PR(ovs_prediction,"General One-vs-Rest classifier (a.k.a. One-vs-All)")
     // instantiate the base classifier       
       
       import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}      
       println("--------------------------------------------------------------Applying General Gradient Booster tree Classfier ------------------------------------------")
       val gbt = new GBTClassifier().
                        setMaxIter(10)
       
       val gbt_prediction = gbt.fit(pipedDF).transform(pipedtestDF)
       gbt_prediction.show()
       
       //Lets mearsure the accruarcy
       Confusion_matrixCalculation(gbt_prediction,"General Gradient Booster tree Classfier")
       BinaryClassificationEvaluator_ROC(gbt_prediction,"General Gradient Booster tree Classfier")
       BinaryClassificationEvaluator_PR(gbt_prediction,"General Gradient Booster tree Classfier")
            
             

        //create Decion  tree model
        import org.apache.spark.ml.classification.DecisionTreeClassifier
       println("--------------------------------------------------------------Applying General Decision tree Classfier ------------------------------------------")
       val dt_prediction = new DecisionTreeClassifier().fit(pipedDF).transform(pipedtestDF)
       dt_prediction.show()
       
       //Lets mearsure the accruarcy
       Confusion_matrixCalculation(dt_prediction,"General Decision tree Classfier")
       BinaryClassificationEvaluator_ROC(dt_prediction,"General Decision tree Classfier")
       BinaryClassificationEvaluator_PR(dt_prediction,"General Decision tree Classfier")
            
       
       
       
       //create random forest model
       println("--------------------------------------------------------------Applying General RF ------------------------------------------")
       val rf_prediction = new RandomForestClassifier().fit(pipedDF).transform(pipedtestDF)
       rf_prediction.show()
       
       //Lets mearsure the accruarcy
       Confusion_matrixCalculation(rf_prediction,"General Random Forest Classfier")
       BinaryClassificationEvaluator_ROC(rf_prediction,"General Random Forest Classfier")
       BinaryClassificationEvaluator_PR(rf_prediction,"General Random Forest Classfier")
       
 
       
       
       println("--------------------------------------------------------------Appying General LR ------------------------------------------")
       //create random forest model
       val lr_prediction = new LogisticRegression().fit(pipedDF).transform(pipedtestDF)
       lr_prediction.show()
       
       //Lets mearsure the accruarcy
       Confusion_matrixCalculation(lr_prediction,"General Logistic Regression")
       BinaryClassificationEvaluator_ROC(lr_prediction,"General Logistic Regression")
       BinaryClassificationEvaluator_PR(lr_prediction,"General Logistic Regression")
       
       
       
       
      println("lets execute do further turning")
       
      TuningWithCrossValidate_RFC(pipedDF,pipedtestDF)
      TuningWithCrossValidate_LR(pipedDF,pipedtestDF)
      
      
      
}
}