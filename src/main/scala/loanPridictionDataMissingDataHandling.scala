/*
 *   This for filling missing Values in load prediction Data set
 * 
 */

//http://www.chioka.in/differences-between-roc-auc-and-pr-auc/
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}

import org.apache.log4j._

 
 //vectorizing
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
import org.apache.spark.ml.Pipeline

object loanPridictionData {
def main(args: Array[String]) {
        Logger.getLogger("org").setLevel(Level.ERROR)
        val spark = SparkSession
                    .builder()
                    .appName("Java Spark SQL basic example")
                    .config("spark.master", "local")
                    .getOrCreate()
        import spark.sqlContext.implicits._            
        
        // Prepare training data from a list of (label, features) tuples.
        
        var rawTrainDF = spark.read.format("csv").option("Delimiter", ",").option("header","true")
                    .option("inferSchema","true").load("/Users/keeratjohar2305/Downloads/Dataset/AVtrain_LoanPrediction.csv").
                    //drop("Loan_ID").
                    na.fill("Female", Seq("Gender")). // blindly  replaced null values
                    na.fill("Yes", Seq("Married")).   //blindly replaced null values
                    na.fill("2", Seq("Dependents")).  // median replaced null values
                    na.fill("No", Seq("Self_Employed")).  // guess replaced null values
                    
                    na.fill(146.4, Seq("LoanAmount")).    //mean replaced null values
                    na.fill(90, Seq("Loan_Amount_Term")). // mean replaced null values
                    na.fill(1, Seq("Credit_History")).   // mean replaced null values
                    // Seq( "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term") column to double
                    withColumn("ApplicantIncome", col("ApplicantIncome").cast(DoubleType)).
                    withColumn("CoapplicantIncome", col("CoapplicantIncome").cast(DoubleType)).
                    withColumn("LoanAmount", col("LoanAmount").cast(DoubleType)).
                    withColumn("Loan_Amount_Term", col("Loan_Amount_Term").cast(DoubleType))
                    
            
        var rawTestDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").
                     option("inferSchema","true").load("/Users/keeratjohar2305/Downloads/Dataset/AVtest_LoanPrediction2.csv").
                     //withColumn("Loan_Status", lit(0.0) )
                     withColumnRenamed("loan_status", "Loan_Status")
                    
                      
//                    na.fill("Female", Seq("Gender")). // blindly  replaced null values
//                    na.fill("Yes", Seq("Married")).   //blindly replaced null values
//                    na.fill("2", Seq("Dependents")).  // median replaced null values
//                    na.fill("No", Seq("Self_Employed")).  // guess replaced null values
//                    na.fill(146.4, Seq("LoanAmount")).    //mean replaced null values
//                    na.fill(90, Seq("Loan_Amount_Term")). // mean replaced null values
//                    na.fill(1, Seq("Credit_History")).   // mean replaced null values
//                    // Seq( "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term") column to double
//                    withColumn("ApplicantIncome", col("ApplicantIncome").cast(DoubleType)).
//                    withColumn("CoapplicantIncome", col("CoapplicantIncome").cast(DoubleType)).
//                    withColumn("LoanAmount", col("LoanAmount").cast(DoubleType)).
//                    withColumn("Loan_Amount_Term", col("Loan_Amount_Term").cast(DoubleType))

                     
         println("Origina Data summary")
         rawTestDF.printSchema
         rawTrainDF.printSchema
         rawTestDF.show(30)
         ML_scalaAdvanceMethods.dsShape(rawTestDF)
         ML_scalaAdvanceMethods.summaryCustomized(rawTestDF).show()
         
         ML_scalaAdvanceMethods.dataFitnessCheck(rawTestDF)
 
  
      val labelCol = Seq("Loan_Status")
      val traningDF = ML_scalaAdvanceMethods.getStringIndexersArray(labelCol)(0).fit(rawTrainDF).transform(rawTrainDF).
                                drop("Loan_Status").
                               withColumnRenamed("Loan_StatusIndexed", "label")
      val testingDF = ML_scalaAdvanceMethods.getStringIndexersArray(labelCol)(0).fit(rawTestDF).transform(rawTestDF).
                                drop("Loan_Status").
                               withColumnRenamed("Loan_StatusIndexed", "label")
//      import org.apache.spark.ml.feature.Imputer
//      new Imputer()
//  .setInputCols(Array("a", "b"))
//  .setOutputCols(Array("out_a", "out_b").fit(DF).transform(DF)
                               
    //val Array(traningDF,testingDF) = XDF.randomSplit(Array(0.7,0.3),seed=9999)
                      
//        
//         // drop all the na Data 
//         println("drop all na")
//           val newDf_afterDroppedRows = rawDF.na.drop(Seq("LoanAmount", "Loan_Amount_Term", "Credit_History"))
//            println("1. Total Rows Count after Deleting null value records: "+newDf_afterDroppedRows.count())
//        
//            newDf_afterDroppedRows.summary().show()
//        
//         /* Fill missing values (null or NaN) with a specific value for all columns */
//              //println("drop all na [ Fill missing values (null or NaN) with a specific value ]")
//              val filledWith_half = rawDF.na.fill(0.5)
//              println( "2. Total Rows Count after [ Fill missing values (null or NaN) with a specific value For All Columns ]: "+filledWith_half.count())
//              filledWith_half.summary().show()
//              
//         /* Fill missing values (null or NaN) with a specific value for certain columns */
//              val filledWith_halfFewColumns = rawDF.na.fill(0.5, Seq("Credit_History"))
//              println("3. Total Rows Count after [ Fill missing values (null or NaN) with a specific value for certain columns(Credit_History)]: "+filledWith_halfFewColumns.count())
//              filledWith_halfFewColumns.summary().show()
//              
//              val map = Map("ApplicantIncome" -> 1000.0,"LoanAmount" -> 500.0,"Credit_History" -> 0.5)
//              
//         /* Fill missing values of each column with specified value */
//              val fill_FewColumns = rawDF.na.fill(map)
//            println("4. Total Rows Count after [ Fill missing values of each column with specified value ]: \n" +map+ " = "  +fill_FewColumns.count())
//              fill_FewColumns.summary().show()
//        
                               
  //https://github.com/dalwindr/spark_scala_ML1.git
   //https://github.com/dalwindr/ML_dataset.git
   // https://github.com/dalwindr/scala_withBookeh.git
                               
                               
           def missingValFilled(DFmissingCol: org.apache.spark.sql.DataFrame,coloumnName: String) : org.apache.spark.sql.DataFrame ={
                   val colDataDF:org.apache.spark.sql.DataFrame  = DFmissingCol.select(col(coloumnName)).filter(col(coloumnName).isNull)
                   println(" missing count= "+ colDataDF.count())
                  
                    if (colDataDF.count()> 0)
                        {
                        //println("I am here"+ coloumnName)
                         //DFmissingCol.select(coloumnName).show()
                         val mean_colData: Double = DFmissingCol.select(coloumnName).filter(col(coloumnName).isNotNull).rdd.map(x=> x.mkString.toInt).mean()//.asInstanceOf[Double]
                         println(" mean_colData count= "+ mean_colData)
                         val fill_MissingValuesDF:org.apache.spark.sql.DataFrame  = DFmissingCol.na.fill(mean_colData,Seq(coloumnName))
                       
                         println("Missing rows for " + coloumnName + 
                                    "\nBefore:  "+ colDataDF.count() + 
                                    "\nand After: " + fill_MissingValuesDF.select(col(coloumnName)).filter(col(coloumnName).isNull).count() + 
                                    "\nits Mean  =  " + mean_colData)
                       
                        fill_MissingValuesDF
                        }
                    else 
                        DFmissingCol
                   //fill_MissingValuesDF
                }
          
          def EDA_univariate_Filling_missing_Val( df: org.apache.spark.sql.DataFrame) : org.apache.spark.sql.DataFrame   =  {
                val numericCol = Seq("ApplicantIncome","LoanAmount" ,"Credit_History" );     
               val ApplicantIncomeMissingFilled:org.apache.spark.sql.DataFrame = missingValFilled(df, "ApplicantIncome");
               val LoanAmountMissingFilled:org.apache.spark.sql.DataFrame = missingValFilled(ApplicantIncomeMissingFilled, "LoanAmount");
               val Credit_HistoryFilled:org.apache.spark.sql.DataFrame = missingValFilled(LoanAmountMissingFilled, "Credit_History");
                Credit_HistoryFilled;
              }
          
//          
//          println("Value filled with mean")
//          //"4. Value filled with mean")
//          EDA_univariate_Filling_missing_Val(rawDF).summary().show()
//           //Credit_HistoryFilled
//             //String to Struct Type creation to build Struct type schema
//          val schemaString = "Loan_ID,Gender,Married,Dependents,Education, Self_Employed,ApplicantIncome, CoapplicantIncome,LoanAmount,Loan_Amount_Term, Credit_History,Property_Area,Loan_Status"
//          val schema = schemaString.split(",").map{ field => 
//               if(field == "ApplicantIncome" || field == "CoapplicantIncome" ||field == "LoanAmount" || field == "Loan_Amount_Term" || field == "Credit_History")
//                StructField(field, DoubleType)
//                   else
//                StructField(field, StringType)
//              }
//        
//           val schema_Applied = StructType(schema)
//            
//            schema_Applied.foreach(println)
//        
      
    
    val catgoricalCatColumn = Seq("Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area")
    val NumericalCatColumns = Seq( "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")

    //val assembler = new VectorAssembler().setInputCols(catgoricalCatColumn.map(_+"Indexed").toArray ++ NumericalCatColumns).setOutputCol("features").setHandleInvalid("skip")
   
    val stageArray = ML_scalaAdvanceMethods.CategoricalFeatureVectorzing(catgoricalCatColumn) ++ 
                    ML_scalaAdvanceMethods.FeatureAssembler(catgoricalCatColumn,NumericalCatColumns)  
                    
    
    // Chain indexers and tree in a Pipeline
    val pipelinedStages = new Pipeline().setStages( stageArray )
    
           // create piped train DF
       val pipedDF = pipelinedStages.fit(traningDF).transform(traningDF)
       
       println("pipedDF training DF")
       pipedDF.show()
       
       // create piped test DF
       val pipedtestDF = pipelinedStages.fit(testingDF).transform(testingDF)
       println("pipedtestDF testing DF")
       pipedtestDF.show()

    
    
       ML_scalaAdvanceMethods.CallNaiveBayesAlgo(pipedDF,pipedtestDF, "Binary")
       ML_scalaAdvanceMethods.CallOneVsALLAlgo(pipedDF, pipedtestDF, "Binary")
       ML_scalaAdvanceMethods.CallGradiantBoosterTreeLAlgo(pipedDF, pipedtestDF) // Only support binary classification
       ML_scalaAdvanceMethods.CallDecisionTreeClassifierLAlgo(pipedDF, pipedtestDF, "Binary")
       ML_scalaAdvanceMethods.CallRandomForestClassifierLAlgo(pipedDF, pipedtestDF, "Binary")
       ML_scalaAdvanceMethods.CallLogisticRegressionAlgo(pipedDF, pipedtestDF, "Binary")
       //ML_scalaAdvanceMethods.CallMultiLayerPerceptrolAlgo(pipedDF, pipedtestDF, "Binary") // only support MultiClassfiction
       
          
          
          
                
     }
  }
  
     
  