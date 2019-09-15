

  

/*
 *   Here All the features are manually convert to double types and finally convert into dense vectors
 *    Although better ways is to use  Onhot Encoder to convert catagoical column directly to dense vectors
 * 
 * 
 */
//https://github.com/akhil12028/Bank-Marketing-data-set-analysis
//https://archive.ics.uci.edu/ml/datasets/bank+marketing
// label = The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
import org.apache.spark._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.types._
import org.apache.spark.mllib.stat.
{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.expressions.UserDefinedFunction

import ML_scalaAdvanceMethods._
object TermDepositBankMarketingDataLoRA  {


import java.lang.System
import org.apache.log4j._
  def main(args: Array[String]): Unit = { 
    
    Logger.getLogger("org").setLevel(Level.ERROR)
  
   val spark = SparkSession.builder().appName("Random Forest").config("spark.master","local").getOrCreate()
  
    import spark.sqlContext.implicits._

    //Loading data
    val bank_Marketing_Data =
      spark.read.format("com.databricks.spark.csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("Delimiter", ";")
        .load("/Users/keeratjohar2305/Downloads/bank_data.csv").withColumnRenamed("emp.var.rate", "empvarrate")

    bank_Marketing_Data.show(5)
    //Variable Identification
    val selected_Data = bank_Marketing_Data.select("age", "job",
      "marital",
      "default", "housing", "loan", "duration",
      "previous", "poutcome", "empvarrate", "y").withColumn("age",
      col("age").cast(DoubleType))
      .withColumn("duration",
        col("duration").cast(DoubleType))
      .withColumn("previous",
        col("previous").cast(DoubleType))
    selected_Data.show(5)
    
    selected_Data.rdd.take(10).foreach(println)


    //Data Exploration
    /*Summary statistics*/
    //val summary = selected_Data.describe()
    println("Summary Statistics")
    summaryCustomized(selected_Data).show()    
//    /* Unique values for each Field */
    val columnNames = selected_Data.columns
//    val uniqueValues_PerField = columnNames.map { field => field + ":"
//      +selected_Data.select(field).distinct().count()
//    }
//    
//    
//    println("Unique Values for each Field: ")
//    uniqueValues_PerField.foreach(println)
    
    // Frequency of Categories for categorical variables
    val frequency_Variables = columnNames.map {
      fieldName =>
        if (fieldName == "job" || fieldName == "marital" || fieldName
          == "default" ||
          fieldName == "housing" || fieldName ==
          "poutcome")
          Option(fieldName, selected_Data.groupBy(fieldName).count())
        else None
    }
    
    
    val seq_Df_WithFrequencyCount =
      frequency_Variables.filter(optionalDf =>
        optionalDf != None).map { optionalDf => optionalDf.get }
    
    seq_Df_WithFrequencyCount.foreach { case (fieldName, df) =>
      println("Frequency Count of " + fieldName + ":  "+  df.collect().toList)
     
    }

    //Feature Engineering

    /* Applying One Hot encoding of Categorical Variables */
    // One Hot Encoding for Job
    val sqlFunc = udf(coder)
    spark.udf.register("sqlFunc",sqlFunc)
    
    val new_Df_WithDummyJob =
      create_DummyVariables(selected_Data, sqlFunc,"job",0)
    
      val new_Df_WithDummyMarital =
      create_DummyVariables(new_Df_WithDummyJob, sqlFunc,
        "marital", 0)
    val new_Df_WithDummyDefault =
      create_DummyVariables(new_Df_WithDummyMarital, sqlFunc,
        "default", 0)
    val new_Df_WithDummyHousing =
      create_DummyVariables(new_Df_WithDummyDefault, sqlFunc,
        "housing", 0)
    val new_Df_WithDummyPoutcome =
      create_DummyVariables(new_Df_WithDummyHousing, sqlFunc,
        "poutcome", 0)
    val new_Df_WithDummyLoan =
      create_DummyVariables(new_Df_WithDummyPoutcome, sqlFunc,
        "loan", 0)


    println("Before drop:")
     new_Df_WithDummyLoan.show()
      
    val final_Df = new_Df_WithDummyLoan.drop("job")
      .drop("marital")
      .drop("default")
      .drop("housing").drop("loan").drop("poutcome")
    
      println("After drop:")
      final_Df.show()
     
      val indexerModel = new StringIndexer()
      .setInputCol("y")
      .setOutputCol("y_Index")
      .fit(final_Df)
    val indexedDf = indexerModel.transform(final_Df).drop("y")
    indexedDf.show(5)


    //Applying Logistic Regression

    val final_Rdd  =  indexedDf.rdd.map {
      row =>
        val age = row.getAs[Double]("age")
        val duration = row.getAs[Double]("duration")
        val previous = row.getAs[Double]("previous")
        val empvarrate = row.getAs[Double]("empvarrate")
        val job_0 = row.getAs[Double]("job_0")
        val job_1 = row.getAs[Double]("job_1")
        val job_2 = row.getAs[Double]("job_2")
        val job_3 = row.getAs[Double]("job_3")
        val job_4 = row.getAs[Double]("job_4")
        val job_5 = row.getAs[Double]("job_5")
        val job_6 = row.getAs[Double]("job_6")
        val job_7 = row.getAs[Double]("job_7")
        val job_8 = row.getAs[Double]("job_8")
        val job_9 = row.getAs[Double]("job_9")
        val job_10 = row.getAs[Double]("job_10")
        val job_11 = row.getAs[Double]("job_11")

        val marital_0 = row.getAs[Double]("marital_0")
        val marital_1 = row.getAs[Double]("marital_1")
        val marital_2 = row.getAs[Double]("marital_2")
        val marital_3 = row.getAs[Double]("marital_3")

        val default_0 = row.getAs[Double]("default_0")
        val default_1 = row.getAs[Double]("default_1")
        val default_2 = row.getAs[Double]("default_2")

        val housing_0 = row.getAs[Double]("housing_0")
        val housing_1 = row.getAs[Double]("housing_1")
        val housing_2 = row.getAs[Double]("housing_2")

        val poutcome_0 = row.getAs[Double]("poutcome_0")
        val poutcome_1 = row.getAs[Double]("poutcome_1")
        val poutcome_2 = row.getAs[Double]("poutcome_2")

        val loan_0 = row.getAs[Double]("loan_0")
        val loan_1 = row.getAs[Double]("loan_1")
        val loan_2 = row.getAs[Double]("loan_2")
        val label = row.getAs[Double]("y_Index")

        val featurecVec = Vectors.dense(Array(age,duration,previous, empvarrate,
          job_0,job_1,job_2,job_3,job_4,job_5,job_6,job_7,job_8,job_9,job_10,job_11,marital_0,
          marital_1,marital_2,marital_3, default_0,default_1,default_2,housing_0,housing_1,
          housing_2,poutcome_0,poutcome_1,poutcome_2,loan_0,loan_1,loan_2   ))
        LabeledPoint(label, featurecVec)
    }

    
    final_Rdd.take(10).foreach(println)
  
    val splits = final_Rdd.randomSplit(Array(0.8,0.2))
    val training = splits(0).cache()
    val test = splits(1)

    // train the model and create the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    // the the (label point, feature point) on the test data to be predicted
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = "+precision)

    
  }

  def create_DummyVariables(df: DataFrame, udf_Func: UserDefinedFunction, variableType: String, i: Int):
  DataFrame = {
    variableType match {

      case "job" => if (i == 12) df
      else {
        val df_new = df.withColumn (variableType + "_" + i.toString, udf_Func(lit(variableType),col("job"), lit (i) ) )
        create_DummyVariables (df_new, udf_Func,variableType, i + 1)
      }
      case "marital" => if(i == 4) df
      else {
        val df_new = df.withColumn (variableType + "_" + i.toString, udf_Func(lit(variableType),col ("marital"), lit (i) ) )
        create_DummyVariables (df_new, udf_Func, variableType, i + 1)
      }
      case "default" => if(i == 3) df
      else {
        val df_new = df.withColumn (variableType + "_" + i.toString, udf_Func(lit(variableType),col ("default"), lit (i) ) )
        create_DummyVariables (df_new, udf_Func,variableType, i + 1)
      }
      case "housing" => if(i == 3) df
      else {
        val df_new = df.withColumn (variableType +
          "_" + i.toString, udf_Func(lit(variableType),
          col ("housing"), lit (i) ) )
        create_DummyVariables (df_new, udf_Func,
          variableType, i + 1)
      }
      case "poutcome" => if(i == 3) df
      else {
        val df_new = df.withColumn (variableType +
          "_" + i.toString, udf_Func(lit(variableType),
          col ("poutcome"), lit (i) ) )
        create_DummyVariables (df_new, udf_Func,
          variableType, i + 1)
      }
      case "loan" => if(i == 3) df
      else {
        val df_new = df.withColumn (variableType + "_" +i.toString, udf_Func(lit(variableType), col ("loan"), lit (i) ) )
        create_DummyVariables (df_new, udf_Func,variableType, i + 1)
      } } }

  val coder = (variableType:String, columnValue:String,jobNo:Int) =>
    variableType match{
        case "job" => columnValue match {
        case "unemployed" => if (jobNo == 0) 1.0 else 0.0
        case "services" => if (jobNo == 1) 1.0 else 0.0
        case "blue-collar" => if (jobNo == 2) 1.0 else 0.0
        case "unknown" => if (jobNo == 3) 1.0 else 0.0
        case "housemaid" => if (jobNo == 4) 1.0 else 0.0
        case "entrepreneur" => if (jobNo == 5) 1.0 else 0.0
        case "self-employed" => if (jobNo == 6) 1.0 else 0.0
        case "retired" => if (jobNo == 7) 1.0 else 0.0
        case "admin." => if (jobNo == 8) 1.0 else 0.0
        case "management" => if (jobNo == 9) 1.0 else 0.0
        case "technician" => if (jobNo == 10) 1.0 else 0.0
        case "student" => if (jobNo == 11) 1.0 else 0.0
      }
      case "marital" => columnValue match {
        case "unknown" => if(jobNo == 0) 1.0 else 0.0
        case "divorced" => if(jobNo == 1) 1.0 else 0.0
        case "single" => if(jobNo == 2) 1.0 else 0.0
        case "married" => if(jobNo == 3) 1.0 else 0.0
      }
      case "default" => columnValue match {
        case "unknown" => if(jobNo == 0) 1.0 else 0.0
        case "no" => if(jobNo == 1) 1.0 else 0.0
        case "yes" => if(jobNo == 2) 1.0 else 0.0
      }
      case "housing" => columnValue match {
        case "unknown" => if(jobNo == 0) 1.0 else 0.0
        case "no" => if(jobNo == 1) 1.0 else 0.0
        case "yes" => if(jobNo == 2) 1.0 else 0.0
      }
      case "poutcome" => columnValue match {
        case "nonexistent" => if(jobNo == 0) 1.0 else 0.0
        case "failure" => if(jobNo == 1) 1.0 else 0.0
        case "success" => if(jobNo == 2) 1.0 else 0.0
      }
      case "loan" => columnValue match {
        case "unknown" => if(jobNo == 0) 1.0 else 0.0
        case "no" => if(jobNo == 1) 1.0 else 0.0
        case "yes" => if(jobNo == 2) 1.0 else 0.0
      } }
}
