/*
 * Data Analysing and data wrangling
 * 
 */

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
import org.apache.log4j._
import org.apache.commons.math3.stat.descriptive.moment.Mean
//
//import org.apache.spark._
//import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}



object Uber_Dataset_Analysis {
  def main(args:Array[String]): Unit = {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
        val spark = SparkSession
                    .builder()
                    .appName("Java Spark SQL basic example")
                    .config("spark.master", "local")
                    .getOrCreate()
       import spark.sqlContext.implicits._     
    //Loading data
    var rawDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").option("inferSchema","true")
  .                          load("/Users/keeratjohar2305/Downloads/uber.csv")
  
    import spark.implicits._
    
    //1. Raw data Summary
    rawDF.summary().show()
    rawDF.printSchema
    
    //  base wise ,date wise , number of trips
    val uberData_new = rawDF.withColumn("BaseNo_Date",concat(col("dispatching_base_number"), lit(":"), col("date")))
    val maxTrips_PerBaseAndDate = uberData_new.groupBy("BaseNo_Date").max("trips")
    maxTrips_PerBaseAndDate.show(10)

    //Define coder Function
    val coder1 = (dateValue:String) => {
                                        val format = new java.text.SimpleDateFormat("MM/dd/yyyy")
                                         val formated_Date = format.parse(dateValue)
                                         formated_Date.getMonth()+1
                                       }

    val coder2 =(baseMonthConcat:String) => baseMonthConcat.split(":")
    
    // Find the month on which basement has more trips
    val sqlFunc1 = udf(coder1)
    val sqlFunc2 = udf(coder2)
    
    spark.udf.register("sqlFunc1",sqlFunc1)
    spark.udf.register("sqlFunc2",sqlFunc2)
    
     println(" appended with 2 new columns  (month, BaseNo_Month)")
    val uberdata_newMonthCol= rawDF.withColumn("month",sqlFunc1(col("date")))
    val uberData_ConcatBaseNo_Month = uberdata_newMonthCol.withColumn("BaseNo_Month",concat(col("dispatching_base_number"), lit(":"), col("month")))
    uberData_ConcatBaseNo_Month.show()
    
    println("month wise base wise trips")
    val sumTrips_PerBaseAndMonth = uberData_ConcatBaseNo_Month.groupBy(col("BaseNo_Month")).sum("trips")
    sumTrips_PerBaseAndMonth.show()
    
    println("split BaseNo_Month into two List in  ")
    val sumTrips_PerBaseMonth_new = sumTrips_PerBaseAndMonth.withColumn("BaseNo",sqlFunc2(col("BaseNo_Month")))
    sumTrips_PerBaseMonth_new.show()
    
    val maxTrips_PerBaseMonth = sumTrips_PerBaseMonth_new.groupBy("BaseNo").max("sum(trips)")
                                .withColumnRenamed("max(sum(trips))","MaxTrips_PerMonth")
    
    val maxTrips_Final = maxTrips_PerBaseMonth
                          .join(sumTrips_PerBaseMonth_new,
                              sumTrips_PerBaseMonth_new("BaseNo") === maxTrips_PerBaseMonth("BaseNo") && 
                              sumTrips_PerBaseMonth_new("sum(trips)") === maxTrips_PerBaseMonth("MaxTrips_PerMonth"))
                          .select ("BaseNo_Month","MaxTrips_PerMonth")
    println("Maximum Trips per basement per month:")
    maxTrips_Final.show() }
  

}