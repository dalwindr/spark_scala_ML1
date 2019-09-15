
/**
  * Detecting Outlier using std deviations
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

object DetectingOutlierTitanicDataSet {
  
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
  .                          load("/Users/keeratjohar2305/Downloads/Dataset/Titanic_train.csv")
        //rawDF.show()
        //rawDF.describe().show() 
         println("Origina Data summary")
        rawDF.summary().show()
        rawDF.printSchema
    val fareVaues_AtDiffr_Itervals =  scala.collection.mutable.ListBuffer[Long]()
    val minValue = 0.0
    val maxValue = 513
    val bins = 5
    val range = (maxValue - minValue) / 5.0
    var minCounter = minValue
    var maxCounter = range


    while (minCounter < maxValue) {
      val valuesBetweenRange = rawDF.filter(col("Fare").between(minCounter, maxCounter))
      fareVaues_AtDiffr_Itervals.+=(valuesBetweenRange.count())
      minCounter = maxCounter
      maxCounter = maxCounter + range
    }
    println("Fare Values at Different Ranges:")
    fareVaues_AtDiffr_Itervals.foreach(println)

  // 1.1 Outlier Detection mean+2(sigma) or mean-2(sigma)

    //https://www.youtube.com/watch?v=eeRrNT7DUHk -- unsstand deviation for continious or range of data
    //https://www.youtube.com/watch?v=t8kDuV1Alt4
    val meanFare = rawDF.select("Fare").select(mean("Fare")).first()(0).asInstanceOf[Double]
    val stddevFare = rawDF.select(stddev("Fare")).first()(0).asInstanceOf[Double]
    val upper_threshold = meanFare + 2*stddevFare
    val lower_threshold = meanFare - 2*stddevFare
    val fareValues_MoreThanUpperthrshold = rawDF.select("Fare").filter(col("Fare") > upper_threshold)
    val fareValues_LessThanLowerthrshold = rawDF.select("Fare").filter(col("Fare") < lower_threshold)
    
    val summary_FareValuesMoreThanUppr = fareValues_MoreThanUpperthrshold.summary()
    println("Summary Of Fare Values Greater Than Upper Threshold:")
    summary_FareValuesMoreThanUppr.show()
    val summary_FareValuesLessThanLowr = fareValues_LessThanLowerthrshold.describe()
    println("Summary Of Fare Values Less Than Lower Threshold:")
    summary_FareValuesLessThanLowr.show()

   // 1.2. Outlier Detection mean+3(sigma) or mean-3(sigma)
    val upper_threshold1 = meanFare + 3*stddevFare
    val lower_threshold1 = meanFare - 3*stddevFare
    val fareValues_MoreThanUpperthrshold1 = rawDF.select("Fare").filter(col("Fare") > upper_threshold1)
    val fareValues_LessThanLowerthrshold1 = rawDF.select("Fare").filter(col("Fare") < lower_threshold1)
    val summary_FareValuesMoreThanUppr1 = fareValues_MoreThanUpperthrshold1.describe()
    println("Summary Of Fare Values Greater Than Upper Threshold:")
    summary_FareValuesMoreThanUppr1.show()
    val summary_FareValuesLessThanLowr1 = fareValues_LessThanLowerthrshold1.describe()
    println("Summary Of Fare Values Less Than Lower Threshold:")
    summary_FareValuesLessThanLowr1.show()

    // 1.3. Calculating z scores and apply outlier detection method:
    val titanic_Data_StdFareValues = rawDF.withColumn("StdFareValue", (col("Fare")- meanFare)/stddevFare)
    val mean_FareStdvalue = titanic_Data_StdFareValues.select(mean("StdFareValue")).first()(0).asInstanceOf[Double]
    
    val stddev_FareStdvalue = titanic_Data_StdFareValues.select(stddev("StdFareValue")).first()(0).asInstanceOf[Double]
    val upper_threshold_std = mean_FareStdvalue + 3*stddev_FareStdvalue
    val lower_threshold_std = mean_FareStdvalue - 3*stddev_FareStdvalue
    
    val fareValues_MoreThanUpperthrshold_std = titanic_Data_StdFareValues
                                                          .select("StdFareValue")
                                                          .filter(titanic_Data_StdFareValues("StdFareValue") > upper_threshold_std)
    val fareValues_LessThanLowerthrshold_std =titanic_Data_StdFareValues
                                                          .select("StdFareValue")
                                                          .filter(titanic_Data_StdFareValues("StdFareValue") < lower_threshold_std)

    val summary_FareValuesMoreThanUppr_Std =fareValues_MoreThanUpperthrshold_std.describe()
    println("Summary Of Standardized Fare Values Greater Than Upper Threshold")
      summary_FareValuesMoreThanUppr_Std.show()
    
   val summary_FareValuesLessThanLowr_Std =fareValues_LessThanLowerthrshold_std.describe()
   println("Summary Of Standardized Fare Values Less Than Lower Threshold")
      summary_FareValuesLessThanLowr_Std.show()


    // 2. Mean of Fare variable ( Median)
    val fare_Details_Df = rawDF.select("Fare")
    val fare_DetailsRdd = fare_Details_Df.rdd.map{row => row.getDouble(0)}
    val countOfFare = fare_DetailsRdd.count()
    val sortedFare_Rdd = fare_DetailsRdd.sortBy(fareVal => fareVal )
    val sortedFareRdd_WithIndex = sortedFare_Rdd.zipWithIndex()

    val median_Fare = if(countOfFare%2 ==1)
                    sortedFareRdd_WithIndex.filter{case(fareVal:Double, index:Long) => index == (countOfFare-1)/2}.first._1
                  else{
                    val elementAtFirstIndex = sortedFareRdd_WithIndex.filter{case(fareVal:Double, index:Long) => index == (countOfFare/2)-1}.first._1
                    val elementAtSecondIndex = sortedFareRdd_WithIndex.filter{case(fareVal:Double, index:Long) => index == (countOfFare/2)}.first._1
                    (elementAtFirstIndex+elementAtSecondIndex)/2.0
                  }
    println("median_Fare=" + median_Fare)
    
    //UDF Code
    val coder= (fareValue:Double, medianValue:Double) =>  
                          if((fareValue-medianValue) < 0) 
                                    -1*(fareValue-medianValue)
                          else
                                    (fareValue-medianValue)
    val sqlFunc = udf(coder)
    spark.udf.register("sqlFunc",sqlFunc)
    
    // 2.1 Apply Mean Absoluate Deviation MAD for outlier detection
    val fare_Details_WithAbsDeviations = fare_Details_Df.withColumn("AbsDev_FromMedian",sqlFunc(col("Fare"), lit(median_Fare)))
    val fare_AbsDevs_Rdd = fare_Details_WithAbsDeviations.rdd.map{row =>row.getDouble(1)}
    val count = fare_AbsDevs_Rdd.count()
    val sortedFareAbsDev_Rdd = fare_AbsDevs_Rdd.sortBy(fareAbsVal => fareAbsVal )
    val sortedFare_AbsDevRdd_WithIndex = sortedFareAbsDev_Rdd.zipWithIndex()
    val median_AbsFareDevs = 
                  if(count%2 ==1)
                      sortedFare_AbsDevRdd_WithIndex.filter{case(fareAbsVal:Double,index:Long) =>index == (count-1)/2}.first._1
                  else{
                        val elementAtFirstIndex = sortedFare_AbsDevRdd_WithIndex
                                                          .filter{case(fareAbsVal:Double,index:Long) =>index == (count/2)-1}.first._1
                        val elementAtSecondIndex = sortedFare_AbsDevRdd_WithIndex
                                                          .filter{case(fareAbsVal:Double, index:Long) => index == (count/2)}.first._1
                        (elementAtFirstIndex+elementAtSecondIndex)/2.0
                     }
    val mad = 1.4826*median_AbsFareDevs
    println("Median Absolute Deviation is:"+mad)

    // 2.3 Outlier based on MAD (median)
    val upper_mad = median_Fare + 3 * mad
    val lower_mad = median_Fare - 3 * mad
    val fareValues_MoreThanUpperthrshold_mad= rawDF.select("Fare").filter(col("Fare") > upper_mad)
    val fareValues_LessThanLowerthrshold_mad = rawDF.select("Fare").filter(col("Fare") < lower_mad)
    
    val summary_FareValuesMoreThanUppr_MAD = fareValues_MoreThanUpperthrshold_mad.describe()
    println("Summary Of Fare Values Greater Than Upper Threshold In MAD Approach:")
        summary_FareValuesMoreThanUppr_MAD.show()
    
        val summary_FareValuesLessThanLowr_MAD =  fareValues_LessThanLowerthrshold_mad.describe()
    println("Summary Of Fare Values Less Than Lower Threshold In MAD Approach:")
    summary_FareValuesLessThanLowr_MAD.show()

  }

  
}