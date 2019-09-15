
import java.lang.System
import org.apache.spark.SparkConf
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.log4j._
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics

object ModelBuilder_LR_diamond {
  def main(args: Array[String]) {
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  val spark = SparkSession.builder()
              .config("spark.master","local")
              .appName("Diagmon ML logisitic Algo")
              .getOrCreate()
  val fileName="/Users/keeratjohar2305/Downloads/diamonds.csv"
  val rawRDD = spark.read.format("csv")
              .option("header","true")
              .option("inferSchema","true")
              .option("Delimiter",",")
              .load(fileName)
             //.withColumn("priceOutputVar")
  rawRDD.show()
  
  rawRDD.printSchema()
  }
              
}