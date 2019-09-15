  
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

import java.lang.System

object ModelBuilder_bankTransaction_LR {
  def main(args:Array[String])={
    
  Logger.getLogger("org").setLevel(Level.ERROR)
    
 
      val sparkSession = SparkSession
      .builder()
      .config("spark.master", "local")
      .appName("LogisticRegressionExample")
      .getOrCreate()
  
  var modelFile = "/Users/keeratjohar2305/Downloads/ML_MODEL/bankTransactionLR_prediction_model";
  var file = "/Users/keeratjohar2305/Downloads/DBS-Hackathon/Data/TestData.csv";
  
  val sqlContext = new SQLContext(sparkSession.sparkContext)
  val df = sqlContext.read.format("com.databricks.spark.csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .option("delimiter",",")
                    .load(file)
  
  df.show()
  
  
  val featureCols =  Array("Amount", "Distance_from_primary_loc")
  
  //val featureCols =  Array("CustomerID", "TID", "Amount", "Distance_from_primary_loc")
  print("Assembler to Add feature columns")
  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val df2 = assembler.transform(df)
  df2.show()
  
  
  print("label indexex to Add feature columns")
  val labelIndexer = new StringIndexer().setInputCol("is_fraud").setOutputCol("label")
  val df3 = labelIndexer.fit(df2).transform(df2)
  df3.show()

  
  val lr = new LogisticRegression()
  
  val model = lr.fit(df3)
  model.save(modelFile)
  println("Training model saved as $modelFile")
  //System.exit(1)
      
    
    
    
  }
   
}
