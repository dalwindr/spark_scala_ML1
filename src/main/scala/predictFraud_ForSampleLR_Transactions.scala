
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary,LogisticRegression}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.LogisticRegressionModel


import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.sql.streaming.{OutputMode, Trigger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import java.lang.System
import org.apache.log4j._
import org.apache.spark.sql.Row

object predictFraud_ForSampleLR_Transactions {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
   
    var spark = SparkSession
    .builder()
    .appName("Detecting Fraud with LR Model")
    .config("spark.master","local")
    .getOrCreate()

  var transRDD =spark.sparkContext.parallelize(Seq(Row(1231006815,234654618,"John Stellar","Debit",1023444,20190715,124501,99999999.2324,1,0,7987545476L)))  
  print(transRDD.collect().mkString(","))
  val schema = StructType(
      Array(StructField("CustomerID", IntegerType),
            StructField("TID", IntegerType),
            StructField("CustomerName", StringType),
            StructField("Type", StringType),
            StructField("Location", IntegerType),
            StructField("Date", IntegerType),
            StructField("Time", IntegerType),
            StructField("Amount", DoubleType),
            StructField("Distance_from_primary_loc", IntegerType),
            StructField("is_fraud", IntegerType),
            StructField("ContactNo", LongType))) ;
    var transDF = spark.createDataFrame(transRDD,schema) 
    
    print("\nRaw Transactions")
    transDF.show()
    //transDF.createOrReplaceTempView("test");
    
    print("Assemble to Add selected feature to DataFrame ")
    val featureCols =  Array( "Amount", "Distance_from_primary_loc");
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features") ;
    val df2 = assembler.transform(transDF) ; 
    df2.show(false)
    
    print("LabelIndexer to Add selected Label to DataFrame ")
    val labelIndexer = new StringIndexer().setInputCol("is_fraud").setOutputCol("label");
    val df3 = labelIndexer.fit(df2).transform(df2) ;
    
    df3.show()
   
    print("model loading ")
    val modelFile="/Users/keeratjohar2305/Downloads/ML_MODEL/bankTransactionLR_prediction_model"
    val model = LogisticRegressionModel.load(modelFile) ;

    print("prediction started")
    val predictions = model.transform(df3);
    var df4 = predictions.select("CustomerID", "TID", "Amount", "Distance_from_primary_loc");
    predictions.show()
    
     print("prediction started")
    val df5 = predictions.filter("prediction=1");
    df5.show(false);

    
  //val sparkConf = new SparkConf().setAppName("HdfsWordCount")
  //val ssc = new StreamingContext(sparkConf, Seconds(10))
  //val lines = ssc.textFileStream("/Users/keeratjohar2305/Downloads/SPARK_POC/DBS_data")
  // val words = lines.foreachRDD{
//   rdd =>
    
//    val spark = SparkSession.builder.appName("Spark-Kafka-Integration").master("local").getOrCreate();import spark.implicits._;
//    val schema = StructType(
//      Array(StructField("CustomerID", StringType),
//            StructField("TID", StringType), 
//            StructField("CustomerName", StringType),
//            StructField("Type", StringType),
//            StructField("Location", StringType),
//            StructField("Date", StringType),
//            StructField("Time", StringType),
//            StructField("Amount", StringType),
//            StructField("Distance_from_primary_loc", StringType),
//            StructField("is_fraud", StringType),
//            StructField("ContactNo", StringType))) ;
//            
//  println("""
//    ssssssssssssssssssssssssssss
//    #######################################################################
//
//
//
//
//
//    ###########################       STOP""")
//  val fileStreamDf1=rdd.map(x=> x.split(",")).map{x=> (x(0).toLong,x(1).toInt,x(2),x(3),x(4).toInt,x(5).toInt,x(6).toInt,x(7).toInt,x(8).toInt,x(9).toInt,x(10).toLong)}.toDF("CustomerID","TID","CustomerName","Type","Location","Date","Time","Amount","Distance_from_primary_loc","is_fraud","ContactNo")
//
//  println("sdfdsffd",fileStreamDf1.show());
//println("""
//    xxxxxxxxxxxxx""")
//
//
// fileStreamDf1.createOrReplaceTempView("test");
//
//  val featureCols =  Array( "Amount", "Distance_from_primary_loc");
//  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features") ;
//  val labelIndexer = new StringIndexer().setInputCol("is_fraud").setOutputCol("label");
//  val df2 = assembler.transform(fileStreamDf1) ; 
//  val df3 = labelIndexer.fit(df2).transform(df2) ;
//  df3.select("features").show()
//  val model = LogisticRegressionModel.load(modelFile) ;
//
//  val predictions = model.transform(df3);
//
//  var df4 = predictions.select("CustomerID", "TID", "Amount", "Distance_from_primary_loc");
//
//  val df5 = df4.filter("prediction=1");
//  df5.show(false);
//}
//
//
//    
//    ssc.start()
//    ssc.awaitTermination()

   
    
    
    
    
  }
  
}