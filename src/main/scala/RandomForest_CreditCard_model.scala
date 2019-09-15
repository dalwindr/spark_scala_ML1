//https://www.kaggle.com/uciml/german-credit/downloads/german-credit-risk.zip/1
import java.lang.System
import org.apache.log4j._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator}
import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel,LogisticRegression,LogisticRegressionModel}
import org.apache.spark.ml.linalg.{Vectors,DenseVector}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object RandomForest_CreditCard_model {
def main(args: Array[String]){
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  val spark = SparkSession.builder().appName("Random Forest").config("spark.master","local").getOrCreate()
  var rawDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").option("inferSchema","true").load("/Users/keeratjohar2305/Downloads/german_credit_data.csv")
  rawDF.show()
  rawDF.printSchema 
  rawDF.summary().show()
  
  
  // DF columns list
  //describe(colunms_list.tail.toSeq:_*).show()
  val colunms_list = rawDF.columns
  
  
  //println(rawDF.describe(colunms_list.tail.toSeq:_*).summary().show())
  val colunms_list1 = Array("Sex","Housing","Saving accounts","Checking account","Purpose")
  
  //print(rawDF.select(colunms_list.head,colunms_list.tail:_*))
  // Array of String indexexer
  val indexer_list=colunms_list1.toSeq.map(col=> new StringIndexer().setInputCol(col).setOutputCol(col +"Indexer")  ).toArray
 
  //Array of OneHotEncoder Encoder
  val encoded_list=colunms_list1.toSeq.map(col=> new OneHotEncoderEstimator().setInputCols(Array(col+"Indexer")).setOutputCols(Array(col +"Encoded"))).toArray
  
  
  //assembler
  var numericfeature_list= rawDF.columns.filter(colunms_list1.toSet(_)==false).tail 
  val myassembler =  new VectorAssembler().setInputCols(numericfeature_list ++ colunms_list1.map(_+"Encoded")).setOutputCol("features")
  val labelIndexer = new StringIndexer().setInputCol("age").setOutputCol("label")
   
  val Stages= indexer_list ++ encoded_list // ++ Array(myassembler) ++ Array(labelIndexer)
  Stages.foreach(println)
  val partialPipeline = new Pipeline().setStages(Stages).fit(rawDF).transform(rawDF)
  partialPipeline.show()
      
  //describe()
  var Sex_Map= rawDF.select("Sex").distinct().collect.zipWithIndex.map(x=>(x._1.mkString, x._2+1)).toMap
  var Housing_Map= rawDF.select("Housing").distinct().collect.zipWithIndex.map(x=>(x._1.mkString, x._2+1)).toMap
  var Saving_accounts_Map= rawDF.select("Saving accounts").distinct().collect.zipWithIndex.map(x=>(x._1.mkString, x._2+1)).toMap
  var Checking_account_Map= rawDF.select("Checking account").distinct().collect.zipWithIndex.map(x=>(x._1.mkString, x._2+1)).toMap
  var Purpose_Map= rawDF.select("Purpose").distinct().collect.zipWithIndex.map(x=>(x._1.mkString, x._2+1)).toMap
  
  val Sex_UDF = udf ((v: String) => Sex_Map(v) )
  val Housing_UDF = udf ((v: String) => Housing_Map(v) )
  val Saving_accounts_UDF = udf ((v: String) => Saving_accounts_Map(v) )
  val Checking_account_UDF = udf ((v: String) => Checking_account_Map(v) )
  val Purpose_UDF = udf ((v: String) => Purpose_Map(v) )
  
  spark.udf.register("Sex_UDF",Sex_UDF)
  spark.udf.register("Housing_UDF",Housing_UDF)
  spark.udf.register("Saving_accounts_UDF",Saving_accounts_UDF)
  spark.udf.register("Checking_account_UDF",Checking_account_UDF)
  spark.udf.register("Purpose_UDF",Purpose_UDF)
  import spark.implicits._
  //rawDF.select(col("Housing"))
  val DF2 = rawDF.withColumn("SexBin",Sex_UDF(col("Sex")))
                  .withColumn("HousingBin",Housing_UDF(col("Housing")))
                  .withColumn("Saving_accountsBin",Saving_accounts_UDF(col("Saving accounts")))
                  .withColumn("Checking_accountBin",Checking_account_UDF(col("Checking account")))
                  .withColumn("PurposeBin",Purpose_UDF(col("Purpose")))
 DF2.show()
  
 rawDF.groupBy("SexBin").count.show()
 rawDF.groupBy("HousingBin").count.show()
 rawDF.groupBy("aving_accountsBin").count.show()
 rawDF.groupBy("Checking_accountBin").count.show()
 rawDF.groupBy("PurposeBin").count.show()
  rawDF.groupBy("Age").count.show()
 rawDF.groupBy("Job").count.show()
 rawDF.groupBy("Credit amount").count.show()
 rawDF.groupBy("Duration").count.show()
 rawDF.agg(avg("Age"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}
 
 
}
}