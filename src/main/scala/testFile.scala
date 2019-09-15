import java.lang.System
import org.apache.log4j._

import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object testFile extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("Random Forest").config("spark.master","local").getOrCreate()
  //val ColTypeDF =  sc.parallelize(df.dtypes.map(_._2.substring(0,4))).toDF.show()

    var mySchemaSeq = Array("a", "cnt", "major")
    val  df = spark.sqlContext.createDataFrame(Seq(("a", 1, "m1"), ("a", 1, "m2")
    , ("a", 2, "m3"),
    ("a", 3, "m4"), ("b", 4, "m1"), ("b", 1, "m2"),
    ("b", 2, "m3"), ("c", 3, "m1"), ("c", 4, "m3"),
    ("c", 5, "m4"), ("d", 6, "m1"), ("d", 1, "m2"),
    ("d", 2, "m3"), ("d", 3, "m4"), ("d", 4, "m5"),
    ("e", 4, "m1"), ("e", 5, "m2"), ("e", 1, "m3"),
    ("e", 1, "m4"), ("e", 1, "m5"))).toDF(mySchemaSeq: _*)
    
    df.groupBy("a").pivot("major").max("cnt")//.fillna(0)
    
    
   val df2=  df.withColumn("new" , when(col("cnt").isin ( 1,2,3,4) ,col("cnt"))).withColumn("new1" , when(col("a").isin ( "a","b","c","d") ,col("a")))
   //.groupBy(col("new"))
    
  // df2.select(
       df2.columns.map(c=> count(when(col(c).isNaN || col(c).isNull(), col(c))))
   val title_list=("Mrs"->"Mrs", "Mr"-> "Mr", "Master"-> "Master", "Miss"-> "Miss", "Major"-> "Mr", "Rev"->"Mr",
                    "Ms"-> "Miss", "Mlle"-> "Mlle","Col"-> "Mr", "Capt"-> "Mr", "Mme"->"Mrs", "Countess"->"Mrs",
                    "Don"-> "Mr", "Jonkheer"-> "Mr")
  
                    
                    
}