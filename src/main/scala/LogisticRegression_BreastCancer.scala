//https://www.bmc.com/blogs/using-logistic-regression-scala-spark/
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.log4j._

import java.lang.System

import ML_scalaAdvanceMethods._
import scala.util.Random

/**
*  Missing values show up as a dot.  The dot function below
*  returns -1 if there is a dot or blank value. And it converts strings to double.
* Then later we will
*  delete all rows that have any -1 values.
*/
object LogisticRegression_BreastCancer {
  
def dot (s: String) : Double = {
        if (s.contains(".") || s.length == 0) {
          return -1
        } else {
return s.toDouble
}
}


/**
*  We are going to use a Dataframe.  It requires a schema.
* So we create that below.  We use the same column names
*  as are in the .dat file.
*/
def main(args: Array[String]){
val schema = StructType (
StructField("STR", DoubleType, true) ::
StructField("OBS", DoubleType, true) ::
StructField("AGMT", DoubleType, true) ::
StructField("FNDX", DoubleType, true) ::
StructField("HIGD", DoubleType, true) ::
StructField("DEG",DoubleType, true) ::
StructField("CHK", DoubleType, true) ::
StructField("AGP1", DoubleType, true) ::
StructField("AGMN", DoubleType, true) ::
StructField("NLV", DoubleType, true) ::
StructField("LIV", DoubleType, true) ::
StructField("WT", DoubleType, true) ::
StructField("AGLP", DoubleType, true) ::
StructField("MST", DoubleType, true) ::  Nil)
/**
*  Read in the .dat file and use the regular expression
*  \s+ to split it by spaces into an RDD.
*/


var spark = SparkSession
            .builder()
            .appName("Spark Logistic Regression APP")
            .config("spark.master","local[*]")
            .getOrCreate()

            
Logger.getLogger("org").setLevel(Level.ERROR)
            
val modelFile =  "/Users/keeratjohar2305/Downloads/breast_Cancer_lr_model"

val readingsRDD = spark.sparkContext.textFile("/Users/keeratjohar2305/Downloads/breast_Cancer.csv")
val RDD1 = readingsRDD.map(_.split("\\s+"))
var _first = RDD1.first().mkString;
val RDD= RDD1.filter(x=> x(0) != _first ).map( x=> x(0).split(",")).filter(_.size==14 )

RDD.take(4).foreach(x=>println(x.mkString(",")))

/**
*   Run the dot function over every element in the RDD to convert them
*   to doubles, since that if the format requires by the Spark ML LR model.
*   Note that we skip the first one since that is just a blank space.
*/


val rowRDD = RDD.map(s => Row(dot(s(0)),dot(s(1))
,dot(s(2)),dot(s(3)),dot(s(4)),dot(s(5)),dot(s(6)),dot(s(7)),dot(s(8)),dot(s(9)),dot(s(10)),dot(s(11)),dot(s(12)),dot(s(13))
))
rowRDD.take(4).foreach(x=>println(x.mkString(",")))



/**
* Now create a dataframe with the schema we described above,
*
*/
val readingsDF = spark.createDataFrame(rowRDD, schema)
print(readingsDF.show())


/**
*  Create a new dataframe dropping all of those with missing values.
*/
var cleanDF = readingsDF.filter(readingsDF("STR") > -1 && readingsDF("OBS") > -1 && readingsDF("AGMT")  > -1  && readingsDF("FNDX") > -1 && readingsDF("HIGD") > -1  && readingsDF("DEG") > -1 && readingsDF("CHK") > -1 && readingsDF("AGP1") > -1  && readingsDF("AGMN") > -1  && readingsDF("NLV") > -1  && readingsDF("LIV") > -1 && readingsDF("WT") > -1 && readingsDF("AGLP") > -1 && readingsDF("MST") > -1)


/**
*  Now comes something more complicated.  Our dataframe has the column headings
*  we created with the schema.  But we need a column called “label” and one called
* “features” to plug into the LR algorithm.  So we use the VectorAssembler() to do that.
* Features is a Vector of doubles.  These are all the values like patient age, etc. that
* we extracted above.  The label indicated whether the patient has cancer.
*/
val featureCols = Array("STR" , "OBS" , "AGMT" , "HIGD" , "DEG" , "CHK" , "AGP1" , "AGMN" , "NLV" , "LIV" , "WT" , "AGLP",  "MST" )
val featurAassembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
val df2 = featurAassembler.transform(cleanDF)
print("Row DF ...............................")
summaryCustomized(cleanDF).show()


print ("Assember DF (observe New {features} Column)..........................")
df2.show(false)


/**
* Then we use the StringIndexer to take the column FNDX and make that the label.
*  FNDX is the 1 or 0 indicator that shows whether the patient has cancer.
* Like the VectorAssembler it will add another column to the dataframe.
*/
val labelIndexer = new StringIndexer().setInputCol("FNDX").setOutputCol("label")
val df3 = labelIndexer.fit(df2).transform(df2)

println("Observe New {label} columnn in the dataset")
df3.show(false)

val Array( trainingdf, testingDF) = df3.randomSplit(Array(0.9,.1),seed=10L)
 
/**
*   Now we declare the LR model and run fit and transform to make predictions.
*/
//Lets create model
val lr = new LogisticRegression()
val model = lr.fit(trainingdf)
println("Logistric Model created and stored  in a file")
model.save(modelFile)

println("Logistric Model created and stored  in a file")
val predictions = model.transform(testingDF)


/**
*  Now we print it out.  Notice that the LR algorithm added a “prediction” column
*  to our dataframe.   The prediction in almost all cases will be the same as the label.  That is
* to be expected it there is a strong correlation between these values.  In other words
* if the chance of getting cancer was not closely related to these variables then LR
* was the wrong model to use.  The way to check that is to check the accuracy of the model.
*  You could use the BinaryClassificationEvaluator Spark ML function to do that.
* Adding that would be a good exercise for you, the reader.
*/
print("lets predict the values Observer new column{prediction}")
predictions.select ("features", "label", "prediction").show()


}
}