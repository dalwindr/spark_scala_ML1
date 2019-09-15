name:= "simple-spark"
version:= "1.0"
scalaVersion:= "2.11.12"
val sparkVersion = "2.4.0"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  //"org.apache.spark" %% "spark-hive" % sparkVersion,
  //"org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion,
  
  "mysql" % "mysql-connector-java" % "5.1.6",
  //"com.databricks"  %% "spark-xml" % "0.4.1",
  //"com.databricks"  %% "spark-avro" % "3.2.0", 
  //"org.apache.kafka" %% "kafka" % "2.1.1",
  //"org.apache.kafka" % "kafka-clients" % "1.0.0"
  //"com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.9.8",
  //"com.fasterxml.jackson.core" %% "jackson-databind" % "2.9.8"
  
)

//dependencyOverrides += "com.fasterxml.jackson.module" % "jackson-module-scala_2.11" % "2.9.8"
//dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind_2.11" % "2.9.8"
//resolvers += Classpaths.typesafeResolvercd 
//addSbtPlugin("com.typesafe.startscript" % "xsbt-start-script-plugin" % "0.5.3")
