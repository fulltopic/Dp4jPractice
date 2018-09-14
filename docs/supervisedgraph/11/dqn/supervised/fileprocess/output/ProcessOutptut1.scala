package rl.dqn.supervised.fileprocess.output

import java.io.{BufferedWriter, FileWriter}

import scala.io.Source

object ProcessOutput1 extends App {
  val fileName = "output20"
  val txt = ".txt"
  val csv = ".csv"
  val filePath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/player/"
  val deli = ","
  val scores = Array[String]("Accuracy:", "Precision:", "Recall:", "F1 Score:")


  val output = new BufferedWriter(new FileWriter(filePath + fileName + csv))
  var params = ""
  var seq:Int = 1

  for (line <- Source.fromFile(filePath + fileName + txt).getLines) {
    if (line.contains("Train a DQRN")) {
      val endIndex = line.indexOf('=')
      params = line.take(endIndex)
      seq = 1
//      output.write("\n" + params + ",")
    }
    if (scores.foldLeft[Boolean](false)((r, s) => { r || line.contains(s)})) {
      var value = line.split(":")(1).trim
      if (line.contains("Accuracy")) {
        output.write("\n" + params + ", " + seq + ", ")
        seq += 1
      }

      var excluded: Int = 0
      if (line.contains("(")) {
        val parts = value.split("\\(")
        value = parts(0)
        excluded = parts(1).split(" ")(0).trim.toInt
      }

      output.write(value + deli + excluded + deli)
    }

  }

  output.close()
}
