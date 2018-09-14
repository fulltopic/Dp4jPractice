package rl.dqn.supervised.test

import java.io.{File, FileInputStream, PrintWriter}

import scala.io.Source

object ExtractStatsFile extends App{
  val fileName = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/supervised/evalstats1532843164953.txt"
  val dstFileName = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/supervised/evalstats1532843164953_extract.txt"
  val lines = Source.fromFile(fileName).getLines
  val fileWriter = new PrintWriter(new File(dstFileName))

  for (line <- lines if (!line.trim.startsWith("Examples")) && line.trim.length > 0) {
//    println("To write " + line)
    fileWriter.write(line + "\n")
  }

  fileWriter.close()
}
