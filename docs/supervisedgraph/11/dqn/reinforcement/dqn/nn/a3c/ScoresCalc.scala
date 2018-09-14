package rl.dqn.reinforcement.dqn.nn.a3c


import scala.io.Source

object ScoresCalc extends App{
  val logFileName = "/home/zf/workspaces/workspace_java/mjpratice/docs/a3c/4/debug2.log"
//  val scoreFileName = "/home/zf/workspaces/workspace_java/mjpratice/docs/a3c/3/scores.log"

  def extractScores(): Unit = {
    val lines = Source.fromFile(logFileName).getLines().filter(line => line.contains("TenhouTcpConnection")).filter(line => (line.contains("AGARI") || line.contains("RYUUKYOKU")))
    var count: Int = 0
    var total: Double = 0.0
    while (lines.hasNext) {
      val line = lines.next()
//      println(line)
      //      if (line.contains("AGARI")) {
        val index = line.indexOf("sc=")
        val scStr = line.drop(index)
//      println(scStr)
        val scIndex = scStr.indexOf("\"", 10)
//        println(scIndex)
        val str = scStr.take(scIndex + 1).trim
//        println(str)

        val scores = str.drop("sc=\"".length).dropRight("\"".length).trim.split(",").map(_.trim).map(_.toInt)
      val reward = scores(1)
        println(count + ", " + reward)

//      total += reward
//      val average = total / (count + 1)
//      println(count + ", " + average)

      count += 1

      //      }
    }
  }

  extractScores()
}
