package dp4jpractice.org.mjdrl.nn.dqn.utils

import java.io.{File, PrintWriter}

import dp4jpractice.org.mjdrl.config.DqnSettings

object RewardLogger {
  val rewardFileName = DqnSettings.RewardFileName + System.currentTimeMillis() + ".txt"
  val rewardWriter: PrintWriter = new PrintWriter(new File(rewardFileName))

//  def setFileName(fileName: String): Unit = {
//    rewardFileName = fileName + System.currentTimeMillis() + ".txt"
//
//    initLogger()
//  }
//
//  private def initLogger(): Unit = {
//    if (rewardWriter != null) {
//      rewardWriter.close()
//    }
//
//    rewardWriter = new PrintWriter(new File(rewardFileName))
//  }

  def writeLog(msg: String): Unit = {
    rewardWriter.print(msg + "\n")
    rewardWriter.flush()
  }
}
