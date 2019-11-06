package dp4jpractice.org.mjdrl.config

import scala.io.Source

sealed class DqnSettingsClass protected (ConfigFileName: String) {
  val LstmSeqCap: Int = 50



  private val keySet = Set[String](
    "ReplayBufferSize",
    "GameEndWaitTime",
    "KALimit",
    "LNLimit",
    "ServerIP",
    "Port",
    "UserName",
    "StartEpsilon",
    "MinEpsilon",
    "EpsilonScaleRate",
    "AggressiveStealValue",
    "UpdateTillSave",
    "RewardFileName",
    "ModelFileName",
    "ModelLstmFileName",
    "LoadModelFileName",
    "LoadModelBiLstmFileName",
    "LoadModelLstmFileName",
    "LoadNNModelFileName",
    "A3CServerIP",
    "A3CServerPort",
    "A3CThreadNum",
    "IsTest",
    "IsPrivateLobby",
    "Local2LayerLstmModelFile"
  ).toArray


  println("The config file " + ConfigFileName)
  private val fileSource = Source.fromInputStream(getClass.getClassLoader.getResourceAsStream(ConfigFileName))
  private val lines = try fileSource.getLines().filter(line => !line.startsWith("#")).filter(line => line.trim.length > 0).toArray finally fileSource.close()
  private val paramMap = scala.collection.mutable.Map[String, String]()

  for (i <- lines.indices) {
    val line = lines(i)
    val pair = line.split("=").map(_.trim)

    if (pair.length == 2) {
      val key = pair(0)
      val value = pair(1)

      paramMap(key) = value
    }else {
      if (line.trim.length > 0) {
        println("Invalid format " + line)
      }
    }
  }

  val TestNames: Array[String] = paramMap.getOrElse("UserNames", "").split(",").map(_.trim)

  val IsTest: Boolean = paramMap.getOrElse("IsTest", "false").toBoolean
  val IsPrivateLobby: Boolean = paramMap.getOrElse("IsPrivateLobby", "false").toBoolean
  val Local2LayerLstmModelFile: String = paramMap.getOrElse("Local2LayerLstmModelFile", "")

  val ReplayBufferSize: Int = paramMap.getOrElse("ReplayBufferSize", "50").toInt

  val GameEndWaitTime: Int = paramMap.getOrElse("GameEndWaitTime", "0").toInt
  val KALimit: Int = paramMap.getOrElse("KALimit", "Int.MaxValue").toInt
  val LNLimit: Int = paramMap.getOrElse("LNLimit", "Int.MaxValue").toInt

  val ServerIP: String = paramMap.getOrElse("ServerIP", "")
  val Port: Int = paramMap.getOrElse("Port", "10080").toInt

  val StartEpsilon: Float = paramMap.getOrElse("StartEpsilon", "1").toFloat
  val MinEpsilon: Float = paramMap.getOrElse("MinEpsilon", "0").toFloat
  val EpsilonScaleRate: Float = paramMap.getOrElse("EpsilonScaleRate", "0.1").toFloat
  val AggressiveStealValue: Float = paramMap.getOrElse("AggressiveStealValue", "0.1").toFloat
  val UpdateTillSave: Int = paramMap.getOrElse("UpdateTillSave", "20").toInt


  val RewardFileName: String = paramMap.getOrElse("RewardFileName", "")
  val ModelFileName: String = paramMap.getOrElse("ModelFileName", "")
  val ModelLstmFileName: String = paramMap.getOrElse("ModelLstmFileName", "")
  val LoadModelFileName: String = paramMap.getOrElse("LoadModelFileName", "")
  val LoadModelBiLstmFileName: String = paramMap.getOrElse("LoadModelBiLstmFileName", "")
  val LoadModelLstmFileName: String = paramMap.getOrElse("LoadModelLstmFileName", "")
  val LoadNNModelFileName: String = paramMap.getOrElse("LoadNNModelFileName", "")

  val A3CServerIP: String = paramMap.getOrElse("A3CServerIP", "")
  val A3CServerPort: Int = paramMap.getOrElse("A3CServerPort", "52222").toInt
  val A3CThreadNum: Int = paramMap.getOrElse("A3CThreadNum", "0").toInt

  val ActionMatchReward: Double = 0.0
  val ActionMisMatchReward: Double = 0.0
  val A3CBaseLine: Double = -3.06
  val InvalidActionPenalty: Double = -1.0
  val A3CWinReward: Double = 1
  val A3CTiePenalty: Double = -0.10

  def printConfig(): Unit = {
    println("IsTest: " + IsTest)
    println("IsPrivateLobby: " + IsPrivateLobby)
    println("ReplayBufferSize: " + ReplayBufferSize)
    println("GameEndWaitTime: " + GameEndWaitTime)
    println("KALimit: " + KALimit)
    println("LNLimit: " + LNLimit)
    println("ServerIP: " + ServerIP)
    println("Port: " + Port)
    println("UserName: " + TestNames.mkString(","))
    println("StartEpsilon: " + StartEpsilon)
    println("MinEpsilon: " + MinEpsilon)
    println("EpsilonScaleRate: " + EpsilonScaleRate)
    println("AggressiveStealValue: " + AggressiveStealValue)
    println("UpdateTillSave: " + UpdateTillSave)
    println("RewardFileName: " + RewardFileName)
    println("ModelFileName: " + ModelFileName)
    println("ModelLstmFileName: " + ModelLstmFileName)
    println("LoadModelFileName: " + LoadModelFileName)
    println("LoadModelBiLstmFileName: " + LoadModelBiLstmFileName)
    println("LoadModelLstmFileName: " + LoadModelLstmFileName)
    println("LoadNNModelFileName: " + LoadNNModelFileName)
    println("A3CServerIP: " + A3CServerIP)
    println("A3CServerPort: " + A3CServerPort)
    println("A3CThreadNum: " + A3CThreadNum)
  }

}

object DqnSettings extends DqnSettingsClass ("DqnSetting_lstm_a3c.txt"){}