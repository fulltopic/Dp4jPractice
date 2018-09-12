package rl.dqn.reinforcement.dqn.config

import rl.dqn.reinforcement.dqn.{TenhouServerIp, TenhouUserName}

import scala.io.Source

object DqnSettings {
  /*
  For All
   */
  private var ConfigFileName = "/home/zf/workspaces/workspace_java/mjconv4/src/main/java/rl/dqn/reinforcement/dqn/config/DqnSetting_lstm_a3c.txt"
//  private var ConfigFileName = "/home/zf/workspaces/workspace_java/mjconv4/src/main/java/rl/dqn/reinforcement/dqn/config/DqnSetting_a3c.txt"

  //  private val ConfigFileName = "/home/ec2-user/mjconv4/src/main/java/rl/dqn/reinforcement/dqn/config/DqnSetting_cloud.txt"
//  private var ConfigFileName = "/root/DqnSetting.txt"
//  def setConfigFile(fileName: String): Unit = {
//    ConfigFileName = fileName
//  }

  val TestName0 = "ID10632D03-BZRFdSab" //fullrand
  val TestName1 = "ID52C61588-fBT9HfYQ" //fullaca
  val TestName2 = "ID2E336B42-BfAf8H9F" //fullcnn
  val TestName3 = "ID5F0B11CB-PU9nHfDM" //fullnn
  val TestName4 = "ID48547DBC-bBYmSGLD" //fulllstm
  val TestName5 = "ID14BA4AD0-38gaDVBG" //fulllsta
  val TestName6 = "ID59DC5036-YQSVMgX6" //deprecated
  val TestName7 = "ID682C234B-PbdLRGV4" //fullacb
  val TestName8 = "ID612E00B1-8eVaBXdR" //fullacc
  val TestName9 = "ID78A24927-c5bhBcQb" //fullacd


//  val TestNames = Array[String] (TestName0, TestName1, TestName2, TestName3, TestName4, TestName5, TestName6)

  val LstmSeqCap: Int = 50

  val Local2LayerLstmModelFile = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/supervised/models/model_lstm_2layer.xml/"

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
    "A3CThreadNum"
  ).toArray


  private val lines = Source.fromFile(ConfigFileName).getLines().filter(line => !line.startsWith("#")).filter(line => line.trim.length > 0).toArray
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


  val ReplayBufferSize: Int = paramMap.getOrElse("ReplayBufferSize", "50").toInt

  val GameEndWaitTime: Int = paramMap.getOrElse("GameEndWaitTime", "0").toInt
  val KALimit: Int = paramMap.getOrElse("KALimit", "Int.MaxValue").toInt
  val LNLimit: Int = paramMap.getOrElse("LNLimit", "Int.MaxValue").toInt

  val ServerIP: String = paramMap.getOrElse("ServerIP", "")
  val Port: Int = paramMap.getOrElse("Port", "10080").toInt

//  private val userNameId = paramMap.getOrElse("UserNameId", "0").toInt
//  val UserName: String = TestNames(userNameId)
  private val UserNameStr = paramMap.getOrElse("UserNames", "")
  val TestNames = UserNameStr.split(",").map(_.trim).toArray

  val StartEpsilon: Float = paramMap.getOrElse("StartEpsilon", "1").toFloat
  val MinEpsilon: Float = paramMap.getOrElse("MinEpsilon", "0").toFloat
  val EpsilonScaleRate: Float = paramMap.getOrElse("EpsilonScaleRate", "0.1").toFloat
  val AggressiveStealValue: Float = paramMap.getOrElse("AggressiveStealValue", "0.1").toFloat
  val UpdateTillSave: Int = paramMap.getOrElse("UpdateTillSave", "50").toInt


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

  def printConfig(): Unit = {
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
