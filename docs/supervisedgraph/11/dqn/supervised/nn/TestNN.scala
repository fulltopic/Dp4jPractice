package rl.dqn.supervised.nn

import akka.event.slf4j.Logger
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.rl4j.network.dqn.{DQN, DQNFactoryStdDense}
import org.nd4j.linalg.learning.config.{Adam, Nesterovs}
import rl.dqn.reinforcement.dqn.config.Supervised._

object TestNN {
  private val logger = Logger("TestNN")
  
  def createConf(numLayer: Int, numHiddenNodes: Int, learningRate: Double, l2: Double): DQNFactoryStdDense.Configuration = {
    DQNFactoryStdDense.Configuration.builder()
      .learningRate(learningRate)
      .numLayer(numLayer)
      .numHiddenNodes(numHiddenNodes)
      .l2(l2)
      .updater(new Nesterovs()) // And try Adam also
      .build()
  }

  def createNN(conf: DQNFactoryStdDense.Configuration, supervised: Boolean, inputWidth: Int, outputNum: Int): MultiLayerNetwork = {
    val inputNums = Array[Int](1, inputWidth)
    val outputNums = outputNum

    if (supervised) {
      DenseNN.createSupervisedNN(conf, inputNums, outputNums)
    } else {
      DenseNN.createDqnNN(conf, inputNums, outputNums)
    }
  }

  def testSetParams(conf: DQNFactoryStdDense.Configuration): Unit = {
    val supervisedModel = createNN(conf, true, PeerStateLen, ActionLen)
    val dqnModel = createNN(conf, false, PeerStateLen, ActionLen)

    val supervisedParams = supervisedModel.paramTable
    logger.info(supervisedParams.getClass.toString)

    logger.info(supervisedParams.size() + "")
    val supervisedKeys = supervisedParams.keySet().toArray
    supervisedKeys.foreach(key => logger.info(key + ""))

    val dqnParams = dqnModel.paramTable
    logger.info(dqnParams.size() + "")
    val dqnKeys = dqnParams.keySet().toArray
    dqnKeys.foreach(key => logger.info(key + ""))


    dqnModel.setParamTable(supervisedParams)
    val newDqnParams = dqnModel.paramTable
    supervisedKeys.foreach(key => {
      logger.info(supervisedParams.get(key).eq(newDqnParams.get(key)).sumNumber().toString)
    })
  }


  def main(args: Array[String]): Unit = {
    val conf: DQNFactoryStdDense.Configuration = createConf(2, 20, 0.01, 0.0005)
    testSetParams(conf)
  }
}
