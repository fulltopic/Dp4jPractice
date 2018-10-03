package dp4jpractice.org.mjdrl.nn.dqn.nn.lstm

import akka.event.slf4j.Logger
import dp4jpractice.org.mjdrl.config.DqnSettings
import dp4jpractice.org.mjdrl.nn.dqn.nn.DqnUtils
import dp4jpractice.org.mjdrl.nn.dqn.utils.RewardLogger
import org.deeplearning4j.rl4j.learning.StepCountable
import org.deeplearning4j.rl4j.network.NeuralNet
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.policy.EpsGreedy
import org.nd4j.linalg.api.ndarray.INDArray
import tenhouclient.impl.MessageParseUtilsImpl
import tenhouclient.impl.mdp.{TenhouArray, TenhouIntegerActionSpace}

import scala.util.Random

class TenhouLstmEpsGreedy(val dqn: IDQN[_], val updateStart: Int, val epsilonNbStep: Int, val minEpsilon: Float, val learning: StepCountable)
  extends EpsGreedy[TenhouArray, Integer, TenhouIntegerActionSpace](null, null, updateStart, epsilonNbStep, null, minEpsilon, learning) {

  val logger = Logger("TenhouLstmEpsGreedy")

  val rd: Random = new Random()
  val startEpsilon: Float = DqnSettings.StartEpsilon


//  override def getNeuralNet: NeuralNet[_ <: AnyRef] = null

  override def nextAction(rawInput: INDArray): Integer = {
    println("============================================> Next action")
    val ep = getEpsilon()
    //    if (learning.getStepCounter % 500 == 1) {
    println("EP " + ep + " " + learning.getStepCounter)
    //    RewardLogger.writeLog("EP " + ep)
    //    }

    val normInput = DqnUtils.getNNInput(rawInput);
    var maxAction: Int = 0

    if (rd.nextFloat() > ep) {
      val fOutput = dqn.output(normInput)

      val pair = MessageParseUtilsImpl.getLegalQAction(rawInput, fOutput)
      val maxQ = pair.getFirst
      maxAction = pair.getSecond
      RewardLogger.writeLog("maxLegalQ: " + maxQ)
      println("maxLegalQ: " + maxQ)

    }else {
      println("Randomly chosen")
      val actions = MessageParseUtilsImpl.getAvailableActions(rawInput)

      println(actions.mkString(", "))
      maxAction = actions(rd.nextInt(actions.size))
      println(maxAction)
    }

    maxAction
  }

//  def nextAction(rawInput: INDArray, input: INDArray): Integer = {
//    println("============================================> Next action 2")
//    val ep = getEpsilon
//    println("EP " + ep + " " + learning.getStepCounter)
//
//    var maxAction: Int = 0
//
//    if (rd.nextFloat() > ep) {
//      val output = dqn.output(input)
//
//      val pair = MessageParseUtils.getLegalQAction(rawInput, output)
//      val maxQ = pair.getFirst
//      maxAction = pair.getSecond
//      RewardLogger.writeLog("maxLegalQ: " + maxQ)
//      println("maxLegalQ: " + maxQ)
//    }else {
//      println("Randomly chosen")
//      val actions = MessageParseUtils.getAvailableActions(rawInput)
//
//      println(actions.mkString(", "))
//      maxAction = actions(rd.nextInt(actions.size))
//      println(maxAction)
//    }
//
//    maxAction
//  }

  def nextAction(rawInput: INDArray, output: INDArray): Integer = {
    println("============================================> Next action 2")
    val ep = getEpsilon
    println("EP " + ep + " " + learning.getStepCounter)

    var maxAction: Int = 0

    if (rd.nextFloat() > ep) {
      val pair = MessageParseUtilsImpl.getLegalQAction(rawInput, output)
      val maxQ = pair.getFirst
      maxAction = pair.getSecond
      RewardLogger.writeLog("maxLegalQ: " + maxQ)
      println("maxLegalQ: " + maxQ)
    }else {
      println("Randomly chosen")
      val actions = MessageParseUtilsImpl.getAvailableActions(rawInput)

      println(actions.mkString(", "))
      maxAction = actions(rd.nextInt(actions.size))
      println(maxAction)
    }

    maxAction
  }

  override def getEpsilon: Float = 0.0f
  //    (0.05).toFloat
}
