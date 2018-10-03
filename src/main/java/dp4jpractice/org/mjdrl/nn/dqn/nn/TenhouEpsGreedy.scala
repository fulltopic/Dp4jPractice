package dp4jpractice.org.mjdrl.nn.dqn.nn

import akka.event.slf4j.Logger
import dp4jpractice.org.mjdrl.config.DqnSettings
import dp4jpractice.org.mjdrl.nn.dqn.utils.RewardLogger
import org.deeplearning4j.rl4j.learning.StepCountable
import org.deeplearning4j.rl4j.network.NeuralNet
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.policy.EpsGreedy
import org.nd4j.linalg.api.ndarray.INDArray
import tenhouclient.impl.MessageParseUtilsImpl
import tenhouclient.impl.mdp.{TenhouArray, TenhouIntegerActionSpace}

import scala.util.Random

//TODO: What's this space for?
class TenhouEpsGreedy(val dqn: IDQN[_], val updateStart: Int, val epsilonNbStep: Int, val minEpsilon: Float, val learning: StepCountable)
  extends EpsGreedy[TenhouArray, Integer, TenhouIntegerActionSpace](null, null, updateStart, epsilonNbStep, null, minEpsilon, learning){

  val logger = Logger("TenhouEpsGreedy")

  val rd: Random = new Random()
  val startEpsilon: Float = DqnSettings.StartEpsilon


//  override def getNeuralNet: NeuralNet[_ <: AnyRef] = {null}

//  def getEpsilon(): Float = math.min(1f, Math.max(minEpsilon, 1f - (learning.getStepCounter - updateStart) * 1f / epsilonNbStep))

  override def nextAction(rawInput: INDArray): Integer = {
    logger.debug("============================================> Next action")
    val ep = getEpsilon()
//    if (learning.getStepCounter % 500 == 1) {
      logger.debug("EP " + ep + " " + learning.getStepCounter)
//    RewardLogger.writeLog("EP " + ep)
//    }

    val normInput = DqnUtils.getNNInput(rawInput);
    var maxAction: Int = 0

    if (rd.nextFloat() > ep) {
      val output = dqn.output(normInput)
      val pair = MessageParseUtilsImpl.getLegalQAction(rawInput, output)
      val maxQ = pair.getFirst
      maxAction = pair.getSecond
//      val actions = MessageParseUtils.getAvailableActions(rawInput).toSet
//      val legalOutput = Nd4j.zeros(ActionLenWoAccept)
//      for (i <- 0 until ActionLenWoAccept) {
//        if (!actions.contains(i)){
//          legalOutput.putScalar(i, Double.MinValue)
//        }else {
//          if (i >= TileNum && i != NOOPWoAccept) {
//            legalOutput.putScalar(i, output.getDouble(i) + DqnSettings.AggressiveStealValue)
//          }else {
//            legalOutput.putScalar(i, output.getDouble(i))
//          }
//        }
//      }
//
//      println(legalOutput)
//      maxAction = Learning.getMaxAction(legalOutput)
//      println(maxAction)
      RewardLogger.writeLog("maxLegalQ: " + maxQ)
      logger.debug("maxLegalQ: " + maxQ)

    }else {
      logger.debug("Randomly chosen")
      val actions = MessageParseUtilsImpl.getAvailableActions(rawInput)

      logger.debug(actions.mkString(", "))
      maxAction = actions(rd.nextInt(actions.size))
      logger.debug(maxAction + "")
    }

    maxAction
  }

  override def getEpsilon: Float = Math.min(1f, Math.max(minEpsilon, startEpsilon - (learning.getStepCounter - updateStart) * DqnSettings.EpsilonScaleRate / epsilonNbStep))
//    (0.05).toFloat
}
