package rl.dqn.reinforcement.dqn.nn.a3c

import org.nd4j.linalg.factory.Nd4j
import rl.dqn.reinforcement.dqn.client.MessageParseUtils
import rl.dqn.reinforcement.dqn.client.MessageParseUtils.getAvailableActions

object TestPolicy extends App{
  val outputValue = Array[Double](0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.33,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.16,  0.05,  0.01,  0.20,  0.25,  0.00)
  val inputValue = Array[Double](192.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  2.00,  0.00,  1.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  1.00,  0.00,  0.00,  192.00,  1.00,  3.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  3.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00)
  val output = Nd4j.create(outputValue)
  val input = Nd4j.create(inputValue)



  //  Pair<Double, Integer>
  val legalActions = getAvailableActions(input).toSet
  var maxQ: Double = Double.MinValue
  var action: Int = 0
  var tmpQ: Double = 0.0
  var tmpA: Int = 0

  for (i <- 0 until 42) {
    if (tmpQ < output.getDouble(i)) {
      tmpQ = output.getDouble(i)
      tmpA = i
    }
    if (legalActions.contains(i)) {
      val q = output.getDouble(i)
//      println(q + " ? " + maxQ)
      if (q > maxQ) {
        maxQ = q
        action = i
      }
    }
  }

  val tiles = for (i <- 0 until 34 if inputValue(i) > 0) yield  i
  println(tiles.mkString(","))

  println(maxQ)
  println(action)
  println(tmpQ)
  println(tmpA)

  println(input.getInt(tmpA))
  println(legalActions.mkString(","))
}
