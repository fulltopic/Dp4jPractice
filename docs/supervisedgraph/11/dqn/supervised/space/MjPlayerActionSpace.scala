package rl.dqn.supervised.space

import java.util.Random

class MjPlayerActionSpace extends MjActionSpace{
  private[this] val Size = 34 + 3 + 1 + 1 + 1 //Drop tiles + (chow, pong, 4) + (ricchi) + hou + noop

  private[this] val NOOP = 0

  private var rd = new Random()
  private var validActionSize = 1 //noop
  private[this] val validActions = new Array[Int](Size)

  override def randomAction(): Integer = {
    val action = rd.nextInt(validActionSize)
    validActions.toList.filter(i => i > 0)(action)
  }
}
