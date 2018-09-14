package rl.dqn.supervised.space

import java.util.Random

import org.deeplearning4j.rl4j.space.ActionSpace
import rl.dqn.reinforcement.dqn.config.Supervised._


class MjActionSpace extends ActionSpace[Integer] {
  private[this] val Size = 34 + 3 + 1 + 1 + 1 //Drop tiles + (chow, pong, 4) + (ricchi) + hou + noop

  private var rd = new Random()
  private var validActionSize = 1 //noop
  private[this] val validActions = new Array[Int](Size)

  def randomAction(): Integer = rd.nextInt(Size)

  def setSeed(seed: Int) = {
    rd = new Random(seed)
  }

  def encode(action: Integer): Object = action

  def noOp(): Integer = NOOP

  def getSize(): Int = Size
}