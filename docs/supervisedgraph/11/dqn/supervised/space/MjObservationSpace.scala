package rl.dqn.supervised.space

import org.deeplearning4j.rl4j.space.ObservationSpace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class MjObservationSpace extends ObservationSpace[Int]{
  val SpaceSize = 13

  def getName(): String = "MjObservationSpace"

  def getShape(): Array[Int] = Array[Int](1, SpaceSize)

  def getLow(): INDArray = Nd4j.zeros(1, SpaceSize)

  def getHigh: INDArray = Nd4j.ones(1, SpaceSize).muli(Double.MaxValue)
}
