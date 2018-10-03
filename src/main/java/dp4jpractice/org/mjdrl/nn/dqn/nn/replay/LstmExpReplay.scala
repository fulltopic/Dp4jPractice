package dp4jpractice.org.mjdrl.nn.dqn.nn.replay

import java.util

import dp4jpractice.org.mjdrl.config.DqnSettings

class LstmExpReplay(){
  private val transList = new java.util.ArrayList[LstmTransition](DqnSettings.ReplayBufferSize)

  def getBatch(size: Int): java.util.ArrayList[LstmTransition] = {
    transList
  }

  def getBatch: util.ArrayList[LstmTransition] = getBatch(transList.size())

  def store(transition: LstmTransition): Unit = {
    transList.add(transition)
  }

  def clear(): Unit = transList.clear()
}
