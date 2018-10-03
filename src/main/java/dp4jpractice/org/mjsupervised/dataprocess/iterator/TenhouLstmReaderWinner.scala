package dp4jpractice.org.mjsupervised.dataprocess.iterator

import akka.event.slf4j.Logger
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import tenhouclient.utils.TenhouConsts._
import tenhouclient.impl.ImplConsts._
//import dp4jpractice.org.mjsupervised.utils.TenhouConsts._
//import dp4jpractice.org.mjsupervised.utils.ImplConsts._



import scala.xml.Node

class TenhouLstmReaderWinner(fileName: String) extends TenhouLstmReader (fileName) {
  private val logger = Logger("TenhouLstmReaderWinner")
  private var tmpDatas = Array.fill[List[(INDArray, Int)]](PlayerNum)(List.empty[(INDArray, Int)])


  override def readScene(sceneNode: Node): Unit = {
    super.readScene(sceneNode)

    datas.foreach(data => {
      sceneDatas = sceneDatas :+ data
    })

    datas = Array.fill[List[(INDArray, Int)]](PlayerNum)(List.empty[(INDArray, Int)])
  }

  override protected def createDataPair(action: Int, who: Int): Unit = {
    val state = currentStats(who)

    val indState = Nd4j.create(state)
//    val indState = Nd4j.zeros(state.length)
//    for (i <- state.indices) {
//      val v = state(i)
//      val remainV = v - v.toInt.toDouble
//      val nV = ((v.toInt & ExtraValueFlag).toDouble + remainV) / NumPerTile.toDouble
//      indState.putScalar(i, nV)
//    }

    //    datas(who) = datas(who) :+ (indState, action)
    tmpDatas(who) = tmpDatas(who) :+ (indState, action)
  }

  def updateData(winner: Int): Unit = {
    datas(winner) = datas(winner) ++ tmpDatas(winner)
  }

  def clearTmpData(): Unit = {
    tmpDatas = Array.fill[List[(INDArray, Int)]](PlayerNum)(List.empty[(INDArray, Int)])
  }

  //TODO: create transition
  override def parseTerminal(node: Node): Unit = {
    val rewards = node.attribute("sc").map(_.text.split(",").map(_.toInt).zipWithIndex.filter(_._2 % 2 == 1).map(_._1))
      .getOrElse(Array.fill[Int](PlayerNum)(0))
    val who = node.attribute("who").map(_.head.text.toInt).getOrElse(-1)
    val winners = scala.collection.mutable.Set.empty[Int]
    for (i <- rewards.indices) {
      if (i == who) {
        val machi = node.attribute("machi").map(_.head.text.toInt).getOrElse(-1)
        val state = currentStats(i).clone()
        val fromWho = node.attribute("fromWho").map(_.head.text.toInt).getOrElse(who)
        if (fromWho != who && machi >= 0) {
          acceptTile(currentStats(who), machi)
        }

        createDataPair(RonWoAccept, i)
      }else {
        createDataPair(NOOPWoAccept, i)
      }

      if (rewards(i) > 0) {
        winners.add(i)
      }
    }

    for (index <- winners) {
      updateData((index))
    }
    clearTmpData()
  }

}
