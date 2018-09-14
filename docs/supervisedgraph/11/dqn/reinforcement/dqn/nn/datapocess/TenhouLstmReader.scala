package rl.dqn.reinforcement.dqn.nn.datapocess

import akka.event.slf4j.Logger
import org.nd4j.linalg.api.ndarray.INDArray
import rl.dqn.reinforcement.dqn.config.Supervised._
import scala.xml.Node

class TenhouLstmReader(pathName: String, winner: Boolean = false) extends TenhouXmlFileReader(pathName) {
  private val logger = Logger("TenhouLstmReader")

  protected var sceneDatas = List.empty[List[(INDArray, Int)]]
  var hasLoad: Boolean = false

  override def readScene(sceneNode: Node): Unit = {
    super.readScene(sceneNode)

    datas.foreach(data => {
      sceneDatas = sceneDatas :+ data
    })

    datas = Array.fill[List[(INDArray, Int)]](PlayerNum)(List.empty[(INDArray, Int)])
  }

  def getData(): List[List[(INDArray, Int)]] = {
    if (!hasLoad) {
      readFile()
      hasLoad = true
    }

    sceneDatas
  }
}
