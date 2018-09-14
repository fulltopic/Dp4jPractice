package rl.dqn.reinforcement.dqn.nn.datapocess

import java.io.File

import akka.event.slf4j.Logger
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.concurrent.{Await, Future}
import scala.util.{Failure, Random, Success}
import rl.dqn.reinforcement.dqn.config.Supervised._

import scala.concurrent.duration.Duration

class TenhouXmlFectcher(xmlParentDir: String, toRandom: Boolean = false) extends BaseDataFetcher {
  val logger = Logger("TenhouXmlFetcher")
  
  val itemRandom: Random = new Random(System.currentTimeMillis())
  val dirs = new File(xmlParentDir).listFiles().filter(_.isDirectory).toList
  var files = List.empty[String]
  dirs.foreach(dir => {
    val dirFiles = dir.listFiles().filter(_.isFile).map(_.getAbsolutePath)
    files = files ++ dirFiles
  })
  val fileOrderList: List[Int] = (0 until files.length).toList
  var fileOrder = Random.shuffle(fileOrderList).toArray
  var fileIndex: Int = 0
  var doneFileNum: Int = 0
  var currentReaders = Array.empty[TenhouXmlFileReader]
  var toLoad: Boolean = true

  var currentDatas = List.empty[(INDArray, Int)]


  override def reset(): Unit = {
    fileIndex = 0
    doneFileNum = 0
    currentReaders = Array.empty[TenhouXmlFileReader]
    currentDatas = List.empty[(INDArray, Int)]
    toLoad = true
  }

  override def hasMore(): Boolean = {
    files.length - fileIndex > 0
  }

  import scala.concurrent.ExecutionContext.Implicits.global
  def getLoadFuture(reader: TenhouXmlFileReader): Future[List[(INDArray, Int)]] = Future {
    reader.readFile()
  }

  def loadNewFiles(numExamples: Int): Unit = {
    val fileNum = math.min(numExamples, files.length - fileIndex)

    currentDatas = List.empty[(INDArray, Int)]
    currentReaders = new Array[TenhouXmlFileReader](numExamples)
    for (i <- 0 until fileNum) {
      currentReaders(i) = new TenhouXmlFileReader(files(fileOrder(fileIndex)))
      fileIndex += 1
    }

    for (i <- fileNum until currentReaders.length) {
      val randomIndex = Random.nextInt(files.length)
      currentReaders(i) = new TenhouXmlFileReader(files(randomIndex))
      fileIndex += 1
      // Let's overflow
    }

    val futures = for (reader <- currentReaders) yield getLoadFuture(reader)
    var count: Int = 0

    futures.foreach(future => {
      future onComplete {
        case Success(playerData) =>
          currentDatas = currentDatas ++ playerData
          count += 1
        case Failure(_) =>
          count += 1
      }
    })

    futures.foreach(future => Await.ready(future, Duration.Inf))

    while (count != numExamples) {
      Thread.sleep(100)
    }

    if (toRandom) {
      currentDatas = Random.shuffle(currentDatas)
    }

    toLoad = false
    logger.debug("Get all readers loaded "  + count)
  }

  val validNumExamples = Array[Int](16, 32, 64, 128, 256).toSet
  override def fetch(numExamples: Int): Unit = {
    if (!validNumExamples.contains(numExamples)) {
      throw new Exception("Invalid numExamples " + numExamples + ", should be " + validNumExamples.mkString(","))
    }

    if (toLoad && hasMore()) {
      loadNewFiles(numExamples)
    }

//    val pairs = currentReaders.map(_.getNext())

    val featureData = Nd4j.create(numExamples, PeerStateLen)
    val labelData = Nd4j.create(numExamples, ActionLenWoAccept)
//    logger.debug("-----------------------------------------------> " + featureData.shape().mkString(", "))
//    logger.debug("-----------------------------------------------> " + labelData.shape().mkString(", "))

    for (i <- 0 until numExamples) {
      var pair: (INDArray, Int) = null

      if (toRandom) {
        currentDatas match {
          case head :: tail =>
            pair = head
            currentDatas = tail
          case Nil =>
            pair = (Nd4j.zeros(PeerStateLen), NOOPWoAccept)
            toLoad = true
        }
      }else {
        pair = currentReaders(i).getNext()
      }
      val feature = pair._1
      val action = pair._2

      featureData.putRow(i, feature)
//      val actionData = Nd4j.create(ActionLenWoAccept)
////      logger.debug("-----------------------------------------------> " + actionData.shape().mkString(", "))
//      for (j <- 0 until ActionLenWoAccept) {
//         actionData.putScalar(i, 0.0)
//      }
      val actionData = Nd4j.zeros(ActionLenWoAccept)
      actionData.putScalar(action, 1.0)
      labelData.putRow(i, actionData)
    }

    toLoad = currentReaders.map(_.getSize()).count(_ > 0) == 0

    curr = new DataSet(featureData, labelData)
  }
}
