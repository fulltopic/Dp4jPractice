package rl.dqn.reinforcement.dqn.nn.datapocess

import java.io.File

import akka.event.slf4j.Logger
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.concurrent.{Await, Future}
import scala.util.{Failure, Random, Success}
import rl.dqn.reinforcement.dqn.config.Supervised.{ActionLenWoAccept, NOOPWoAccept, PeerStateLen, TileNum}

import scala.concurrent.duration.Duration

class TenhouXmlCnnFetcher(xmlParentDir: String, outputLen: Int, winner: Boolean = false) extends BaseDataFetcher {
  val logger = Logger("TenhouXmlCnnFetcher")
  
  val itemRandom: Random = new Random(System.currentTimeMillis())
  private val dirs = new File(xmlParentDir).listFiles().filter(_.isDirectory).toList
  var files = List.empty[String]
  dirs.foreach(dir => {
    val dirFiles = dir.listFiles().filter(_.isFile).map(_.getAbsolutePath)
    files = files ++ dirFiles
  })
  val fileOrderList: List[Int] = (0 until files.length).toList
  private var fileOrder = Random.shuffle(fileOrderList).toArray
  var fileIndex: Int = 0
  var doneFileNum: Int = 0
  var currentReaders = Array.empty[TenhouXmlFileReader]
  var toLoad: Boolean = true

  var currentDatas = List.empty[(INDArray, Int)]

  val dataMap = scala.collection.mutable.HashMap.empty[Int, List[INDArray]]
  for (i <- 0 until ActionLenWoAccept) {
    dataMap(i) = List.empty[INDArray]
  }


  override def reset(): Unit = {
    fileOrder = Random.shuffle(fileOrderList).toArray

    fileIndex = 0
    doneFileNum = 0
    currentReaders = Array.empty[TenhouXmlFileReader]
    currentDatas = List.empty[(INDArray, Int)]
    toLoad = true

    dataIndex  = 0
    drainData = false
    for (i <- 0 until ActionLenWoAccept) {
      dataMap(i) = List.empty[INDArray]
    }
  }

  override def hasMore(): Boolean = {
    files.length - fileIndex > 0
  }

  import scala.concurrent.ExecutionContext.Implicits.global
  def getLoadFuture(reader: TenhouXmlFileReader): Future[List[(INDArray, Int)]] = Future {
    reader.readFile()
  }

  def loadNewFiles(numExamples: Int): Unit = {
    logger.debug("-----------------------------> Load " + files.length + ", " + fileIndex)
    val fileNum = math.min(numExamples, files.length - fileIndex)

    currentDatas = List.empty[(INDArray, Int)]
    currentReaders = new Array[TenhouXmlFileReader](numExamples)

    if (!winner) {
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
    }else {
      for (i <- 0 until fileNum) {
        currentReaders(i) = new TenhouXmlFileReaderWinner(files(fileOrder(fileIndex)))
        fileIndex += 1
      }

      for (i <- fileNum until currentReaders.length) {
        val randomIndex = Random.nextInt(files.length)
        currentReaders(i) = new TenhouXmlFileReaderWinner(files(randomIndex))
        fileIndex += 1
        // Let's overflow
      }
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

//    while (count != numExamples) {
//      Thread.sleep(100)
//    }

    currentDatas = Random.shuffle(currentDatas)

    while (!currentDatas.isEmpty) {
      val pair = currentDatas.head
      currentDatas = currentDatas.tail

      dataMap(pair._2) = dataMap(pair._2) :+ pair._1
    }

    toLoad = false
    logger.debug("Get all readers loaded "  + count)
  }

  val validNumExamples = Array[Int](16, 32, 64, 128, 256).toSet
  var fetchCount: Int = 0
  val caseNum = Array.fill[Int](outputLen + 1)(0)
  val candidates: Array[Int] = (0 until outputLen).toArray   //Array[Int](0, 1, 2)
  private var dataIndex: Int = 0
  private var drainData: Boolean = false

  override def fetch(numExamples: Int): Unit = {
    fetchCount += 1
    logger.debug("Fetch count " + fetchCount)

    if (!validNumExamples.contains(numExamples)) {
      throw new Exception("Invalid numExamples " + numExamples + ", should be " + validNumExamples.mkString(","))
    }

//    if (toLoad && hasMore()) {
//      loadNewFiles(numExamples)
//    }

    val featureData = Nd4j.create(numExamples, PeerStateLen)
    val labelData = Nd4j.create(numExamples, ActionLenWoAccept)


    for (i <- 0 until numExamples) {
      if (!drainData) {
        if (dataMap(candidates(dataIndex)).isEmpty) {
          if (hasMore()) {
            loadNewFiles(numExamples)

            if (dataMap(candidates(dataIndex)).isEmpty) {
              drainData = true
              fileIndex = files.length
            }
          }else {
            drainData = true
            fileIndex = files.length
          }
        }
      }


      if (drainData) {
        featureData.putRow(i, Nd4j.zeros(PeerStateLen))
        labelData.putRow(i, Nd4j.zeros(ActionLenWoAccept))
      }else {
        featureData.putRow(i, dataMap(candidates(dataIndex)).head)
        dataMap(candidates(dataIndex)) = dataMap(candidates(dataIndex)).tail
        val actionData = Nd4j.zeros(ActionLenWoAccept)
        actionData.putScalar(dataIndex, 1.0)
        labelData.putRow(i, actionData)
        dataIndex = (dataIndex + 1) % outputLen
      }

//      if (dataMap(candidates(dataIndex)).isEmpty) {
//        if (hasMore()) {
//          loadNewFiles(numExamples)
//        }
//
//        if (dataMap(candidates(dataIndex)).isEmpty) {
//          featureData.putRow(i, Nd4j.zeros(PeerStateLen))
//          labelData.putRow(i, Nd4j.zeros(outputLen))
//
//          fileIndex = files.length // hasMore = false
//        } else {
//
//        }
//        dataIndex = (dataIndex + 1) % outputLen
//      } else {
//        featureData.putRow(i, dataMap(candidates(dataIndex)).head)
//        dataMap(candidates(dataIndex)) = dataMap(candidates(dataIndex)).tail
//        val actionData = Nd4j.zeros(outputLen)
//        actionData.putScalar(dataIndex, 1.0)
//        labelData.putRow(i, actionData)
//        dataIndex = (dataIndex + 1) % outputLen
//      }


      //      var pair: (INDArray, Int) = null
      //      if (dataMap(candidates(dataIndex)).isEmpty) {
//        toLoad = true
//        pair = (Nd4j.zeros(PeerStateLen), outputLen)
//        caseNum(outputLen) += 1
//      }else {
//        pair = (dataMap(candidates(dataIndex)).head, dataIndex)
//        dataMap(candidates(dataIndex)) = dataMap(candidates(dataIndex)).tail
//        caseNum(dataIndex) += 1
//      }
//      dataIndex = (dataIndex + 1) % outputLen
//
//      val feature = pair._1
//      val action = pair._2
//
//      featureData.putRow(i, feature)
//
//      val actionData = Nd4j.zeros(outputLen)
//      if (action < outputLen) {
//        actionData.putScalar(action, 1.0)
//      }
//      labelData.putRow(i, actionData)
    }

//    logger.debug(caseNum.mkString(", "))
    curr = new DataSet(featureData, labelData)
  }
}
