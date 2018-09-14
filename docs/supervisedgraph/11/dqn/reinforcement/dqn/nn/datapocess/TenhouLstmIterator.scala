package rl.dqn.reinforcement.dqn.nn.datapocess

import java.io.File
import java.util

import akka.event.slf4j.Logger
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

import rl.dqn.reinforcement.dqn.config.Supervised._

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.util.{Failure, Random, Success}

class TenhouLstmIterator(xmlParentDir: String, batchSize: Int, winner: Boolean = false, val model: MultiLayerNetwork = null) extends  DataSetIterator{
  val logger = Logger("TenhouLstmIterator")
  
  val itemRandom: Random = new Random(System.currentTimeMillis())
  private val dirs = new File(xmlParentDir).listFiles().filter(_.isDirectory).toList
  var files = List.empty[String]
  dirs.foreach(dir => {
    val dirFiles = dir.listFiles().filter(_.isFile).map(_.getAbsolutePath)
    files = files ++ dirFiles
  })
  logger.debug("Get files " + files.length)

  val seqLen: Int = 16
  private val fileOrder = Random.shuffle(files.indices.toList).toArray
  var fileIndex: Int = 0
  var doneFileNum: Int = 0
  var currentReaders = Array.empty[TenhouLstmReader]
  var toLoad: Boolean = true

  var currentData = List.empty[List[(INDArray, Int)]]

  val fileNumPerLoad: Int = 8


  override def next(): DataSet = {
    this.next(batchSize)
  }

  override def totalExamples(): Int = {
    logger.debug("----------------------------> Called totalExamples: ")
    0
  }

  override def inputColumns(): Int = {
    PeerStateLen
  }

  override def totalOutcomes(): Int = ActionLenWoAccept

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def reset(): Unit = {
    fileIndex = 0
    currentData = List.empty[List[(INDArray, Int)]]
    //TODO: Reset currentData
  }

  override def batch(): Int = batchSize

  override def numExamples(): Int = {
    logger.debug("-------------------------> Total num examples ")
    0
  }

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = {
    //do nothing
  }

  override def getPreProcessor: DataSetPreProcessor = null

  override def getLabels: util.List[String] = {
    //do nothing
    null
  }

  override def cursor(): Int = {
    logger.debug("-------------------------> Cursor ")
    0
  }

  override def hasNext: Boolean = {
    fileIndex < files.length
  }


  import scala.concurrent.ExecutionContext.Implicits.global
  def getLoadFuture(reader: TenhouLstmReader): Future[List[List[(INDArray, Int)]]] = Future {
    reader.getData()
  }


  protected def loadNewFile(): Unit = {
    logger.debug("-----------------------------> Load " + files.length + ", " + fileIndex)
    val fileNum = math.min(fileNumPerLoad, files.length - fileIndex)

    currentReaders = new Array[TenhouLstmReader](fileNum)

    if (!winner) {
      for (i <- 0 until fileNum) {
        currentReaders(i) = new TenhouLstmReader(files(fileOrder(fileIndex)))
        fileIndex += 1
      }
    }else {
      for (i <- 0 until fileNum) {
        currentReaders(i) = new TenhouLstmReaderWinner(files(fileOrder(fileIndex)))
        fileIndex += 1
      }
    }


    val futures = for (reader <- currentReaders) yield getLoadFuture(reader)
    var count: Int = 0

    futures.foreach(future => {
      future onComplete {
        case Success(playerData) =>
          currentData = currentData ++ playerData
          count += 1
        case Failure(_) =>
          count += 1
      }
    })

    futures.foreach(future => Await.ready(future, Duration.Inf))

    currentData = currentData.filter(data => data.nonEmpty)

    currentData = Random.shuffle(currentData)

    toLoad = false
    logger.debug("Get all readers loaded "  + count)
  }


  override def next(num: Int): DataSet = {
    if (currentData.length < num && hasNext) {
      loadNewFile()
    }


    val actionLen = ActionLenWoAccept
    val input = Nd4j.create(Array[Int](batchSize, inputColumns(), seqLen), 'f')
    val labels = Nd4j.create(Array[Int](batchSize, actionLen, seqLen), 'f')


    for (i <- 0 until batchSize) {
      if (!currentData.isEmpty) {
        val data = currentData.head
        currentData = currentData.tail

        val actualLen = math.min(data.length, seqLen)

        for (j <- 0 until actualLen) {
          var delta: INDArray = null
//          delta = data(j)._1

          if (j == 0) {
            delta = data(j)._1
          }else {
            delta = data(j)._1.sub(data(j - 1)._1)
          }
//          logger.debug(delta)

          for (k <- 0 until inputColumns()) {
//            input.putScalar(Array[Int](i, k, j), data(j)._1.getDouble(k))
            input.putScalar(Array[Int](i, k, j), delta.getDouble(k))
          }
          labels.putScalar(Array[Int](i, data(j)._2, j), 1.0)
        }
      }
    }

    new DataSet(input, labels)
  }


//  var myData = List.empty[(INDArray, Int)]
//  var lastData: INDArray = null
//  var lastState0: util.Map[String, INDArray] = null
//  var lastState1: util.Map[String, INDArray] = null
//  override def next(num: Int): DataSet = {
//    if (myData.isEmpty && hasNext) {
//      loadNewFile()
//      while (currentData.nonEmpty) {
//        myData = myData ++ currentData.head
//        myData = myData :+ (null, -1)
//        currentData = currentData.tail
//      }
//    }
//
//    val myBatchSize: Int = 1
//    val mySeqLen: Int = 1
//
//    val actionLen = ActionLenWoAccept
//    val input = Nd4j.create(Array[Int](myBatchSize, inputColumns(), mySeqLen), 'f')
//    val labels = Nd4j.create(Array[Int](myBatchSize, actionLen, mySeqLen), 'f')
//
//
//    if (myData.nonEmpty) {
//      val data = myData.head
//      myData = myData.tail
//
//      if (data._2 >= 0) {
//        val inputArray = data._1
//        val action = data._2
//        var delta = inputArray
//        if (lastData != null) {
////          logger.debug("lastData " + lastData)
//
//          delta = inputArray.sub(lastData)
//          if (model != null) {
//          }
//        }
//        lastData = data._1
//
//        for (k <- 0 until inputColumns()) {
//          input.putScalar(Array[Int](0, k, 0), delta.getDouble(k))
//        }
//        labels.putScalar(Array[Int](0, action, 0), 1.0)
////        logger.debug("data_.1 " + data._1)
////        logger.debug("inputArray " + inputArray)
////        logger.debug("delta " + delta)
////        logger.debug("action " + action)
//      } else {
//        if (model != null)
//        {
////          logger.debug("Reset previous state")
////          model.rnnClearPreviousState()
//        }
//        lastData = null
//      }
//    }
//    new DataSet(input, labels)
//  }

//  var currentScene = Array.empty[(INDArray, Int)]
//  var stepIndex: Int = 0
//  val myBatchSize: Int = 1

//  override def next(num: Int): DataSet = {
//    if (currentData.length < num && hasNext) {
//      loadNewFile()
//    }
//
//    if (stepIndex >= currentScene.length) {
//      currentScene = currentData.head.toArray
//      currentData = currentData.tail
//      stepIndex = 0
////      logger.debug("reset " + stepIndex + ", " + currentScene.length)
//    }
////    logger.debug(stepIndex)
//
//    val mySeqLen = stepIndex + 1
//    val actionLen = ActionLenWoAccept
//    val input = Nd4j.create(Array[Int](myBatchSize, inputColumns(), mySeqLen), 'f')
//    val labels = Nd4j.create(Array[Int](myBatchSize, actionLen, mySeqLen), 'f')
//
//    for (i <- 0 until mySeqLen) {
////      if (i >= currentScene.length) {
////        logger.debug("---------------------------------------------->")
////        logger.debug(i)
////        logger.debug(stepIndex)
////        logger.debug(currentScene.length)
////        logger.debug(currentData.length)
////        logger.debug(fileIndex)
////        logger.debug(files.length)
////      }
//      var delta = currentScene(i)._1
//      if (i > 0) {
//        delta = currentScene(i)._1.sub(currentScene(i - 1)._1)
//      }
//      val action = currentScene(i)._2
//      for (k <- 0 until inputColumns()) {
//        input.putScalar(Array[Int](0, k, i), delta.getDouble(k))
//      }
//      labels.putScalar(Array[Int](0, action, i), 1.0)
//    }
//
//    stepIndex += 1
////    logger.debug("update " + stepIndex)
//
//    new DataSet(input, labels)
//  }

}
