package dp4jpractice.org.mjsupervised.dataprocess.iterator

import java.io.File
import java.util

import akka.event.slf4j.Logger
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import tenhouclient.impl.ImplConsts._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.util.{Failure, Random, Success}

class TenhouLstmIterator(xmlParentDir: String, batchSize: Int, winner: Boolean = false, val model: MultiLayerNetwork = null) extends  DataSetIterator{
  val logger = Logger("TenhouLstmIterator")
  println("xmlParentDir " + xmlParentDir)
  
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


  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = {
    //do nothing
  }

  override def getPreProcessor: DataSetPreProcessor = null

  override def getLabels: util.List[String] = {
    //do nothing
    null
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
      if (currentData.nonEmpty) {
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
          logger.debug("" + delta)

          for (k <- 0 until inputColumns()) {
            input.putScalar(Array[Int](i, k, j), delta.getDouble(k.toLong))
          }
          labels.putScalar(Array[Int](i, data(j)._2, j), 1.0)
        }
      }
    }

    new DataSet(input, labels)
  }

}
