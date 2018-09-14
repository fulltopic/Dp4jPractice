package rl.dqn.supervised.fileprocess

import java.io.{File, FileInputStream, ObjectInputStream}

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher
import org.deeplearning4j.util.MathUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import rl.dqn.supervised.utils.{DbTransition, ReplayDB}
import rl.dqn.reinforcement.dqn.config.Supervised._

import scala.util.Random

//TODO: Make use of Nd4j.shuffle
class ReplayTransFetcher(dirPath: String)  extends BaseDataFetcher{
//  val fileRandom: Random = new Random(fileSeed)
  val itemRandom: Random = new Random(System.currentTimeMillis())
  val files: Array[File] = new File(dirPath).listFiles().filter(_.isFile)
  val fileOrderList: List[Int] = (0 until ((files.length / 2) * 2)).toList //Make sure even file numbers
  var itemOrder: Array[Int] = Array.emptyIntArray
  var dataset: Vector[DbTransition] = scala.collection.immutable.Vector.empty[DbTransition]
  var fileIndex: Int = 0
  var itemIndex: Int = 0
  val BatchSizeLimit: Int = 1024


  //TODO: Check effect of shuffle
  var fileOrder = Random.shuffle(fileOrderList).toArray

  def loadDbObj(): ReplayDB = {
//    println("Load file " + fileOrder(fileIndex))
    val file = files(fileOrder(fileIndex))
    fileIndex += 1
    new ObjectInputStream(new FileInputStream(file)).readObject().asInstanceOf[ReplayDB]
  }


  def nextFiles(): Unit = {
//    fileOrderList.foreach(println)

    val trans1 = loadDbObj().datas
    val trans2 = loadDbObj().datas

    dataset = scala.collection.immutable.Vector.empty[DbTransition]

    for (i <- 0 until trans1.size()) {
      dataset = dataset :+ trans1.get(i)
    }
    for(i <- 0 until trans2.size()) {
      dataset = dataset :+ trans2.get(i)
    }

    itemIndex = 0
    val itemOrderList = (0 until dataset.length).toList
    itemOrder = Random.shuffle(itemOrderList).toArray
  }

  override def hasMore(): Boolean = {
    if (itemIndex < dataset.length) true

    fileIndex match {
      case index if index < files.length - 1 => true
      case index if index == files.length - 1 => (dataset.length - itemIndex) > BatchSizeLimit
      case _ => false
    }
  }

  //TODO: datasetiterator.next. make cursor = 0 and never updated. So depends on hasMore only
  override def fetch(numExamples: Int): Unit = {
    if (numExamples > BatchSizeLimit) throw new IllegalArgumentException("numExample no more than " + BatchSizeLimit)
    if (!hasMore) throw new IllegalStateException("No more examples")

    if (dataset.length - itemIndex < numExamples) nextFiles() //give up remains. Should not happen as always %16 == 0

    val featureData = Nd4j.create(numExamples, PeerStateLen)
    val labelData = Nd4j.create(numExamples, ActionLen)

    for (i <- 0 until numExamples)  {
      val item = dataset(itemOrder(itemIndex))
      itemIndex += 1

      featureData.putRow(i, item.getObservation()(0))
      val labels = Nd4j.create(ActionLen)
      val action = item.getAction
      labels.putScalar(action, 1.0)
      labelData.putRow(i, labels)
    }

    curr = new DataSet(featureData, labelData)
  }

  override def reset(): Unit = {
    fileIndex = 0
    itemIndex = 0
    curr = null

    fileOrder = Random.shuffle(fileOrderList).toArray
  }
}
