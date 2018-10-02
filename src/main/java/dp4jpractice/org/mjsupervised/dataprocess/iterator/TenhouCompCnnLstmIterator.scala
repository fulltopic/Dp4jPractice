package dp4jpractice.org.mjsupervised.dataprocess.iterator

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import dp4jpractice.org.mjsupervised.utils.ImplConsts._

class TenhouCompCnnLstmIterator(xmlParentDir: String, batchSize: Int, winner: Boolean = false, model: MultiLayerNetwork = null) extends  TenhouLstmIterator(xmlParentDir, batchSize, winner, model){
  override val seqLen: Int = 20

  override def next(num: Int): DataSet = {
    currentData = currentData.filter(data => data.nonEmpty)

    if (currentData.length < num && hasNext) {
      loadNewFile()
      currentData = currentData.filter(data => data.nonEmpty)
    }


//    val actionLen = ActionLenWoAccept
//    val input = Nd4j.create(Array[Int](batchSize, inputColumns(), seqLen), 'f')
//    val labels = Nd4j.create(Array[Int](batchSize, actionLen, seqLen), 'f')
//
//
//    for (i <- 0 until batchSize) {
//      if (!currentData.isEmpty) {
//        val data = currentData.head
//        currentData = currentData.tail
//
//        val actualLen = math.min(data.length, seqLen)
//
//        for (j <- 0 until actualLen) {
//          var delta: INDArray = null
//          //          delta = data(j)._1
//
//          if (j == 0) {
//            delta = data(j)._1
//          }else {
//            delta = data(j)._1.sub(data(j - 1)._1)
//          }
//          //          logger.debug(delta)
//
//          for (k <- 0 until inputColumns()) {
//            //            input.putScalar(Array[Int](i, k, j), data(j)._1.getDouble(k))
//            input.putScalar(Array[Int](i, k, j), delta.getDouble(k))
//          }
//          labels.putScalar(Array[Int](i, data(j)._2, j), 1.0)
//        }
//      }
//    }
//
//    new DataSet(input, labels)

    val actionLen = ActionLenWoAccept
//    val input = Nd4j.create(batchSize, 1, seqLen, inputColumns())
//    val labels = Nd4j.create(batchSize, 1, seqLen, actionLen)
//
//
//    for (i <- 0 until batchSize) {
//      if (!currentData.isEmpty) {
//        val data = currentData.head
//        currentData = currentData.tail
//
//        val actualLen = math.min(data.length, seqLen)
//
//        for (j <- 0 until actualLen) {
//          for (k <- 0 until inputColumns()) {
//            input.putScalar(i, 0, j, k, data(j)._1.getDouble(k))
//          }
//          val action = data(j)._2
//          labels.putScalar(i, 0, j, action, 1.0)
//        }
//      }
//    }

    val seqLen = 20
//    val input = Nd4j.create(seqLen, PeerStateLen)
//    val labels = Nd4j.create(seqLen, ActionLenWoAccept)
    var input: INDArray = null
    var labels: INDArray = null

    if (currentData.nonEmpty) {
      val data = currentData.head
      currentData = currentData.tail

      val actualLen = math.min(data.length, seqLen)
//      val actualLen = data.length
//      println(actualLen)

      input = Nd4j.zeros(seqLen, PeerStateLen)
      labels = Nd4j.zeros(seqLen, ActionLenWoAccept)

      for (i <- 0 until actualLen) {
        for (j <- 0 until PeerStateLen) {
          input.putScalar(i, j, data(i)._1.getDouble(j))
        }
//        if (i == 0) {
//          for (j <- 0 until PeerStateLen) {
//            input.putScalar(i, j, data(i)._1.getDouble(j))
//          }
//        }else {
//          for (j <- 0 until PeerStateLen) {
//            input.putScalar(i, j, data(i)._1.getDouble(j) - data(i - 1)._1.getDouble(j))
//          }
//        }
        labels.putScalar(i, data(i)._2,1.0)
      }
    }

//    println(input)

    new DataSet(input, labels)
  }
}
