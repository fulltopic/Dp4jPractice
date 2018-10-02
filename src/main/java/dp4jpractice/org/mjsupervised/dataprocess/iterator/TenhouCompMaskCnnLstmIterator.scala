package dp4jpractice.org.mjsupervised.dataprocess.iterator

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import dp4jpractice.org.mjsupervised.utils.ImplConsts._

class TenhouCompMaskCnnLstmIterator(xmlParentDir: String, batchSize: Int, winner: Boolean = false, model: MultiLayerNetwork = null) extends TenhouCompCnnLstmIterator(xmlParentDir, batchSize, winner, model) {
  override val seqLen: Int = 20

//  override def next(num: Int): DataSet = {
//    currentData = currentData.filter(data => data.nonEmpty)
//
//    if (currentData.length < num && hasNext) {
//      loadNewFile()
//      currentData = currentData.filter(data => data.nonEmpty)
//    }
//
//    val actionLen = ActionLenWoAccept
//
//    val seqLen = 20
//
//    val batchSize = math.min(num, currentData.length)
//    val input: INDArray = Nd4j.zeros(batchSize, seqLen, PeerStateLen)
//    val labels: INDArray = Nd4j.zeros(batchSize, seqLen, ActionLenWoAccept)
//    val inputMask: INDArray = Nd4j.zeros(batchSize, seqLen)
//    val labelMask: INDArray = Nd4j.zeros(batchSize, seqLen)
//    var index: Int = 0
//
//    while (currentData.nonEmpty && index < batchSize) {
//      val data = currentData.head
//      currentData = currentData.tail
//
//      val actualLen = math.min(data.length, seqLen)
//
//      for (i <- 0 until actualLen) {
//        inputMask.putScalar(index, i, 1.0)
//        labelMask.putScalar(index, i, 1.0)
//        for (j <- 0 until PeerStateLen) {
//          input.putScalar(index, i, j, data(i)._1.getDouble(j))
//        }
//
//        labels.putScalar(index, i, data(i)._2,1.0)
//      }
//
//
//      index += 1
//    }
//
//    new DataSet(input, labels, inputMask, labelMask)
//  }

  override def next(num: Int): DataSet = {
    currentData = currentData.filter(data => data.nonEmpty)

    if (currentData.length < num && hasNext) {
      loadNewFile()
      currentData = currentData.filter(data => data.nonEmpty)
    }

    val actionLen = ActionLenWoAccept

    val seqLen = 20

    val batchSize = math.min(num, currentData.length)
    val input: INDArray = Nd4j.zeros(Array[Int](batchSize, PeerStateLen, seqLen), 'f')
    val labels: INDArray = Nd4j.zeros(Array[Int](batchSize, ActionLenWoAccept, seqLen), 'f')
    val inputMask: INDArray = Nd4j.zeros(batchSize, seqLen)
    val labelMask: INDArray = Nd4j.zeros(batchSize, seqLen)
    var index: Int = 0

    while (currentData.nonEmpty && index < batchSize) {
      val data = currentData.head
      currentData = currentData.tail

      val actualLen = math.min(data.length, seqLen)

      for (j <- 0 until actualLen) {
                  inputMask.putScalar(index, j, 1.0)
                  labelMask.putScalar(index, j, 1.0)
        for (i <- 0 until PeerStateLen) {
          input.putScalar(index, i, j, data(j)._1.getDouble(i))
        }

        labels.putScalar(index, data(j)._2, j,1.0)
      }


      index += 1
    }

    new DataSet(input, labels, inputMask, labelMask)
  }

//    override def next(num: Int): DataSet = {
//      currentData = currentData.filter(data => data.nonEmpty)
//
//      if (currentData.length < num && hasNext) {
//        loadNewFile()
//        currentData = currentData.filter(data => data.nonEmpty)
//      }
//
//      val actionLen = ActionLenWoAccept
//
//      val seqLen = 20
//
//      val batchSize = math.min(num, currentData.length)
//      val input: INDArray = Nd4j.zeros(batchSize, PeerStateLen, seqLen)
//      val labels: INDArray = Nd4j.zeros(batchSize, ActionLenWoAccept, seqLen)
////      val inputMask: INDArray = Nd4j.zeros(batchSize, seqLen)
////      val labelMask: INDArray = Nd4j.zeros(batchSize, seqLen)
//      var index: Int = 0
//
//      while (currentData.nonEmpty && index < batchSize) {
//        val data = currentData.head
//        currentData = currentData.tail
//
//        val actualLen = math.min(data.length, seqLen)
//
//        for (j <- 0 until actualLen) {
////          inputMask.putScalar(index, i, 1.0)
////          labelMask.putScalar(index, i, 1.0)
//          for (i <- 0 until PeerStateLen) {
//            input.putScalar(index, i, j, data(j)._1.getDouble(i))
//          }
//
//          labels.putScalar(index, data(j)._2, j,1.0)
//        }
//
//
//        index += 1
//      }
//
//      new DataSet(input, labels)
//    }
}
