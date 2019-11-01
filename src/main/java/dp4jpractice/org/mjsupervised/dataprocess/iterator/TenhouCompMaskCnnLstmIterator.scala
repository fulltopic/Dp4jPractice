package dp4jpractice.org.mjsupervised.dataprocess.iterator

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import tenhouclient.impl.ImplConsts._

class TenhouCompMaskCnnLstmIterator(xmlParentDir: String, batchSize: Int, winner: Boolean = false, model: MultiLayerNetwork = null)
  extends TenhouCompCnnLstmIterator(xmlParentDir, batchSize, winner, model)
{
  override val seqLen: Int = 20

  override def next(num: Int): DataSet = {
    currentData = currentData.filter(data => data.nonEmpty)

    if (currentData.length < num && hasNext) {
      loadNewFile()
      currentData = currentData.filter(data => data.nonEmpty)
    }

    val actionLen = ActionLenWoAccept

    val seqLen = 20

    val batchSize = math.min(num, currentData.length)
    logger.debug("Get batch size from iterator " + num)
    val input: INDArray = Nd4j.zeros(Array[Int](num, PeerStateLen, seqLen), 'f')
    val labels: INDArray = Nd4j.zeros(Array[Int](num, ActionLenWoAccept, seqLen), 'f')
    val inputMask: INDArray = Nd4j.zeros(num.toLong, seqLen)
    val labelMask: INDArray = Nd4j.zeros(num.toLong, seqLen)

    var index: Int = 0

    while (currentData.nonEmpty && index < batchSize) {
      val data = currentData.head
      currentData = currentData.tail

      val actualLen = math.min(data.length, seqLen)

      for (j <- 0 until actualLen) {
                  inputMask.putScalar(index, j, 1.0)
                  labelMask.putScalar(index, j, 1.0)
        for (i <- 0 until PeerStateLen) {
          input.putScalar(index, i, j, data(j)._1.getDouble(i.toLong))
        }

        labels.putScalar(index, data(j)._2, j,1.0)
      }


      index += 1
    }

    new DataSet(input, labels, inputMask, labelMask)
  }

}
