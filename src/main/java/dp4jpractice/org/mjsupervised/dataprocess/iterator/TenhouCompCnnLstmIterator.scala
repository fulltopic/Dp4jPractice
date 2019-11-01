package dp4jpractice.org.mjsupervised.dataprocess.iterator

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import tenhouclient.impl.ImplConsts._
//import dp4jpractice.org.mjsupervised.utils.ImplConsts._

class TenhouCompCnnLstmIterator(xmlParentDir: String, batchSize: Int, winner: Boolean = false, model: MultiLayerNetwork = null)
  extends  TenhouLstmIterator(xmlParentDir, batchSize, winner, model){
  override val seqLen: Int = 20

  override def next(num: Int): DataSet = {
    currentData = currentData.filter(data => data.nonEmpty)

    if (currentData.length < num && hasNext) {
      loadNewFile()
      currentData = currentData.filter(data => data.nonEmpty)
    }


//    val actionLen = ActionLenWoAccept

    val seqLen = 20
    var input: INDArray = null
    var labels: INDArray = null

    if (currentData.nonEmpty) {
      val data = currentData.head
      currentData = currentData.tail

      val actualLen = math.min(data.length, seqLen)

      input = Nd4j.zeros(seqLen.toLong, PeerStateLen)
      labels = Nd4j.zeros(seqLen.toLong, ActionLenWoAccept)

      for (i <- 0 until actualLen) {
        for (j <- 0 until PeerStateLen) {
          input.putScalar(i, j, data(i)._1.getDouble(j.toLong))
        }
        labels.putScalar(i, data(i)._2,1.0)
      }
    }

    new DataSet(input, labels)
  }
}
