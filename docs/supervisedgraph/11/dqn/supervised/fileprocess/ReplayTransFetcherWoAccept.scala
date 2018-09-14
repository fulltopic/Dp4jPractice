package rl.dqn.supervised.fileprocess

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import rl.dqn.reinforcement.dqn.config.Supervised._

class ReplayTransFetcherWoAccept(dirPath: String) extends ReplayTransFetcher(dirPath) {
  override def fetch(numExamples: Int): Unit = {
    if (numExamples > BatchSizeLimit) throw new IllegalArgumentException("numExample no more than " + BatchSizeLimit)
    if (!hasMore) throw new IllegalStateException("No more examples")

    if (dataset.length - itemIndex < numExamples) nextFiles() //give up remains. Should not happen as always %16 == 0

    val featureData = Nd4j.create(numExamples, PeerStateLen)
    val labelData = Nd4j.create(numExamples, ActionLenWoAccept)

    for (i <- 0 until numExamples)  {
      val item = dataset(itemOrder(itemIndex))
      itemIndex += 1

      featureData.putRow(i, item.getObservation()(0))
      val labels = Nd4j.create(ActionLenWoAccept)
      val action = item.getAction
      labels.putScalar(action, 1.0)
      labelData.putRow(i, labels)
    }

    curr = new DataSet(featureData, labelData)
  }
}
