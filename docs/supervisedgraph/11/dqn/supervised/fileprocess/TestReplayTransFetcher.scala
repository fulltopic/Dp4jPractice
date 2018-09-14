package rl.dqn.supervised.fileprocess

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator

object TestReplayTransFetcher {
  val dirPath = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db1/objs/testfetcher"
  val batchSize: Int = 128
  val exampleNum: Int = 1

  def testFetcher(): Unit = {
    val fetcher = new ReplayTransFetcher(dirPath)
    val datasetIterator = new BaseDatasetIterator(batchSize, exampleNum, fetcher)
    var exampleCount = 0
    while (datasetIterator.hasNext) {
      val data = datasetIterator.next()
      val features = data.getFeatureMatrix
      val labels = data.getLabels

      val num = features.shape()(0)

      if(num != batchSize) println("Not match " + num + " != " + batchSize)

      exampleCount += batchSize
    }

    println("---------------------------------> total " + exampleCount)
  }

  def testLabels(): Unit = {
    val fetcher = new ReplayTransFetcher(dirPath)
    val datasetIterator = new BaseDatasetIterator(batchSize, exampleNum, fetcher)
    var exampleCount = 0
    while (datasetIterator.hasNext && exampleCount < 1024) {
      val data = datasetIterator.next()
      val features = data.getFeatureMatrix
      val labels = data.getLabels

      val num = features.shape()(0)
      if(num != batchSize) println("Not match " + num + " != " + batchSize)

      println(labels)
      for (i <- 0 until batchSize) {
        println(labels.getRow(i).sumNumber() == 1)
      }

      exampleCount += num
    }
  }

  def main(arg: Array[String]): Unit = {
    testFetcher()
  }
}
