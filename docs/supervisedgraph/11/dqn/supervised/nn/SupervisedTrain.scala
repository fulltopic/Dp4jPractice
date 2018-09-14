package rl.dqn.supervised.nn

// TODO: 1. Make number of examples % 128 == 0
// 2. Create example file with number ~ 10,000
// 3. Random
// 4. Save
// 5. Create train/validate/test set
// 6. Decide number of hyper-parameters
// 7. Print out statistics
// 8. Set up cloud environment
// 9. Make small files to accelerate access efficiency
// 10. Spark framework

import akka.event.slf4j.Logger
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{LearningRatePolicy, NeuralNetConfiguration, Updater, WorkspaceMode}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.util.Constants
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import rl.dqn.reinforcement.dqn.config.Supervised._
import rl.dqn.supervised.fileprocess.ReplayTransFetcher

//TODO: Normalization
object SupervisedTrain {
  private val logger = Logger("SupervisedTrain")

  def createConf(numLayer: Int, numHiddenNodes: Int, l2: Double, learningRate: Double): DQNFactoryStdDense.Configuration = {
    DQNFactoryStdDense.Configuration.builder()
      .learningRate(learningRate)
      .numLayer(numLayer)
      .numHiddenNodes(numHiddenNodes)
      .l2(l2)
      .updater(new Adam())
      .build()
  }

  def createNN(conf: DQNFactoryStdDense.Configuration, supervised: Boolean): MultiLayerNetwork = {
    val inputNums = Array[Int](1, PeerStateLen)
    val outputNums = ActionLen

    if (supervised) {
      DenseNN.createSupervisedNN(conf, inputNums, outputNums)
    } else {
      DenseNN.createDqnNN(conf, inputNums, outputNums)
    }
  }


  def simpleTrain(): MultiLayerNetwork = {
     val conf = TestNN.createConf(2, 128, 0.01, 0.0005)
    TestNN.createNN(conf, true, PeerStateLen, ActionLen)
  }


  def singleLayerTrain(): MultiLayerNetwork = {
    val conf = TestNN.createConf(1, 64, 0.01, 0.0005)
    TestNN.createNN(conf, true, PeerStateLen, ActionLen)
  }

  def cnn_1Train(): MultiLayerNetwork = {
    val hiddenNode = 128
    val denseLayerNum = 1
    val iterationNum = 2
    val cnnLayerNum = 1
    val cnnOut = 32
    val channelNum = 1

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, 0.1)
//    lrShedule.put(10000, 0.1)
        lrShedule.put(10000, 0.05)
        lrShedule.put(20000, 0.01)
    val listBuilder = new NeuralNetConfiguration.Builder()
              .learningRateDecayPolicy(LearningRatePolicy.Schedule)
              .learningRateSchedule(lrShedule)
//      .learningRate(0.01)
      .iterations(iterationNum)
      .seed(47)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAGRAD)
      .weightInit(WeightInit.XAVIER)
      .regularization(true)
      .l2(0.0005)
      .list()

    listBuilder.layer(0, new ConvolutionLayer.Builder(1, 3)
//                            .padding(0, 1)
                            .nIn(channelNum)
                            .nOut(cnnOut)
                            .activation(Activation.RELU)
                            .build())
//    listBuilder.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
//                            .kernelSize(1, 3)
//                            .build())
//    listBuilder.layer(2, new ConvolutionLayer.Builder(1, 3)
//      .nOut(cnnOut)
//      .activation(Activation.RELU)
//      .build())
//    listBuilder.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
//      .kernelSize(1, 3)
//      .build())

    listBuilder.layer(cnnLayerNum, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.RELU).build())

    for (i <- 1 until denseLayerNum) {
      listBuilder.layer(i + cnnLayerNum, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.RELU).build())
    }

    listBuilder.layer(denseLayerNum + cnnLayerNum,
      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(hiddenNode).nOut(ActionLen).activation(Activation.SOFTMAX).build())


    val mlnConf = listBuilder.setInputType(InputType.convolutionalFlat(1, PeerStateLen, channelNum))
        .pretrain(false).backprop(true).build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    model
  }

  def trainModel(model: MultiLayerNetwork): Unit = {
    val trainDir = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db1/objs/train/"
    val validDir = "/home/zf/workspaces/workspace_java/mjconv2/datasets/tenhoulogs/logs/db1/objs/validation/"

    val epochNum = 16
    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))

    //    logger.info("Train model....");
    //    model.setListeners(new ScoreIterationListener(1));


    val trainFetcher = new ReplayTransFetcher(trainDir)
    val trainIte = new BaseDatasetIterator(64, 1, trainFetcher)

    val validFetcher = new ReplayTransFetcher(validDir)
    val validIte = new BaseDatasetIterator(64, 1, validFetcher)

    for (i <- 0 until epochNum) {
      trainIte.reset()
      model.fit(trainIte)

      validIte.reset()
      val eval = model.evaluate(validIte)
      //      val eval = new Evaluation()
      //      model.doEvaluation(validIte, eval)

      logger.info("=========================================> Evaluation")
      logger.info(eval.stats())
    }

    logger.info("::::::::::::::::::::::::::::::::::::::::: End of training")
  }


  def main(args: Array[String]): Unit = {
      trainModel(singleLayerTrain())
  }
}
