package rl.dqn.supervised.nn

import java.io.{File, FileOutputStream, PrintWriter}

import akka.event.slf4j.Logger
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{LearningRatePolicy, NeuralNetConfiguration, Updater, WorkspaceMode}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.LossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions
import rl.dqn.reinforcement.dqn.config.Supervised._
import rl.dqn.supervised.fileprocess.{ReplayTransFetcher, ReplayTransFetcherWoAccept}

import scala.io.Source
import scala.util.Random

object SupervisedTrainNoAccept {
  private val logger = Logger("SupervisedTrainNoAccept")
  
  def singleLayerTrain(): MultiLayerNetwork = {
    val conf = TestNN.createConf(3, 128, 0.5, 0.0005)
    TestNN.createNN(conf, true, PeerStateLen, ActionLenWoAccept)
  }

  def getHiddenNodeNum(): Int = {
    val nums = Array[Int](32, 64, 128, 256, 512)
    nums(Random.nextInt(nums.length))
  }

  def nnLayer(): Int = {
    val nums = Array[Int](1, 2, 3)
    nums(Random.nextInt(nums.length))
  }

  def learningRate(): Double = {
    val rates = Array[Double](0.1, 0.05, 0.01, 0.005)
    rates(Random.nextInt(rates.length))
  }

  def l2Reg(): Double = {
    val regs = Array[Double](0.0005, 0.0001, 0.001, 0.005)
    regs(Random.nextInt(regs.length))
  }

  def randomDenseTrain(): Unit = {
    for (i <- 0 until 7) {
      val layer = nnLayer()
      val nodes = getHiddenNodeNum()
      val lr = learningRate()
      val reg = l2Reg()
      logger.info("=======================================================================================================")
      logger.info("=======================================================================================================")
      logger.info("Test " + layer + ", " + nodes + ", " + lr + ", " + reg)
      val conf = TestNN.createConf(layer, nodes, lr, reg)
      val model = TestNN.createNN(conf, true, PeerStateLen, ActionLenWoAccept)
      trainModel(model, 2)
    }
  }

  def cnn_1Train(): MultiLayerNetwork = {
    val hiddenNode = 128
    val denseLayerNum = 2
    val iterationNum = 2
    val cnnLayerNum = 2
    val cnnOut = 32
    val channelNum = 1

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, 0.1)
    //    lrShedule.put(10000, 0.1)
//    lrShedule.put(10000, 0.05)
//    lrShedule.put(20000, 0.01)
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
      .nOut(hiddenNode)
      .activation(Activation.RELU)
      .build())
        listBuilder.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                                .kernelSize(1, 3)
                                .build())
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
      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(hiddenNode).nOut(ActionLenWoAccept).activation(Activation.SOFTMAX).build())


    val mlnConf = listBuilder.setInputType(InputType.convolutionalFlat(1, PeerStateLen, channelNum))
      .pretrain(false).backprop(true).build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    model
  }

  def createdqrn(iterationNum: Int): MultiLayerNetwork = {
    val hiddenNode = 128
    val denseLayerNum = 2
//    val iterationNum = 2
    val cnnLayerNum = 0
    val cnnOut = 32
    val channelNum = 1
    val lstmSize = 128
    val inputNums = Array[Int](1, PeerStateLen)

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, 10.0)
        lrShedule.put(100000, 5.0)
        lrShedule.put(200000, 1.0)

    val listBuilder = new NeuralNetConfiguration.Builder()
//      .learningRateDecayPolicy(LearningRatePolicy.)
//        .lrPolicyPower(0.99)
//        .lrPolicyDecayRate(0.99)
//        .lrPolicySteps(5)
//      .learningRateSchedule(lrShedule)
            .learningRate(10)
      .iterations(iterationNum)
      .seed(47)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAGRAD)
      .weightInit(WeightInit.XAVIER)
      .regularization(true)
      .l2(0.0005)
      .list()

    listBuilder.layer(0, new DenseLayer.Builder().nIn(PeerStateLen).nOut(hiddenNode).activation(Activation.RELU).build())


    for (i <- 1 until cnnLayerNum + denseLayerNum - 1) {
      listBuilder.layer(i + cnnLayerNum, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.RELU).build())
    }
    listBuilder.layer(cnnLayerNum + denseLayerNum - 1, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.SOFTSIGN).build())



    listBuilder.layer(denseLayerNum + cnnLayerNum, new GravesLSTM.Builder().nOut(lstmSize).activation(Activation.SOFTSIGN).build())
    listBuilder.layer(denseLayerNum + cnnLayerNum + 1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT). //MCXENT, NEGATIVELOGLIKELIHOOD
      activation(Activation.SOFTMAX).nOut(ActionLenWoAccept).build())


    val mlnConf = listBuilder.pretrain(false).backprop(true).build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    model
  }

  def getRandomParam[T](params: Array[T]): T = {
    params(Random.nextInt(params.length))
  }

  def generateParams(): (Int, Int, Int, Double, Activation, LossFunctions.LossFunction) = {
    val hiddenNodes = Array[Int](64, 128, 256, 512)
    val denseLayerNums = Array[Int](1, 2, 3)
    val lstmSizes = Array[Int](64, 128, 256, 512)
    val startLr = Array[Double](10.0, 8.0, 5.0, 1.0)
    val lastDnnActivations = Array[Activation](Activation.RELU, Activation.TANH, Activation.SOFTSIGN)
    val outputLossFunc = Array[LossFunctions.LossFunction](LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)

    (getRandomParam(hiddenNodes), getRandomParam(denseLayerNums), getRandomParam(lstmSizes), getRandomParam(startLr), getRandomParam(lastDnnActivations), getRandomParam(outputLossFunc))
  }



  val logFile = new PrintWriter(new File("/home/ec2-user/tenhoulogs/logs/db4/dqrn/player/output.txt"))
  def writeLog(content: String): Unit = {
    logFile.write(content + "\n")
    logFile.flush()
  }

  def createClouddqrn(): MultiLayerNetwork = {
//    val (hiddenNode, denseLayerNum, lstmSize, startLr, lastDnnAct, lossFunc) = generateParams()
    val hiddenNode: Int = 128
    val denseLayerNum: Int = 1
    val lstmSize: Int = 256
    val startLr: Double = 5
    val lastDnnAct = Activation.TANH
    val lossFunc = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD

    logFile.write("" + hiddenNode + ", " + denseLayerNum + ", " + lstmSize + ", " + startLr + ", " + lastDnnAct + ", " + lossFunc)

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, startLr)
    lrShedule.put(100000, startLr / 2)
    lrShedule.put(200000, startLr / 4)

    val listBuilder = new NeuralNetConfiguration.Builder()
      .learningRate(10)
      .iterations(1)
      .seed(47)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAGRAD)
      .weightInit(WeightInit.XAVIER)
      .regularization(true)
      .l2(0.0005)
      .list()


    if(denseLayerNum > 1) {
      listBuilder.layer(0, new DenseLayer.Builder().nIn(PeerStateLen).nOut(hiddenNode).activation(Activation.RELU).build())
    }else if (denseLayerNum == 1) {
      listBuilder.layer(0, new DenseLayer.Builder().nIn(PeerStateLen).nOut(hiddenNode).activation(lastDnnAct).build())
    }


    for (i <- 1 until denseLayerNum - 1) {
      listBuilder.layer(i, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.RELU).build())
    }

    if (denseLayerNum > 1) {
      listBuilder.layer(denseLayerNum - 1, new DenseLayer.Builder().nOut(hiddenNode).activation(lastDnnAct).build())
    }



    listBuilder.layer(denseLayerNum, new GravesLSTM.Builder().nOut(lstmSize).activation(Activation.SOFTSIGN).build())
    listBuilder.layer(denseLayerNum + 1, new RnnOutputLayer.Builder(lossFunc). //MCXENT, NEGATIVELOGLIKELIHOOD
      activation(Activation.SOFTMAX).nOut(ActionLenWoAccept).build())


    val mlnConf = listBuilder.pretrain(false).backprop(true).build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    model
  }

  def trainCloudModel(model: MultiLayerNetwork): Unit = {
    val trainDir = "/home/ec2-user/tenhoulogs/logs/db4/dqrn/player/train"
    val validDir = "/home/ec2-user/tenhoulogs/logs/db4/dqrn/player/validation/"
    val modelFileName = "/home/ec2-user/tenhoulogs/logs/db4/dqrn/player/models/model"

    writeLog("===================================================> Train a DQRN")

    val batchSize = 128
    val epochNum = 2

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))

//    logger.info("Train model....");
//    model.setListeners(new ScoreIterationListener(1));


    val trainFetcher = new ReplayTransFetcherWoAccept(trainDir)
    val trainIte = new BaseDatasetIterator(batchSize, 1, trainFetcher)

    val validFetcher = new ReplayTransFetcherWoAccept(validDir)
    val validIte = new BaseDatasetIterator(batchSize, 1, validFetcher)
    var seq: Int = 1

    for (i <- 0 until epochNum) {
      trainIte.reset()
      model.fit(trainIte)

      validIte.reset()
      val eval = model.evaluate(validIte)
      //      val eval = new Evaluation()
      //      model.doEvaluation(validIte, eval)

      writeLog("=========================================> Evaluation")
      writeLog(eval.stats())

      val MJModelFile = new File(modelFileName + "_" + System.currentTimeMillis() + ".xml")
      val fos = new FileOutputStream(MJModelFile)
      ModelSerializer.writeModel(model, fos, true)
    }

    writeLog("::::::::::::::::::::::::::::::::::::::::: End of training")
  }

  def trainModel(model: MultiLayerNetwork, iterationNum: Int): Unit = {
//    val trainDir = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/train/"
//    val validDir = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/validation"
//    val trainDir = "/home/ec2-user/tenhoulogs/logs/db4/train/"
//    val validDir = "/home/ec2-user/tenhoulogs/logs/db4/validation/"
//    val trainDir = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/small/train"
//    val validDir = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/small/validation"
    val trainDir = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/player/train"
    val validDir = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/player/validation"

    val batchSize = 128
    val epochNum = 16

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))

//        logger.info("Train model....");
//        model.setListeners(new ScoreIterationListener(1));


    val trainFetcher = new ReplayTransFetcherWoAccept(trainDir)
    val trainIte = new BaseDatasetIterator(batchSize, iterationNum, trainFetcher)

    val validFetcher = new ReplayTransFetcherWoAccept(validDir)
    val validIte = new BaseDatasetIterator(batchSize, iterationNum, validFetcher)

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
//    trainModel(cnn_1Train())
//    randomDenseTrain()
//    trainModel(singleLayerTrain())
//    val iterationNum = 1
//    trainModel(createdqrn(iterationNum), iterationNum)
//    for (_ <- 0 until 20) {
      trainCloudModel(createClouddqrn())
//    }
  }

}
