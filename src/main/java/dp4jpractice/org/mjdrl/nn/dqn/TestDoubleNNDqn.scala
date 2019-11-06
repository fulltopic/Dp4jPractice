package dp4jpractice.org.mjdrl.nn.dqn

import java.io.{File, FileInputStream}

import akka.event.slf4j.Logger
import dp4jpractice.org.mjdrl.config.DqnSettings
import dp4jpractice.org.mjdrl.nn.dqn.nn.{TenhouSimpleDenseQLDiscrete, TenhouSimpleDoubleDenseQLDiscrete}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater, WorkspaceMode}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.network.dqn.DQN
import org.deeplearning4j.rl4j.util.DataManager
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions
import tenhouclient.config.ClientConfig
import tenhouclient.impl.ImplConsts.{ActionLenWoAccept, PeerStateLen}
import tenhouclient.impl.mdp.{TenhouEncodableMdp, TenhouEncodableMdpFactory}

object TestDoubleNNDqn {
  private val logger = Logger("TestDoubleNNDqn")

  def createModel(): MultiLayerNetwork = {
    val hiddenNode: Int = 128
    val denseLayerNum: Int = 1
    val startLr: Double = 0.01
    val lossFunc = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD

    logger.info("" + hiddenNode + ", " + denseLayerNum + ", " + startLr  + ", " + lossFunc)

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, startLr)
    //    lrShedule.put(20, startLr / 2)
    //    lrShedule.put(230, 0.1)
    //    lrShedule.put(230, 1.0)

    val listBuilder = new NeuralNetConfiguration.Builder()
      //      .learningRateDecayPolicy(LearningRatePolicy.Schedule)
      //      .learningRateSchedule(lrShedule)
//      .learningRate(startLr)
//      .iterations(1)
      .seed(47)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//      .updater(Updater.RMSPROP)
      .updater(new RmsProp(startLr))
      .weightInit(WeightInit.XAVIER)
//      .regularization(true)
      .l2(0.0005)
      .list()

    listBuilder.layer(0, new DenseLayer.Builder().nIn(PeerStateLen).nOut(hiddenNode).activation(Activation.RELU).build())
    listBuilder.layer(1, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.RELU).build())
    listBuilder.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(ActionLenWoAccept).activation(Activation.IDENTITY).build())


    val mlnConf = listBuilder
//      .pretrain(false).backprop(true)
      .build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))


    model
  }

  def loadParams(model: MultiLayerNetwork): Unit = {
    val modelFileName = DqnSettings.LoadNNModelFileName
    var fis: FileInputStream = null

    try {
      fis = new FileInputStream(new File(modelFileName))
    }catch {
      case e: Throwable =>
        e.printStackTrace()
        fis = null
    }

    if (fis != null) {
      val superModel = ModelSerializer.restoreMultiLayerNetwork(fis)
      val superParams = superModel.paramTable
      model.setParamTable(superParams)
    }

  }


  val Test_QL =
    new QLearning.QLConfiguration(
      123,    //Random seed
      200,    //Max step By epoch
      150000, //Max step
      150000, //Max size of experience replay
      20,     //size of batches = 1 as LSTM required
      50,    //target update (hard)
      0,     //num step noop warmup
      0.1,   //reward scaling
      0.99,   //gamma
      1.0,    //td-error clipping
      0.05f,   //min epsilon
      1000,   //num step for eps greedy anneal
      true    //double DQN
    );

  def createDqn(): TenhouSimpleDenseQLDiscrete = {
    val manager = new DataManager(true) //TODO: Why true?
//    val mdp = new TenhouEncodableMdp()
    val clientConfig = new ClientConfig(DqnSettings.TestNames, DqnSettings.A3CServerIP, DqnSettings.A3CServerPort, DqnSettings.LNLimit, DqnSettings.IsPrivateLobby)
    val mdp = new TenhouEncodableMdpFactory(true, clientConfig)

//    val model = createModel()
//    loadParams(model)
    val model = ModelSerializer.restoreMultiLayerNetwork(DqnSettings.LoadNNModelFileName)
    val dqn = new DQN(model)

    new TenhouSimpleDoubleDenseQLDiscrete(Test_QL, mdp, dqn, manager)
  }

  def main(args: Array[String]): Unit = {
    //  Thread.sleep(3000)
    val test = Nd4j.zeros(2.toLong, 2)
    logger.info(test.toStringFull)
    val dqn = createDqn()
    dqn.train()
  }
}
