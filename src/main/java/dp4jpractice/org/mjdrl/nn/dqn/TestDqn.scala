package dp4jpractice.org.mjdrl.nn.dqn

import java.io.{File, FileInputStream}

import akka.event.slf4j.Logger
import dp4jpractice.org.mjdrl.config.DqnSettings
import dp4jpractice.org.mjdrl.nn.dqn.nn.TenhouSimpleDenseQLDiscrete
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater, WorkspaceMode}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.network.dqn.DQN
import org.deeplearning4j.rl4j.util.DataManager
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.lossfunctions.LossFunctions
import tenhouclient.config.ClientConfig
import tenhouclient.impl.ImplConsts._
import tenhouclient.impl.mdp.{TenhouEncodableMdp, TenhouEncodableMdpFactory}

import scala.util.Random

object TestDqn {
  private val logger = Logger("TestDqn")
  
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

  def createModel(): MultiLayerNetwork = {
    val hiddenNode: Int = 128
    val denseLayerNum: Int = 1
    val lstmSize: Int = 256
    val startLr: Double = 5
    val lastDnnAct = Activation.TANH
    val lossFunc = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD

    logger.info("" + hiddenNode + ", " + denseLayerNum + ", " + lstmSize + ", " + startLr + ", " + lastDnnAct + ", " + lossFunc)

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, startLr)
    lrShedule.put(100000, startLr / 2)
    lrShedule.put(200000, startLr / 4)

    val listBuilder = new NeuralNetConfiguration.Builder()
//      .learningRate(10)
//      .iterations(1)
      .seed(47)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//      .updater(Updater.ADAGRAD)
      .updater(new AdaGrad(10))
      .weightInit(WeightInit.XAVIER)
//      .regularization(true)
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

//    listBuilder.layer(denseLayerNum + 1, new RnnOutputLayer.Builder(lossFunc). //MCXENT, NEGATIVELOGLIKELIHOOD
//      activation(Activation.SOFTMAX).nOut(ActionLenWoAccept).build())
    listBuilder.layer(denseLayerNum + 1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                      .nOut(ActionLenWoAccept).build())


    val mlnConf = listBuilder
//      .pretrain(false).backprop(true)
      .build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    model
  }

  def loadParams(model: MultiLayerNetwork): Unit = {
//    val modelFileName = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/player/outputs/models/model1_2.xml"
    val modelFileName = DqnSettings.LoadModelFileName
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
      1,     //size of batches = 1 as LSTM required
      500,    //target update (hard)
      2,     //num step noop warmup
      0.1,   //reward scaling
      0.99,   //gamma
      1.0,    //td-error clipping
      0.05f,   //min epsilon
      1000,   //num step for eps greedy anneal
      false    //double DQN
    );

  def createDqn(): TenhouSimpleDenseQLDiscrete = {
    val manager = new DataManager(true) //TODO: Why true?
//    val mdp = new TenhouEncodableMdp()

    val clientConfig = new ClientConfig(DqnSettings.TestNames, DqnSettings.A3CServerIP, DqnSettings.A3CServerPort, DqnSettings.LNLimit, DqnSettings.IsPrivateLobby)

    //        MDP mdp = new TenhouEncodableMdp(false, -1);
    val mdp = new TenhouEncodableMdpFactory(true, clientConfig)

    val model = createModel()
    loadParams(model)
    val dqn = new DQN(model)

    new TenhouSimpleDenseQLDiscrete(Test_QL, mdp, dqn, manager)
  }

  def main(args: Array[String]): Unit = {
    //  Thread.sleep(3000)
    val test = Nd4j.zeros(2.toLong, 2)
    logger.info(test.toStringFull)
    val dqn = createDqn()
    dqn.train()
  }
}
