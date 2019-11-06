package dp4jpractice.org.mjdrl.nn.dqn

import java.io.{File, FileInputStream}

import akka.event.slf4j.Logger
import dp4jpractice.org.mjdrl.config.DqnSettings
import dp4jpractice.org.mjdrl.nn.dqn.nn.lstm.{LstmDqn, TenhouLstmQLDiscrete}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.util.DataManager
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import tenhouclient.config.ClientConfig
import tenhouclient.impl.ImplConsts._
import tenhouclient.impl.mdp.{TenhouEncodableMdp, TenhouEncodableMdpFactory}

object TestLstmDqn {
  private val logger = Logger("TestLstmDqn")
//  def getRandomParam[T](params: Array[T]): T = {
//    params(Random.nextInt(params.length))
//  }

//  def generateParams(): (Int, Int, Int, Double, Activation, LossFunctions.LossFunction) = {
//    val hiddenNodes = Array[Int](64, 128, 256, 512)
//    val denseLayerNums = Array[Int](1, 2, 3)
//    val lstmSizes = Array[Int](64, 128, 256, 512)
//    val startLr = Array[Double](10.0, 8.0, 5.0, 1.0)
//    val lastDnnActivations = Array[Activation](Activation.RELU, Activation.TANH, Activation.SOFTSIGN)
//    val outputLossFunc = Array[LossFunctions.LossFunction](LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//
//    (getRandomParam(hiddenNodes), getRandomParam(denseLayerNums), getRandomParam(lstmSizes), getRandomParam(startLr), getRandomParam(lastDnnActivations), getRandomParam(outputLossFunc))
//  }

  def createModel(): MultiLayerNetwork = {
    val lstmLayerSize: Int = 256

    val conf = new NeuralNetConfiguration.Builder().
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//      .iterations(1).learningRate(0.05)
      .seed(12345)
//        .regularization(true)
      .l2(0.001)
      .dropOut(0.5)
      .weightInit(WeightInit.DISTRIBUTION)
//      .updater(Updater.RMSPROP)
      .updater(new RmsProp(0.05))
      .list
      .layer(0, new GravesLSTM.Builder().nIn(PeerStateLen).nOut(lstmLayerSize).activation(Activation.TANH).build)
      .layer(1, new GravesLSTM.Builder().nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE).activation(Activation.IDENTITY)        //MCXENT + softmax for classification
      .nIn(lstmLayerSize).nOut(ActionLenWoAccept).build())
//      .pretrain(false)
//      .backprop(true)
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()


    model
  }

  def loadParams(model: MultiLayerNetwork): Unit = {
    //    val modelFileName = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/player/outputs/models/model1_2.xml"
    val modelFileName = DqnSettings.LoadModelLstmFileName
    logger.info("---------------------------------------> Load model " + modelFileName)
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
      Int.MaxValue,    //target update (hard) //TODO
      2,     //num step noop warmup
      0.025,   //reward scaling
      0.99,   //gamma
      1.0,    //td-error clipping
      0.05f,   //min epsilon
      1000,   //num step for eps greedy anneal
      false    //double DQN
    );


  def createDqn(): TenhouLstmQLDiscrete = {
    val manager = new DataManager(true) //TODO: Why true?
//    val mdp = new TenhouEncodableMdp()
    val clientConfig = new ClientConfig(DqnSettings.TestNames, DqnSettings.A3CServerIP, DqnSettings.A3CServerPort, DqnSettings.LNLimit, DqnSettings.IsPrivateLobby)

    //        MDP mdp = new TenhouEncodableMdp(false, -1);
    val mdp = new TenhouEncodableMdpFactory(true, clientConfig)

//    val model = createModel()
//    loadParams(model)

    val model = ModelSerializer.restoreMultiLayerNetwork(DqnSettings.LoadModelLstmFileName)
    val dqn = new LstmDqn(model)

    new TenhouLstmQLDiscrete(Test_QL, mdp, dqn, manager)
  }

  def main(args: Array[String]): Unit = {
    //  Thread.sleep(3000)
    val test = Nd4j.zeros(2.toLong, 2)
    logger.info(test.toStringFull)
    val dqn = createDqn()
    dqn.train()
  }
}
