package dp4jpractice.org.mjdrl.nn.dqn

import java.io.{File, FileInputStream}

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{ NeuralNetConfiguration, Updater, WorkspaceMode}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.network.dqn.DQN
import org.deeplearning4j.rl4j.util.DataManager
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import tenhouclient.impl.ImplConsts._
import tenhouclient.impl.mdp.TenhouEncodableMdp
import dp4jpractice.org.mjdrl.config.DqnSettings
import dp4jpractice.org.mjdrl.nn.dqn.nn.TenhouSimpleDenseQLDiscrete
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.schedule.{MapSchedule, ScheduleType}

import scala.util.Random

object TestCnnDqn {

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
    val hiddenNode: Int = 256
    //    val denseLayerNum: Int = 1
    //    val lstmSize: Int = 256
    val startLr: Double = 1
    //    val lastDnnAct = Activation.TANH
    //    val lossFunc = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD

    //    println("" + hiddenNode + ", " + denseLayerNum + ", " + lstmSize + ", " + startLr + ", " + lastDnnAct + ", " + lossFunc)

    val lrShedule: java.util.Map[Integer, java.lang.Double] = new java.util.HashMap[Integer, java.lang.Double]()
    lrShedule.put(0, startLr)
    lrShedule.put(4000, startLr / 2)
    lrShedule.put(10000, startLr / 4)

    val listBuilder = new NeuralNetConfiguration.Builder()
      .updater(new AdaGrad(new MapSchedule(ScheduleType.ITERATION, lrShedule)))
//      .learningRateDecayPolicy(LearningRatePolicy.Schedule)
//      .learningRateSchedule(lrShedule)
      //      .learningRate(0.05)
//      .iterations(1)
      .seed(47)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//      .updater(Updater.ADAGRAD)
      .weightInit(WeightInit.XAVIER)
//      .regularization(true)
      .l2(0.0005)
      .list()

    listBuilder.layer(0, new ConvolutionLayer.Builder(1, 3)
      //                            .padding(0, 1)
      .nIn(1)
      .nOut(hiddenNode)
      .activation(Activation.RELU)
      .build())
    listBuilder.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
      .kernelSize(1, 3)
      .build())
    //
    //    listBuilder.layer(2, new ConvolutionLayer.Builder(1, 2)
    //          .nOut(256)
    //          .activation(Activation.RELU)
    //          .build())
    //    listBuilder.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
    //          .kernelSize(1, 2)
    //          .build())

    listBuilder.layer(2, new DenseLayer.Builder().nOut(hiddenNode).activation(Activation.RELU).build())
    listBuilder.layer(3,
      new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(hiddenNode).nOut(ActionLenWoAccept).activation(Activation.IDENTITY).build())

    val mlnConf = listBuilder.setInputType(InputType.convolutionalFlat(1, PeerStateLen, 1))
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
      DqnSettings.MinEpsilon,   //min epsilon
      1000,   //num step for eps greedy anneal
      false    //double DQN
    );

  def createDqn(): TenhouSimpleDenseQLDiscrete = {
    val manager = new DataManager(true) //TODO: Why true?
    val mdp = new TenhouEncodableMdp()

    val model = createModel()
    loadParams(model)
    val dqn = new DQN(model)

    new TenhouSimpleDenseQLDiscrete(Test_QL, mdp, dqn, manager)
  }

  def main(args: Array[String]): Unit = {
    //  Thread.sleep(3000)
    val test = Nd4j.zeros(2.toLong, 2)
    println(test)

    //    val rewardFileName = "/home/ec2-user/tenhoulogs/dqntest/rewards/reward"
    //    val modelFileName = "/home/ec2-user/tenhoulogs/dqntest/models/model"
    //    RewardLogger.setFileName(rewardFileName)
    val dqn = createDqn()
    //    dqn.setModelFileName(modelFileName)
    dqn.train()
    //    createDqn().train()
  }
}
