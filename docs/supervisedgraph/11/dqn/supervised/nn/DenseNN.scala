package rl.dqn.supervised.nn

import java.io.File
import java.util

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{LearningRatePolicy, MultiLayerConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rl4j.network.dqn.{DQN, DQNFactory, DQNFactoryStdDense}
import org.deeplearning4j.rl4j.util.Constants
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

object DenseNN {
  def buildDQN(conf: DQNFactoryStdDense.Configuration, numInputs: Array[Int], numOutputs: Int): DQN[_] = {
    val factory = new DQNFactoryStdDense(conf)
    factory.buildDQN(numInputs, numOutputs)
  }

  def loadDQN(fileName:  String): DQN[_] = {
    val modelFile = new File(fileName)
    val nn: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    new DQN(nn)
  }

  def createDqnOutput(conf: DQNFactoryStdDense.Configuration, numOutputs: Int): OutputLayer
      = new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
    .nIn(conf.getNumHiddenNodes).nOut(numOutputs).build()

  def createSupervisedOutput(conf: DQNFactoryStdDense.Configuration, numOutputs: Int): OutputLayer
    = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(numOutputs).activation(Activation.SOFTMAX).build()


  def createNNPart(conf: DQNFactoryStdDense.Configuration, numInputs: Array[Int]): NeuralNetConfiguration.ListBuilder = {
    var nIn = 1
    for (i <- numInputs) {
      nIn *= i
    }

//    val lrShedule: util.Map[Integer, java.lang.Double] = new util.HashMap[Integer, java.lang.Double]()
//    lrShedule.put(0, 0.1)
//    lrShedule.put(10000, 0.05)
//    lrShedule.put(10000, 0.005)
//    lrShedule.put(20000, 0.01)
    val listBuilder = new NeuralNetConfiguration.Builder()
//        .learningRateDecayPolicy(LearningRatePolicy.Inverse)
//      .lrPolicyDecayRate(0.0001)
//        .setLrPolicyDecayRate(0.0001)
//        .learningRateSchedule(lrShedule)
      .learningRate(conf.getLearningRate)
      .seed(Constants.NEURAL_NET_SEED)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(conf.getUpdater)
      .weightInit(WeightInit.XAVIER)
      .regularization(true)
      .l2(conf.getL2)
      .list()

    listBuilder.layer(0, new DenseLayer.Builder().nIn(nIn).nOut(conf.getNumHiddenNodes).activation(Activation.RELU).build())

    for (i <- 1 until conf.getNumLayer) {
      listBuilder.layer(i, new DenseLayer.Builder().nIn(conf.getNumHiddenNodes).nOut(conf.getNumHiddenNodes).activation(Activation.RELU).build())
    }

    listBuilder
  }

  def createNN(conf: DQNFactoryStdDense.Configuration, numInputs: Array[Int], numOutputs: Int, outputLayer: OutputLayer): MultiLayerNetwork = {
    val listBuilder = createNNPart(conf, numInputs)
    listBuilder.layer(conf.getNumLayer, outputLayer)
    val mlnConf = listBuilder.pretrain(false).backprop(true).build
    mlnConf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE)
    val model = new MultiLayerNetwork(mlnConf)
    model.init()

    model
  }


  // set/get paramTable
  def createDqnNN(conf: DQNFactoryStdDense.Configuration, numInputs: Array[Int], numOutputs: Int): MultiLayerNetwork =
    createNN(conf, numInputs, numOutputs, createDqnOutput(conf, numOutputs))

  def createSupervisedNN(conf: DQNFactoryStdDense.Configuration, numInputs: Array[Int], numOutputs: Int): MultiLayerNetwork =
    createNN(conf, numInputs, numOutputs, createSupervisedOutput(conf, numOutputs))


}
