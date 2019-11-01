package dp4jpractice.org.mjsupervised

import java.io.{File, FileOutputStream, PrintWriter}
import java.util

import dp4jpractice.org.mjsupervised.dataprocess.iterator.TenhouCompMaskCnnLstmIterator
import dp4jpractice.org.mjsupervised.dataprocess.preprocessor.{TenhouFF2CnnPreProcessor, TenhouInputPreProcessor}
import dp4jpractice.org.mjsupervised.nn.A3CTenhouModelFactory
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.preprocessor.{CnnToFeedForwardPreProcessor, FeedForwardToCnnPreProcessor, FeedForwardToRnnPreProcessor}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.{InvocationType, IterationListener, TrainingListener}
import org.deeplearning4j.optimize.listeners.{EvaluativeListener, ScoreIterationListener}
import org.deeplearning4j.rl4j.network.ac.{ActorCriticFactoryCompGraphStdDense, ActorCriticLoss}
import org.deeplearning4j.rl4j.util.Constants
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

object SupervisedComp extends App{
  private val logger = LoggerFactory.getLogger(SupervisedComp.getClass)

  def CreateActorCritic(netConf: ActorCriticFactoryCompGraphStdDense.Configuration, numInputs: Int, numOutputs: Int, supervised: Boolean): ComputationGraph = {
    var inputType = InputType.feedForward(numInputs)
    if (!supervised) inputType = InputType.recurrent(numInputs)
    val confB = new NeuralNetConfiguration.Builder()
      .seed(Constants.NEURAL_NET_SEED)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(netConf.getUpdater)
//      .updater(new Adam(netConf.getLearningRate)) //.updater(Updater.NESTEROVS).momentum(0.9)
      .weightInit(WeightInit.XAVIER)
      .l2(netConf.getL2)
      .graphBuilder.addInputs("input")
      .addLayer("conv0", new ConvolutionLayer.Builder(1, 4).nOut(8).activation(Activation.RELU).build, "input")

//    confB.addLayer("conv1", new ConvolutionLayer.Builder(1, 4).nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "conv0")
    confB.addLayer("dense0", new DenseLayer.Builder().nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "conv0")
//    confB.addLayer("dense1", new DenseLayer.Builder().nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "dense0")

    if (supervised) {
      confB.addLayer("softmax", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(numOutputs).build, "dense0")
      confB.setOutputs("softmax")
    }
    else {
      confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(1).build, 2 + "")
      confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss).activation(Activation.SOFTMAX).nOut(numOutputs).build, 2 + "")
      confB.setOutputs("value", "softmax")
    }


    confB.setInputTypes(InputType.convolutionalFlat(1, 74, 1))

    val graphConf = confB.asInstanceOf[ComputationGraphConfiguration.GraphBuilder]

    //        confB.inputPreProcessor("0", new TenhouInputPreProcessor());

    val cgconf = graphConf.build
    val model = new ComputationGraph(cgconf)
    model.init()

    if (netConf.getListeners != null) {
      val listeners = new util.ArrayList[TrainingListener]()
      for (listener <- netConf.getListeners) {
        listeners.add(listener)
      }
      model.setListeners(listeners)
    } else model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER))
    model
  }

  def CreatePreProcessorActorCritic(netConf: ActorCriticFactoryCompGraphStdDense.Configuration, numInputs: Int, numOutputs: Int, supervised: Boolean): ComputationGraph = {
    var inputType = InputType.feedForward(numInputs)
    if (!supervised) inputType = InputType.recurrent(numInputs)

    val kernelW = 4
    val originW = 74
    val hiddenKernels = 8
    var convNum: Int = 0

    val confB = new NeuralNetConfiguration.Builder()
      .seed(Constants.NEURAL_NET_SEED)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(netConf.getUpdater)
//      .updater(new Adam(netConf.getLearningRate)) //.updater(Updater.NESTEROVS).momentum(0.9)
      //.updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
      .weightInit(WeightInit.XAVIER)
      .l2(netConf.getL2)
      .graphBuilder.addInputs("input")
      .addLayer("conv0", new ConvolutionLayer.Builder(1, kernelW).nIn(1).nOut(hiddenKernels).activation(Activation.RELU).build, "input")

    convNum += 1

    confB.addLayer("conv1", new ConvolutionLayer.Builder(1, kernelW).nIn(hiddenKernels).nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "conv0")
    convNum += 1

//    println(originW - (kernelW - 1) * convNum)
    confB.addLayer("dense0", new DenseLayer.Builder().nIn((originW - (kernelW - 1) * convNum) * netConf.getNumHiddenNodes).nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "conv1")

    val lstmLayer = new GravesLSTM.Builder().nIn(netConf.getNumHiddenNodes).nOut(netConf.getNumHiddenNodes).activation(Activation.TANH).build()
    confB.addLayer("lstm0", lstmLayer, "dense0")

    if (supervised) {

      confB.addLayer("softmax", new RnnOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(netConf.getNumHiddenNodes).nOut(numOutputs).build, "lstm0")
      confB.setOutputs("softmax")
    }
    else {
      confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(1).build, 2 + "")
      confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss).activation(Activation.SOFTMAX).nOut(numOutputs).build, 2 + "")
      confB.setOutputs("value", "softmax")
    }


//    confB.setInputTypes(InputType.convolutionalFlat(1, 74, 1))

//    confB.inputPreProcessor("conv0", new RnnToCnnPreProcessor(1, 74, 1))
    confB.inputPreProcessor("conv0", new TenhouFF2CnnPreProcessor(1, originW, 1))
    confB.inputPreProcessor("dense0", new CnnToFeedForwardPreProcessor(1, originW - (kernelW - 1) * convNum, netConf.getNumHiddenNodes))
    confB.inputPreProcessor("lstm0", new FeedForwardToRnnPreProcessor)
//    confB.inputPreProcessor("densepre", new TenhouInputPreProcessor())

    val graphConf = confB.asInstanceOf[ComputationGraphConfiguration.GraphBuilder]


    val cgconf = graphConf.build
    cgconf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)

    val model = new ComputationGraph(cgconf)
    model.init()

    if (netConf.getListeners != null) {
      val listeners = new util.ArrayList[TrainingListener]()

      for (listener <- netConf.getListeners) {
        listeners.add(listener)
      }
      model.setListeners(listeners)
    } else model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER))
    model
  }

  def CreateLstmActorCritic(netConf: ActorCriticFactoryCompGraphStdDense.Configuration, numInputs: Int, numOutputs: Int, supervised: Boolean): ComputationGraph = {
    var inputType = InputType.feedForward(numInputs)
    if (!supervised) inputType = InputType.recurrent(numInputs)

    val kernelW = 4
    val originW = 74
    val hiddenKernels = 8
    var convNum: Int = 0

    val confB = new NeuralNetConfiguration.Builder()
      .seed(Constants.NEURAL_NET_SEED)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(netConf.getUpdater)
//      .updater(new Adam(netConf.getLearningRate)) //.updater(Updater.NESTEROVS).momentum(0.9)
      //.updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
      .weightInit(WeightInit.XAVIER)
      .l2(netConf.getL2)
      .graphBuilder.addInputs("input")
      .addLayer("densepre", new DenseLayer.Builder().nIn(originW).nOut(originW).activation(Activation.IDENTITY).build(), "input")
      .addLayer("conv0", new ConvolutionLayer.Builder(1, kernelW).nIn(1).nOut(hiddenKernels).activation(Activation.RELU).build, "densepre")

    convNum += 1

    confB.addLayer("conv1", new ConvolutionLayer.Builder(1, kernelW).nIn(hiddenKernels).nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "conv0")
    convNum += 1

    confB.addLayer("dense0", new DenseLayer.Builder().nIn((originW - (kernelW - 1) * convNum) * netConf.getNumHiddenNodes).nOut(netConf.getNumHiddenNodes).activation(Activation.RELU).build, "conv1")

    val lstmLayer = new GravesLSTM.Builder().nIn(netConf.getNumHiddenNodes).nOut(netConf.getNumHiddenNodes).activation(Activation.TANH).build()
    confB.addLayer("lstm0", lstmLayer, "dense0")

    if (supervised) {

      confB.addLayer("softmax", new RnnOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(netConf.getNumHiddenNodes).nOut(numOutputs).build, "lstm0")
      confB.setOutputs("softmax")
    }
    else {
      confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(1).build, 2 + "")
      confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss).activation(Activation.SOFTMAX).nOut(numOutputs).build, 2 + "")
      confB.setOutputs("value", "softmax")
    }


    //    confB.setInputTypes(InputType.convolutionalFlat(1, 74, 1))

    //    confB.inputPreProcessor("conv0", new RnnToCnnPreProcessor(1, 74, 1))
    confB.inputPreProcessor("conv0", new FeedForwardToCnnPreProcessor(1, originW, 1))
    confB.inputPreProcessor("dense0", new CnnToFeedForwardPreProcessor(1, originW - (kernelW - 1) * convNum, netConf.getNumHiddenNodes))
    confB.inputPreProcessor("lstm0", new FeedForwardToRnnPreProcessor)
    confB.inputPreProcessor("densepre", new TenhouInputPreProcessor())

    val graphConf = confB.asInstanceOf[ComputationGraphConfiguration.GraphBuilder]


    val cgconf = graphConf.build
    val model = new ComputationGraph(cgconf)
    model.init()

    if (netConf.getListeners != null) {
      val listeners = new util.ArrayList[TrainingListener]()

      for (listener <- netConf.getListeners) {
        listeners.add(listener)
      }
      model.setListeners(listeners)
    } else model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER))
    model
  }

  private def getConfig(lstmTraining: Boolean): ActorCriticFactoryCompGraphStdDense.Configuration = {
    val config = ActorCriticFactoryCompGraphStdDense.Configuration.builder

      .updater(new Adam(1e-2))
//      .learningRate(1e-2)
      .l2(0)
      .numHiddenNodes(128)
      .numLayer(2)
      .useLSTM(lstmTraining)
      .build

    config
  }



  private val ResourcePath = "datasets/"
  private val resourceDir = new File(ResourcePath)
  private val resourcePath = resourceDir.getAbsolutePath
  private val trainPath = resourcePath + "/mjsupervised/xmlfiles/train/"
  private val validPath = resourcePath + "/mjsupervised/xmlfiles/validation/"
  private val testPath = resourcePath + "/mjsupervised/xmlfiles/minitest/"
  private val modelPath = resourcePath + "/mjsupervised/xmlfiles/models/nnmodel"
  private val logFile = new PrintWriter(new File(resourcePath + "/mjsupervised/logs/evals_dense_conv_dense_lstm" + System.currentTimeMillis() + ".txt"))

  def writeLog(content: String): Unit = {
    logFile.write(content + "\n")
    logFile.flush()
  }

  def train(): Unit = {
    val batchSize = 64
    val epochNum = 2

    val trainIte = new TenhouCompMaskCnnLstmIterator(trainPath, batchSize, true)
    val validIte = new TenhouCompMaskCnnLstmIterator(validPath, batchSize, true)
    val testIte = new TenhouCompMaskCnnLstmIterator(testPath, batchSize, true)

    val model = A3CTenhouModelFactory.CreateLstmComputeGraph(getConfig(true), 74, 42, true);

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage), new EvaluativeListener(validIte, 1, InvocationType.EPOCH_END))

    model.fit(trainIte, epochNum)


    writeLog("::::::::::::::::::::::::::::::::::::::::: End of training")
//    uiServer.detach(statsStorage)
  }

  def test(): Unit = {
    val batchSize: Int = 64
    val testPath = resourcePath + "/mjsupervised/xmlfiles/smalltest/"
    val testIte = new TenhouCompMaskCnnLstmIterator(testPath, batchSize, true)

    val modelPath = "docs/supervisedgraph/9/nnmodel_1536430104600.xml"

    val model = ModelSerializer.restoreComputationGraph(modelPath, true);

    val eval = model.evaluate(testIte)
    println(eval)
  }

  def testRnn(): Unit = {
    val batchSize: Int = 1
    val testPath = resourcePath + "/mjsupervised/xmlfiles/xmlfiles/minitest/"
    val testIte = new TenhouCompMaskCnnLstmIterator(testPath, batchSize, true)


    val modelPath = "docs/supervisedgraph/9/nnmodel_1536430104600.xml"
    val model = ModelSerializer.restoreComputationGraph(modelPath, true);
    val expCount: Int = Int.MaxValue
    var count: Int = 0
    var totalExample: Int = 0
    var matchExample: Int = 0

    while(testIte.hasNext && count < expCount) {
      val next = testIte.next()
      val feature = next.getFeatures
      val label = next.getLabels
      val mask = next.getFeaturesMaskArray

      for (i <- 0 until feature.shape()(2).toInt) {
        if (mask.getInt(0, i) > 0) {
          val input = Nd4j.create(Array[Int](1, 74, 1), 'f')
          for (j <- 0 until 74) {
            input.putScalar(0, j, 0, feature.getDouble(0, j, i))
          }
          val output = model.rnnTimeStep(input)(0)
          println(">>>>>>>>>>>>>>>>>>>>>>>> Output " + output.shape().mkString(", ") + " " + output)

          var maxAction: Int = -1
          var maxV: Double = 0
          var maxLabel: Int = -1
          for (j <- 0 until 42) {
            val v = output.getDouble(0, j, 0)
            if (v > maxV) {
              maxV = v
              maxAction = j
            }
            if (label.getDouble(0, j, i) > 0.5) {
              maxLabel = j
            }
          }

          totalExample += 1
          if (maxAction == maxLabel) {
            matchExample += 1
          }
          println("" + maxV + ", " + maxAction + " : " + maxLabel)
        }
      }


      model.rnnClearPreviousState()
      count += 1
    }

    println("" + matchExample + " / " + totalExample)
  }

  train()
//  test()
//  testRnn()
}
