package rl.dqn.reinforcement.supervised

import java.io.{File, FileOutputStream, PrintWriter}
import java.util

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, GradientNormalization, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.preprocessor.{CnnToFeedForwardPreProcessor, FeedForwardToCnnPreProcessor, FeedForwardToRnnPreProcessor, RnnToCnnPreProcessor}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.rl4j.network.ac.{ActorCriticFactoryCompGraphStdDense, ActorCriticLoss}
import org.deeplearning4j.rl4j.util.Constants
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import rl.dqn.reinforcement.dqn.nn.a3c.{A3CTenhouModelFactory, TenhouFF2CnnPreProcessor, TenhouInputPreProcessor}
import rl.dqn.reinforcement.dqn.nn.datapocess.{TenhouCompCnnLstmIterator, TenhouCompMaskCnnLstmIterator, TenhouLstmIterator, TenhouXmlCnnFetcher}
//import rl.dqn.reinforcement.dqn.test.TestSupervisedLstm.{trainPath, validPath}
import rl.dqn.reinforcement.dqn.config.Supervised._

object SupervisedComp extends App{
  private def getConfig(lstmTraining: Boolean): ActorCriticFactoryCompGraphStdDense.Configuration = {
    val config = ActorCriticFactoryCompGraphStdDense.Configuration.builder
      .learningRate(1e-2)
      .l2(0)
      .numHiddenNodes(128)
      .numLayer(2)
      .useLSTM(lstmTraining)
      .build

    config
  }


  var trainPath: String = ""
  var validPath: String = ""
  var modelPath: String = ""
  var logFile: PrintWriter = null;

  def initLocalPath(): Unit = {
    trainPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/train/"
//    trainPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/smalltest"
    validPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/validation/"
    modelPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/supervised/models/supervisedmodel"
    logFile = new PrintWriter(new File("/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/supervised/evals/evals_dense_conv_dense_lstm" + System.currentTimeMillis() + ".txt"))
  }

  def writeLog(content: String): Unit = {
    logFile.write(content + "\n")
    logFile.flush()
  }

  def train(): Unit = {
    val batchSize = 64
    val iterationNum = 1
    val epochNum = 16

    val trainIte = new TenhouCompMaskCnnLstmIterator(trainPath, batchSize, true)
    val validIte = new TenhouCompMaskCnnLstmIterator(validPath, batchSize, true)

//    val trainFetcher = new TenhouXmlCnnFetcher(trainPath, TileNum, true)
//    val trainIte = new BaseDatasetIterator(batchSize, iterationNum, trainFetcher)
//    val validFetcher = new TenhouXmlCnnFetcher(validPath, TileNum, true)
//    val validIte = new BaseDatasetIterator(batchSize, iterationNum, validFetcher)

//    val model = CreatePreProcessorActorCritic(getConfig(true), 74, 42, true)
    val model = A3CTenhouModelFactory.CreateLstmComputeGraph(getConfig(true), 74, 42, true);

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))

    for (i <- 0 until epochNum) {
      trainIte.reset()
      model.fit(trainIte)

      validIte.reset()
      val eval = model.evaluate(validIte)
      //      val eval = new Evaluation()
      //      model.doEvaluation(validIte, eval)


      val stat = eval.stats()
      writeLog("=========================================> Evaluation")
      writeLog(stat)
      println(stat)

      val MJModelFile = new File(modelPath + "_" + System.currentTimeMillis() + ".xml")
      val fos = new FileOutputStream(MJModelFile)
      ModelSerializer.writeModel(model, fos, true)

//      Nd4j.getWorkspaceManager.destroyAllWorkspacesForCurrentThread()
    }

    writeLog("::::::::::::::::::::::::::::::::::::::::: End of training")
  }

  def test(): Unit = {
    val batchSize: Int = 64
    val testPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/smalltest/"
    val testIte = new TenhouCompMaskCnnLstmIterator(testPath, batchSize, true)

    val modelPath = "/home/zf/workspaces/workspace_java/mjpratice/docs/supervisedgraph/9/nnmodel_1536430104600.xml"
//    val modelPath = "/home/zf/workspaces/workspace_java/mjpratice/docs/supervisedgraph/8/nnmodel_1536417087462.xml"
//    val modelPath = "/home/zf/workspaces/workspace_java/mjpratice/docs/supervisedgraph/2/nnmodel_1536131828774.xml"
//    val modelPath = "/home/zf/workspaces/workspace_java/mjpratice/docs/supervisedgraph/3/nnmodel_1536198964481.xml"

    val model = ModelSerializer.restoreComputationGraph(modelPath, true);

    val eval = model.evaluate(testIte)
    println(eval)
  }

  def testRnn(): Unit = {
    val batchSize: Int = 1
    val testPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/minitest/"
    val testIte = new TenhouCompMaskCnnLstmIterator(testPath, batchSize, true)


    val modelPath = "/home/zf/workspaces/workspace_java/mjpratice/docs/supervisedgraph/9/nnmodel_1536430104600.xml"
    val model = ModelSerializer.restoreComputationGraph(modelPath, true);
    val expCount: Int = Int.MaxValue
    var count: Int = 0
    var totalExample: Int = 0
    var matchExample: Int = 0

    while(testIte.hasNext && count < expCount) {
      val next = testIte.next()
      val feature = next.getFeatureMatrix
      val label = next.getLabels
      val mask = next.getFeaturesMaskArray

//      println(feature.shape().mkString(", "))
//      println(mask.shape().mkString(", "))
      for (i <- 0 until feature.shape()(2)) {
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

//      println("++++++++++++++++++++++++ Label " + label)


      model.rnnClearPreviousState()
      count += 1
    }

    println("" + matchExample + " / " + totalExample)
//    val eval = model.evaluate(testIte)
//    println(eval)
  }

  initLocalPath()
  train()
//  test()
//  testRnn()
}
