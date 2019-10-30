package dp4jpractice.org.mjsupervised.nn;

import dp4jpractice.org.mjsupervised.dataprocess.preprocessor.TenhouInputPreProcessor;
import dp4jpractice.org.mjsupervised.dataprocess.preprocessor.TenhouRnnToCnnPreProcessor;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.network.ac.ActorCriticLoss;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class A3CTenhouModelFactory {
    static Logger StaticLogger = LoggerFactory.getLogger(A3CTenhouModelFactory.class);

    public static ComputationGraph CreateDenseComputeGraph(ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) {
        StaticLogger.info("Create dense model");

        ComputationGraphConfiguration.GraphBuilder confB =
                new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                        .learningRate(netConf.getLearningRate())
                        //.updater(Updater.NESTEROVS).momentum(0.9)
//                        .updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
                        .updater(netConf.getUpdater() != null ? netConf.getUpdater() : new Adam(1e-2))
                        .weightInit(WeightInit.XAVIER)
//                        .regularization(netConf.getL2() > 0)
                        .l2(netConf.getL2()).graphBuilder()
                        .setInputTypes(netConf.isUseLSTM() ? InputType.recurrent(numInputs)
                                : InputType.feedForward(numInputs)).addInputs("input")
                        .addLayer("0", new DenseLayer.Builder().nIn(numInputs)
                                        .nOut(netConf.getNumHiddenNodes()).activation(Activation.RELU).build(),
                                "input");


        for (int i = 1; i < netConf.getNumLayer(); i++) {
            confB.addLayer(i + "", new DenseLayer.Builder().nIn(netConf.getNumHiddenNodes()).nOut(netConf.getNumHiddenNodes())
                    .activation(Activation.RELU).build(), (i - 1) + "");
        }


        if (netConf.isUseLSTM()) {
            confB.addLayer(netConf.getNumLayer() + "", new LSTM.Builder().activation(Activation.TANH)
                    .nOut(netConf.getNumHiddenNodes()).build(), (netConf.getNumLayer() - 1) + "");

            confB.addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                    .nOut(1).build(), netConf.getNumLayer() + "");

            confB.addLayer("softmax", new RnnOutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                    .nOut(numOutputs).build(), netConf.getNumLayer() + "");
        } else {
            confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                    .nOut(1).build(), (netConf.getNumLayer() - 1) + "");

            confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                    .nOut(numOutputs).build(), (netConf.getNumLayer() - 1) + "");
        }

        confB.setOutputs("value", "softmax");

        confB.inputPreProcessor("0", new TenhouInputPreProcessor());
//        Map<String, InputPreProcessor> processorMap  = confB.getInputPreProcessors();
//        for (Map.Entry<String, InputPreProcessor> entry: processorMap.entrySet()) {
//            System.out.println(entry.getKey() + ": " + entry.getValue().getClass());
//        }


        ComputationGraphConfiguration cgconf = confB
//                                                .pretrain(false)
//                                                .backprop(true)
                                                .build();
        cgconf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);


        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();

        return model;
    }

    public static ComputationGraph CreateLstmComputeGraph(ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs, boolean supervised) {
        StaticLogger.info("Create lstm model "  + supervised);

        InputType inputType = InputType.feedForward(numInputs);
//        if (!supervised) inputType = InputType.recurrent(numInputs)

        int kernelW = 4;
        int originW = 74;
        int hiddenKernels = 8;
        int convNum = 0;

        ComputationGraphConfiguration.GraphBuilder confB = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
//                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .learningRate(netConf.getLearningRate())
                .updater(netConf.getUpdater()) //.updater(Updater.NESTEROVS).momentum(0.9)
                //.updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
//                .regularization(netConf.getL2() > 0)
                .l2(netConf.getL2())
                .graphBuilder().addInputs("input")
                .addLayer("conv0", new ConvolutionLayer.Builder(1, kernelW).nIn(1).nOut(hiddenKernels).activation(Activation.RELU).build(), "input");

        convNum ++;

        confB.addLayer("conv1", new ConvolutionLayer.Builder(1, kernelW).nIn(hiddenKernels).nOut(netConf.getNumHiddenNodes()).activation(Activation.RELU).build(), "conv0");
        convNum ++;

//    println(originW - (kernelW - 1) * convNum)
        confB.addLayer("dense0", new DenseLayer.Builder().nIn((originW - (kernelW - 1) * convNum) * netConf.getNumHiddenNodes()).nOut(netConf.getNumHiddenNodes()).activation(Activation.RELU).build(), "conv1");

        GravesLSTM lstmLayer = new GravesLSTM.Builder().nIn(netConf.getNumHiddenNodes()).nOut(netConf.getNumHiddenNodes()).activation(Activation.TANH).build();
        confB.addLayer("lstm0", lstmLayer, "dense0");


        if (supervised) {
            confB.addLayer("softmax", new RnnOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(netConf.getNumHiddenNodes()).nOut(numOutputs).build(), "lstm0");
            confB.setOutputs("softmax");

            confB.inputPreProcessor("conv0", new TenhouRnnToCnnPreProcessor(1, originW, 1));
            confB.inputPreProcessor("dense0", new CnnToFeedForwardPreProcessor(1, originW - (kernelW - 1) * convNum, netConf.getNumHiddenNodes()));
//            confB.inputPreProcessor("lstm0", new ZeroMeanAndUnitVariancePreProcessor());
            confB.inputPreProcessor("lstm0", new FeedForwardToRnnPreProcessor());
        }else {
            System.out.println("------------------------> Created model with preprocessor");
            confB.addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(netConf.getNumHiddenNodes()).nOut(1).build(), "lstm0");
            confB.addLayer("softmax", new RnnOutputLayer.Builder(new TenhouActorCriticLoss()).activation(Activation.SOFTMAX).nIn(netConf.getNumHiddenNodes()).nOut(numOutputs).build(), "lstm0");
            confB.setOutputs("value", "softmax");

            confB.inputPreProcessor("conv0", new RnnToCnnPreProcessor(1, originW, 1));
            confB.inputPreProcessor("dense0", new CnnToFeedForwardPreProcessor(1, originW - (kernelW - 1) * convNum, netConf.getNumHiddenNodes()));
            confB.inputPreProcessor("lstm0", new FeedForwardToRnnPreProcessor());
//            confB.inputPreProcessor("value", new RnnToFeedForwardPreProcessor());
//            confB.inputPreProcessor("softmax", new RnnToFeedForwardPrePr
// ocessor());
        }

//    confB.setInputTypes(InputType.convolutionalFlat(1, 74, 1))

//    confB.inputPreProcessor("conv0", new RnnToCnnPreProcessor(1, 74, 1))

//        confB.inputPreProcessor("densepre", new TenhouInputPreProcessor())

        ComputationGraphConfiguration cgconf = confB
//                .pretrain(false).backprop(true)
                .build();
        cgconf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);

        TenhouComputationGraph model = new TenhouComputationGraph(cgconf);
        model.init();

        if (netConf.getListeners() != null) {
            List<TrainingListener> listeners = new ArrayList<TrainingListener>();
            for (TrainingListener listener: netConf.getListeners()) {
                listeners.add(listener);
            }
//            model.setListeners(listeners);
            model.setListeners(netConf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return model;
    }

    public static ComputationGraph CreateComputeGraph(ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) {
        if (netConf.isUseLSTM()) {
            return CreateLstmComputeGraph(netConf, numInputs, numOutputs, false);
        }else {
            return CreateDenseComputeGraph(netConf, numInputs, numOutputs);
        }
    }


    public static ComputationGraph LoadModel(String modelFilePath, ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) throws Exception{
        ComputationGraph newModel = CreateComputeGraph(netConf, numInputs, numOutputs);
        ComputationGraph model = ModelSerializer.restoreComputationGraph(modelFilePath);

        for (String key: newModel.paramTable().keySet()) {
            System.out.println("Key " + key);
        }
        for (String key: model.paramTable().keySet()) {
            System.out.println("Import Key " + key);
        }

        for (String key: newModel.paramTable().keySet()) {
            if (!model.paramTable().containsKey(key)) {
                StaticLogger.error("Does not contains layer " + key);
                throw new Exception("Not the same model, missing " + key);
            }else {
                StaticLogger.info("Found layer " + key);
            }
        }

        model.init();
//        if (netConf.getListeners() != null) {
//            model.setListeners(netConf.getListeners());
//        } else {
//            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
//        }
        //        return new ActorCriticCompGraph(model);

        return model;
    }

    public static ComputationGraph ImportModel(String modelFilePath, ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) throws  Exception {
        ComputationGraph importModel = ModelSerializer.restoreComputationGraph(modelFilePath);
        ComputationGraph newModel = CreateComputeGraph(netConf, numInputs, numOutputs);

        Map<String, INDArray> paramTable = importModel.paramTable();
        Map<String, INDArray> newParamTable = newModel.paramTable();
        newModel.init();

        for (String key: paramTable.keySet()) {
            StaticLogger.info("Load param for " + key);
            if (newParamTable.containsKey(key)) {
                StaticLogger.info("Set params for " + key);
                newModel.setParam(key, paramTable.get(key));
            }
        }

//        newModel.setUpdater(importModel.getUpdater());

//        if (netConf.getListeners() != null) {
//            newModel.setListeners(netConf.getListeners());
//        } else {
//            newModel.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
//        }

//        return new ActorCriticCompGraph(newModel);

        return newModel;
    }

    public static ActorCriticCompGraph CreateActorCritic(ActorCriticFactoryCompGraphStdDense.Configuration netConf, ComputationGraph model) {
//        ComputationGraph model = CreateComputeGraph(netConf, numInputs, numOutputs);

//        model.init();
        if (netConf.getListeners() != null) {
            model.setListeners(netConf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        System.out.println("TenhouActorCriticCompGraph");
        return new TenhouActorCriticCompGraph(model);

    }

    public static ActorCriticCompGraph GetCriticActor(String modelFilePath, ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) {
        ComputationGraph model = null;
        boolean modelReady = false;

        try {
            model = LoadModel(modelFilePath, netConf, numInputs, numOutputs);
            modelReady = true;
            StaticLogger.info("Loaded model from " + modelFilePath);
        }catch(Exception e){
            StaticLogger.error("Failed to load model from " + modelFilePath + " as " + e);
            e.printStackTrace();
        }

        if (!modelReady) {
            try {
                model = ImportModel(modelFilePath, netConf, numInputs, numOutputs);
                modelReady = true;
                StaticLogger.info("Imported model from " + modelFilePath);
            }catch(Exception e) {
                StaticLogger.error("Failed to import model from " + modelFilePath + " as " + e);
                e.printStackTrace();
            }
        }

        if (!modelReady) {
            model = CreateComputeGraph(netConf, numInputs, numOutputs);
            modelReady = true;
            StaticLogger.info("Created new model ");
        }

        return CreateActorCritic(netConf, model);
    }

}
