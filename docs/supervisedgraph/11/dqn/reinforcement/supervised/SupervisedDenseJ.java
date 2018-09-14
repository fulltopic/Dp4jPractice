package rl.dqn.reinforcement.supervised;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.network.ac.ActorCriticLoss;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

public class SupervisedDenseJ {
    public ComputationGraph CreateActorCritic(ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs, boolean supervised) {

        ComputationGraphConfiguration.GraphBuilder confB =
                new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED).iterations(1)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .learningRate(netConf.getLearningRate())
                        //.updater(Updater.NESTEROVS).momentum(0.9)
//                        .updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
                        .updater(netConf.getUpdater() != null ? netConf.getUpdater() : new Adam())
                        .weightInit(WeightInit.XAVIER)
                        .regularization(netConf.getL2() > 0)
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

        if (supervised) {
            confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                    .nOut(numOutputs).build(), (netConf.getNumLayer() - 1) + "");
            confB.setOutputs("softmax");
        } else {
            confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                    .nOut(1).build(), (netConf.getNumLayer() - 1) + "");

            confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                    .nOut(numOutputs).build(), (netConf.getNumLayer() - 1) + "");

            confB.setOutputs("value", "softmax");
        }


//        confB.inputPreProcessor("0", new TenhouInputPreProcessor());

        ComputationGraphConfiguration cgconf = confB.pretrain(false).backprop(true).build();


        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();
        if (netConf.getListeners() != null) {
            model.setListeners(netConf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return model;
    }

    public ActorCriticFactoryCompGraphStdDense.Configuration getConfig() {
        ActorCriticFactoryCompGraphStdDense.Configuration config = ActorCriticFactoryCompGraphStdDense.Configuration
                .builder().learningRate(1e-2).l2(0).numHiddenNodes(128).numLayer(2).useLSTM(false)
                .build();

        return config;
    }

    public void printParamArch(ComputationGraph model) {
        Map<String, INDArray> params = model.paramTable();
        for (String key: params.keySet()) {
            System.out.println(key);
        }
    }

    public static void main(String[] args) {
        SupervisedDenseJ tester = new SupervisedDenseJ();

        ComputationGraph graph = tester.CreateActorCritic(tester.getConfig(), 74, 42, true);
        tester.printParamArch(graph);

        ComputationGraph newGraph = tester.CreateActorCritic(tester.getConfig(), 74, 42, false);
        tester.printParamArch(newGraph);

        Map<String, INDArray> params = graph.paramTable();
        Map<String, INDArray> newParams = newGraph.paramTable();

        for (String key: params.keySet()) {
            if (newParams.containsKey(key)) {
                System.out.println("Set params into " + key);
                newGraph.setParam(key, params.get(key));
            }
        }
    }

}
