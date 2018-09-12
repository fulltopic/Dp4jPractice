package rl.dqn.reinforcement.dqn.nn.a3c;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CThreadDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.network.ac.ActorCriticLoss;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import rl.dqn.reinforcement.dqn.config.DqnSettings;
import rl.dqn.reinforcement.dqn.nn.TenhouArray;

import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class A3CTenhouDense extends A3CDiscrete<rl.dqn.reinforcement.dqn.nn.TenhouArray> {
    static Logger StaticLogger = LoggerFactory.getLogger(A3CTenhouDense.class + "static");
    Logger logger = LoggerFactory.getLogger(A3CTenhouDense.class);
    private ACPolicy tenhouPolicy = null;
    private int saveCounter = 0;


    public static ComputationGraph CreateComputeGraph(ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) {
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


        ComputationGraphConfiguration cgconf = confB.pretrain(false).backprop(true).build();




        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();

        return model;
    }

    public static ActorCriticCompGraph CreateActorCritic(ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) {
        ComputationGraph model = CreateComputeGraph(netConf, numInputs, numOutputs);

//        model.init();
        if (netConf.getListeners() != null) {
            model.setListeners(netConf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return new ActorCriticCompGraph(model);

    }

    public static ActorCriticCompGraph LoadActorCritic(String modelFilePath, ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) throws Exception{
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
        if (netConf.getListeners() != null) {
            model.setListeners(netConf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return new ActorCriticCompGraph(model);
    }

    public static ActorCriticCompGraph ImportActorCritic(String modelFilePath, ActorCriticFactoryCompGraphStdDense.Configuration netConf, int numInputs, int numOutputs) throws  Exception {
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

        if (netConf.getListeners() != null) {
            newModel.setListeners(netConf.getListeners());
        } else {
            newModel.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return new ActorCriticCompGraph(newModel);
    }


    public A3CTenhouDense(MDP<TenhouArray, Integer, DiscreteSpace> mdp, A3CConfiguration conf,
                          DataManager dataManager, IActorCritic actorCritic) {
        super(mdp, actorCritic, conf, dataManager);
        this.tenhouPolicy = new ACTenhouPolicy(actorCritic);
    }

    public ACPolicy getPolicy() {
        return tenhouPolicy;
    }

    protected AsyncThread newThread(int i) {
        return new A3CTenhouThreadDiscrete(mdp.newInstance(), getAsyncGlobal(), getConfiguration(), i, getDataManager());
    }

    public void train() {

        try {
            logger.info("AsyncLearning training starting.");
            launchThreads();

            //this is simply for stat purposes
            getDataManager().writeInfo(this);

            synchronized (this) {
                while (!isTrainingComplete() && getAsyncGlobal().isRunning()) {
//                    getPolicy().play(getMdp(), getHistoryProcessor());
//                    getDataManager().writeInfo(this);
                    logger.info("---------------------> Wait 60000");
                    wait(60000);
                    saveCounter ++;
                    if (saveCounter >= DqnSettings.UpdateTillSave()) {
                        saveCounter = 0;
                        String fileName = DqnSettings.ModelFileName() + System.currentTimeMillis();
                        logger.info("Writing model into " + fileName);
                        this.getNeuralNet().save(fileName);
                    }
                }
            }
        } catch (Exception e) {
            logger.error("Training failed " + e);
            e.printStackTrace();
        }
    }

    public void launchThreads() {
        startGlobalThread();
        for (int i = 0; i < getConfiguration().getNumThread(); i++) {
            Thread t = newThread(i);
            Nd4j.getAffinityManager().attachThreadToDevice(t,
                    (i + 1) % Nd4j.getAffinityManager().getNumberOfDevices());
            t.start();

        }
        logger.info("Threads launched");
    }
}
