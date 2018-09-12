package rl.dqn.reinforcement.dqn.nn.a3c;

import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.util.DataManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import rl.dqn.reinforcement.dqn.config.DqnSettings;
import rl.dqn.reinforcement.dqn.mdp.TenhouEncodableMdp;

import java.io.IOException;

public class TestDenseA3C {
    private static Logger logger = LoggerFactory.getLogger(TestDenseA3C.class);

    private static A3CDiscrete.A3CConfiguration CARTPOLE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //Random seed
                    Integer.MAX_VALUE,            //Max step By epoch
                    Integer.MAX_VALUE,         //Max step
                    DqnSettings.A3CThreadNum(),              //Number of threads
                    Integer.MAX_VALUE,              //t_max
                    1,             //num step noop warmup
                    0.01,           //reward scaling
                    0.99,           //gamma
                    1.0           //td-error clipping
            );




    public static ActorCriticFactoryCompGraphStdDense.Configuration GetConfig() {
        ActorCriticFactoryCompGraphStdDense.Configuration config = ActorCriticFactoryCompGraphStdDense.Configuration
                .builder().learningRate(1e-2).l2(0).numHiddenNodes(128).numLayer(2).useLSTM(false)
                .build();

        return config;
    }

    public static void main(String[] args) throws IOException {

        DqnSettings.printConfig();
//        ActorCriticCompGraph graph = null;
//
//        boolean loaded = false;
//
//        try {
//            logger.info("Try load existing model");
//            graph = A3CTenhouDense.LoadActorCritic(DqnSettings.LoadNNModelFileName(), GetConfig(), 74, 42);
//            loaded = true;
//        }catch (Exception e) {
//            e.printStackTrace();
//            logger.error("Failed to load model file " + DqnSettings.LoadNNModelFileName() + ": " + e);
//        }
//
//        if (!loaded) {
//            try {
//                logger.info("Try load paramTable from existing model");
//                graph = A3CTenhouDense.ImportActorCritic(DqnSettings.LoadNNModelFileName(), GetConfig(),74, 42);
//                loaded = true;
//            }catch (Exception e) {
//                e.printStackTrace();
//                logger.error("Failed to import params from existing model " + DqnSettings.LoadNNModelFileName() + ": " + e);
//            }
//        }
//
//        if (!loaded) {
//            System.out.println("Create new model");
//            graph = A3CTenhouDense.CreateActorCritic(GetConfig(), 74, 42);
//            logger.info("Created a new model");
//        }

        ActorCriticCompGraph graph = A3CTenhouModelFactory.GetCriticActor(DqnSettings.LoadNNModelFileName(), GetConfig(), 74, 42);

        MDP mdp = new TenhouEncodableMdp(false, -1);
        DataManager manager = new DataManager();
        A3CTenhouDense criticActor = new A3CTenhouDense(mdp, CARTPOLE_A3C, manager, graph);


        criticActor.train();
    }
}
