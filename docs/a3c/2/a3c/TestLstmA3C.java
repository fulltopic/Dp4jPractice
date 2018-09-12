package rl.dqn.reinforcement.dqn.nn.a3c;

import lombok.val;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import rl.dqn.reinforcement.dqn.config.DqnSettings;
import rl.dqn.reinforcement.dqn.mdp.TenhouEncodableMdp;
import rl.dqn.reinforcement.dqn.nn.datapocess.TenhouCompCnnLstmIterator;

import java.util.Map;

public class TestLstmA3C {
    private String fileName = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/supervised/models/conv_conv_dense_lstm_bestsofar.xml";

    private static A3CDiscrete.A3CConfiguration CARTPOLE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //Random seed
                    Integer.MAX_VALUE,            //Max step By epoch
                    Integer.MAX_VALUE,         //Max step
                    DqnSettings.A3CThreadNum(),              //Number of threads
                    Integer.MAX_VALUE,              //t_max
                    1,             //num step noop warmup
                    0.1,           //reward scaling
                    0.99,           //gamma
                    1.0           //td-error clipping
            );

    private ActorCriticFactoryCompGraphStdDense.Configuration getConfig() {
        ActorCriticFactoryCompGraphStdDense.Configuration config = ActorCriticFactoryCompGraphStdDense.Configuration.builder()
                .learningRate(1e-3)
                .l2(0)
                .numHiddenNodes(128)
                .numLayer(2)
                .useLSTM(true)
                .build();

        return config;
    }

    private void checkBestModel() {

        try {
            ComputationGraph model = ModelSerializer.restoreComputationGraph(fileName);
            Map<String, INDArray> paramTable = model.paramTable();

            for (String key: paramTable.keySet()) {
                System.out.println(key);
            }

            String validPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/validation/";
            TenhouCompCnnLstmIterator validIte = new TenhouCompCnnLstmIterator(validPath, 64, true, null);

            validIte.reset();
            Evaluation eval = model.evaluate(validIte);
            //      val eval = new Evaluation()
            //      model.doEvaluation(validIte, eval)


            String stat = eval.stats();
            System.out.println("=========================================> Evaluation");
            System.out.println(stat);

        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public void testLoadModel() {
        try {
//            ComputationGraph model = A3CTenhouModelFactory.ImportModel(fileName, getConfig(), 74, 42);
//
//            String validPath = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/xmlfiles/validation/";
//            TenhouCompCnnLstmIterator validIte = new TenhouCompCnnLstmIterator(validPath, 64, true, null);
//
//            validIte.reset();
//            Evaluation eval = model.evaluate(validIte);
//
//
//            String stat = eval.stats();
//            System.out.println("=========================================> Evaluation");
//            System.out.println(stat);
            ComputationGraph model = ModelSerializer.restoreComputationGraph(DqnSettings.LoadNNModelFileName(), true);
            System.out.println(model.getConfiguration().toJson());
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public void train() throws Exception{
        DqnSettings.printConfig();
        ActorCriticCompGraph graph = A3CTenhouModelFactory.GetCriticActor(DqnSettings.LoadNNModelFileName(), getConfig(), 74, 42);

        MDP mdp = new TenhouEncodableMdp(false, -1);
        DataManager manager = new DataManager();
        A3CTenhouDense criticActor = new A3CTenhouDense(mdp, CARTPOLE_A3C, manager, graph);


        criticActor.train();
    }

    public static void main(String[] args) throws Exception{
        TestLstmA3C tester = new TestLstmA3C();
//        tester.testLoadModel();
//        tester.checkBestModel();
        tester.train();
    }
}