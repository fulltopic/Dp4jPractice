package dp4jpractice.org.mjdrl.nn.dqn.nn;

import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tenhouclient.impl.MessageParseUtilsImpl;
import tenhouclient.impl.mdp.TenhouArray;
import tenhouclient.impl.mdp.TenhouIntegerActionSpace;

import java.util.ArrayList;

public class TenhouSimpleDoubleDenseQLDiscrete extends TenhouSimpleDenseQLDiscrete {
    private Logger logger = LoggerFactory.getLogger(TenhouSimpleDoubleDenseQLDiscrete.class);
    
    
    public TenhouSimpleDoubleDenseQLDiscrete(QLearning.QLConfiguration conf, MDP<TenhouArray, Integer, TenhouIntegerActionSpace> mdp, IDQN dqn, DataManager dataManager) {
        super(conf, mdp, dqn, dataManager);
    }

    protected Pair<INDArray, INDArray> setTarget(ArrayList<Transition<Integer>> transitions) {
        logger.debug("---------------------------------------> setTarget");
        if (transitions.size() == 0) {
            logger.debug("To throw exception");
            throw new IllegalArgumentException("too few transitions");
        }

        int size = transitions.size();

        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape()
                : getHistoryProcessor().getConf().getShape();
        int[] nshape = makeShape(size, shape);
        INDArray obs = Nd4j.create(nshape);
        INDArray nextObs = Nd4j.create(nshape);
        int[] actions = new int[size];
        boolean[] areTerminal = new boolean[size];
        printShape(obs, "obs");
        printShape(nextObs, "nextObs");


        for (int i = 0; i < size; i++) {
            Transition<Integer> trans = transitions.get(i);
            areTerminal[i] = trans.isTerminal();
            actions[i] = trans.getAction();

            INDArray[] obsArray = trans.getObservation();
            obs.putRow(i, DqnUtils.getNNInput(obsArray[0]));

            nextObs.putRow(i, DqnUtils.getNNInput(trans.getNextObservation()));
        }

        INDArray dqnOutputAr = dqnOutput(obs);
        INDArray origDqnOutput = dqnOutputAr.dup();
        INDArray dqnOutputNext = targetDqnOutput(nextObs);

        logger.debug("before dqnOutputAr " + dqnOutputAr);

        logger.debug("obs " + obs);
        logger.debug("nextObs " + nextObs);



        double[] tempQ = new double[size];
        for (int i = 0; i < size; i ++) {
            Pair<Double, Integer> tempPair = MessageParseUtilsImpl.getLegalQAction(transitions.get(i).getNextObservation(), dqnOutputNext.getRow(i));
            tempQ[i] = tempPair.getFirst();
        }

//        printShape(getMaxAction, "getMaxAction");
//        printShape(tempPair.getKey(), "tempQ");

        for (int i = 0; i < size; i++) {
            double yTar = transitions.get(i).getReward();
            logger.debug("reward " + yTar);
            if (!areTerminal[i]) {
                double q = tempQ[i];
                yTar += getConfiguration().getGamma() * q;
                logger.debug("q = " + q + ", yTar = " + yTar);
            }


            if(dqnOutputAr.shape().length > 2) {
                double previousV = dqnOutputAr.getDouble(i, actions[i], 0);

                double lowB = previousV - getConfiguration().getErrorClamp();
                double highB = previousV + getConfiguration().getErrorClamp();
                double clamped = Math.min(highB, Math.max(yTar, lowB));

                dqnOutputAr.putScalar(i, actions[i], 0, clamped);
            }
            else {
                double previousV = dqnOutputAr.getDouble(i, actions[i]);
                double lowB = previousV - getConfiguration().getErrorClamp();
                double highB = previousV + getConfiguration().getErrorClamp();
                double clamped = Math.min(highB, Math.max(yTar, lowB));

                dqnOutputAr.putScalar(i, actions[i], clamped);
            }
        }

//        printShape(dqnOutputAr, "dqnOutputAr");
        logger.debug("after dqnOutputAr " + dqnOutputAr);

        logger.debug("Converge? " + dqnOutputAr.sub(origDqnOutput).normmaxNumber());



        return new Pair<INDArray, INDArray>(obs, dqnOutputAr);
    }

}
