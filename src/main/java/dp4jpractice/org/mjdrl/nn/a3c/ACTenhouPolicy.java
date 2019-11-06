package dp4jpractice.org.mjdrl.nn.a3c;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tenhouclient.impl.MessageParseUtilsImpl;
import dp4jpractice.org.mjdrl.config.DqnSettings;
import tenhouclient.impl.mdp.TenhouArray;


import java.util.Random;

public class ACTenhouPolicy extends ACPolicy<TenhouArray> {
    private Logger logger = LoggerFactory.getLogger(ACTenhouPolicy.class);
    private Random tenhouRd = new Random(177 + (System.currentTimeMillis() % 10));

    public ACTenhouPolicy(IActorCritic IActorCritic) {
        this(IActorCritic, null);
    }

    public ACTenhouPolicy(IActorCritic IActorCritic, Random rd) {
        super(IActorCritic, rd);
    }

    public Integer nextAction(INDArray input) {
        INDArray output = this.getNeuralNet().outputAll(input)[1];
        logger.debug("policy " + output);
        Pair<Double, Integer> pair = MessageParseUtilsImpl.getLegalQAction(input, output);
        logger.info(pair.getFirst() + ", " + pair.getSecond());
        return pair.getSecond();
    }

    public Integer nextAction(INDArray input, INDArray output) {
        float epsilon = tenhouRd.nextFloat();
        logger.debug("policy " + output);
        logger.debug("nextAction " + input);
        logger.debug("epsilon " + epsilon);


        if (epsilon > DqnSettings.MinEpsilon() || DqnSettings.IsTest()) {
            Pair<Double, Integer> pair = MessageParseUtilsImpl.getLegalQAction(input, output);
            logger.debug(pair.getFirst() + ", " + pair.getSecond());
            return pair.getSecond();
        }else {
            logger.debug("Random");
            Pair<Double, Integer> pair = MessageParseUtilsImpl.getRandomAction(input, output);
            logger.debug(pair.getFirst() + ", " + pair.getSecond());
            return pair.getSecond();
        }
    }


    //deprecated
    public double play(MDP mdp, IHistoryProcessor hp) {
        return 0;
    }
}
