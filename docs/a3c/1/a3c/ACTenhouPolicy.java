package rl.dqn.reinforcement.dqn.nn.a3c;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import rl.dqn.reinforcement.dqn.client.MessageParseUtils;
import rl.dqn.reinforcement.dqn.config.DqnSettings;
import rl.dqn.reinforcement.dqn.nn.TenhouArray;

import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.collection.immutable.List;

public class ACTenhouPolicy extends ACPolicy<TenhouArray> {
    Logger logger = LoggerFactory.getLogger(ACTenhouPolicy.class);
    Random rd = new Random(177);

    public ACTenhouPolicy(IActorCritic IActorCritic) {
        this(IActorCritic, null);
    }

    public ACTenhouPolicy(IActorCritic IActorCritic, Random rd) {
        super(IActorCritic, rd);
    }

    public Integer nextAction(INDArray input) {
        INDArray output = this.getNeuralNet().outputAll(input)[1];
        logger.info("policy " + output);
        Pair<Double, Integer> pair = MessageParseUtils.getLegalQAction(input, output);
        logger.info(pair.getFirst() + ", " + pair.getSecond());
        return pair.getSecond();
    }

    public Integer nextAction(INDArray input, INDArray output) {
//        INDArray output = this.getNeuralNet().outputAll(input)[1];
        logger.info("policy " + output);
        logger.info("nextAction " + input);

            Pair<Double, Integer> pair = MessageParseUtils.getLegalQAction(input, output);
            logger.info(pair.getFirst() + ", " + pair.getSecond());
            return pair.getSecond();

    }


    public double play(MDP mdp, IHistoryProcessor hp) {
        return 0;
    }
}
