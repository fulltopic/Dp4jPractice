package rl.dqn.reinforcement.dqn.nn.a3c;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CThreadDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import rl.dqn.reinforcement.dqn.nn.TenhouArray;

import java.util.Random;
import java.util.Stack;

import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

public class A3CTenhouThreadDiscrete extends A3CThreadDiscrete<TenhouArray> {
    Logger logger = LoggerFactory.getLogger(A3CTenhouThreadDiscrete.class);

    public A3CTenhouThreadDiscrete(MDP<TenhouArray, Integer, DiscreteSpace> mdp, AsyncGlobal<IActorCritic> asyncGlobal,
                                   A3CDiscrete.A3CConfiguration a3cc, int threadNumber, DataManager dataManager) {
        super(mdp, asyncGlobal, a3cc, threadNumber, dataManager);
    }

    protected Policy<TenhouArray, Integer> getPolicy(IActorCritic net) {
        return new ACTenhouPolicy(net, new Random(conf.getSeed()));
    }


    public SubEpochReturn<TenhouArray> trainSubEpoch(TenhouArray sObs, int nstep) {
        logger.debug("-------------------------------> isRecurrent " + getCurrent().isRecurrent());
        synchronized (getAsyncGlobal()) {
           getCurrent().copy(getAsyncGlobal().getCurrent());
        }
        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        TenhouArray obs = sObs;
        Policy<TenhouArray, Integer> policy = getPolicy(getCurrent());
        ACTenhouPolicy acPolicy = (ACTenhouPolicy)policy;

        Integer action;
        Integer lastAction = null;
        IHistoryProcessor hp = getHistoryProcessor();
        int skipFrame = hp != null ? hp.getConf().getSkipFrame() : 1;

        double reward = 0;
        double accuReward = 0;
        int i = 0;
        while (!getMdp().isDone() && i < nstep * skipFrame) {

            INDArray input = Learning.getInput(getMdp(), obs);
            INDArray hstack = null;

            if (hp != null) {
                hp.record(input);
            }

//            if (i % skipFrame != 0 && lastAction != null) {
//                action = lastAction;
//            } else {
//                hstack = processHistory(input);
//                action = policy.nextAction(hstack);
//            }
            hstack = processHistory(input);
            INDArray[] output = getCurrent().outputAll(hstack);
            action = acPolicy.nextAction(hstack, output[1]);

            StepReply<TenhouArray> stepReply = getMdp().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();
            obs = stepReply.getObservation();


            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + output);
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + accuReward);
            rewards.add(new MiniTrans<>(hstack, action, output, accuReward));
            accuReward = 0;

            //if it's not a skipped frame, you can do a step of training
//            if (i % skipFrame == 0 || lastAction == null || stepReply.isDone()) {
//                obs = stepReply.getObservation();
//
//                if (hstack == null) {
//                    hstack = processHistory(input);
//                }
//                INDArray[] output = getCurrent().outputAll(hstack);
//                rewards.add(new MiniTrans(hstack, action, output, accuReward));
//
//                accuReward = 0;
//            }

            reward += stepReply.getReward();

            i++;
            lastAction = action;
        }

        //a bit of a trick usable because of how the stack is treated to init R
        INDArray input = Learning.getInput(getMdp(), obs);
        INDArray hstack = processHistory(input);

//        if (hp != null) {
//            hp.record(input);
//        }
//
//        if (getMdp().isDone() && i < nstep * skipFrame) {
//            logger.info("Final step null");
//            rewards.add(new MiniTrans(hstack, null, null, 0));
//        } else {
//            INDArray[] output = null;
//            if (getConf().getTargetDqnUpdateFreq() == -1) {
//                logger.info("Output the last stack");
//                output = getCurrent().outputAll(hstack);
//            }
//            else {
//                logger.info("Output the target for last stack");
//                synchronized (getAsyncGlobal()) {
//                    output = getAsyncGlobal().getTarget().outputAll(hstack);
//                }
//            }
//            double maxQ = Nd4j.max(output[0]).getDouble(0);
//            rewards.add(new MiniTrans(hstack, null, output, maxQ));
//        }

        if (rewards.size() > 1) { //only difference
            getAsyncGlobal().enqueue(calcGradient(getCurrent(), rewards), i);
        }else {
            logger.error("Invalid history size, not to compute");
        }

        return new SubEpochReturn<>(i, obs, reward, getCurrent().getLatestScore());
    }

    @Override
    public Gradient[] calcGradient(IActorCritic iac, Stack<MiniTrans<Integer>> rewards) {
        MiniTrans<Integer> minTrans = rewards.pop();

        int size = rewards.size();

        //if recurrent then train as a time serie with a batch size of 1
        boolean recurrent = getAsyncGlobal().getCurrent().isRecurrent();

        int[] shape = getHistoryProcessor() == null ? mdp.getObservationSpace().getShape()
                : getHistoryProcessor().getConf().getShape();
        int[] nshape = recurrent ? Learning.makeShape(1, shape, size)
                : Learning.makeShape(size, shape);

        INDArray input = Nd4j.create(nshape);
        INDArray targets = recurrent ? Nd4j.create(1, 1, size) : Nd4j.create(size, 1);
        INDArray logSoftmax = recurrent ? Nd4j.zeros(1, mdp.getActionSpace().getSize(), size)
                : Nd4j.zeros(size, mdp.getActionSpace().getSize());

        double r = minTrans.getReward();
        for (int i = size - 1; i >= 0; i--) {
            minTrans = rewards.pop();

            r = minTrans.getReward() + conf.getGamma() * r;
            if (recurrent) {
                input.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)).assign(minTrans.getObs());
            } else {
                input.putRow(i, minTrans.getObs());
            }

            //the critic
            targets.putScalar(i, r);

            //the actor
            double expectedV = minTrans.getOutput()[0].getDouble(0);
            double advantage = r - expectedV;

            if (recurrent) {
                logSoftmax.putScalar(0, minTrans.getAction(), i, advantage);
            } else {
                logSoftmax.putScalar(i, minTrans.getAction(), advantage);
            }
        }

        logger.debug("calcGradient input " + input);
        logger.debug("calcGradient logsoftmax " + logSoftmax);
        logger.debug("calcGradient value " + targets);

        Gradient[] rc = iac.gradient(input, new INDArray[] {targets, logSoftmax});
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RC");
        for (int i = 0; i < rc.length; i ++) {
            logger.debug("" + rc[i]);
        }
        return rc;
    }
}
