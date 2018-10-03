package dp4jpractice.org.mjdrl.nn.a3c;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dp4jpractice.org.mjdrl.config.DqnSettings;
import tenhouclient.impl.MessageParseUtilsImpl;
import tenhouclient.impl.mdp.TenhouArray;

import java.util.Random;
import java.util.Set;
import java.util.Stack;

public class A3CTenhouThreadDiscrete extends A3CThreadDiscrete<TenhouArray> {
    Logger logger = LoggerFactory.getLogger(A3CTenhouThreadDiscrete.class);

    public A3CTenhouThreadDiscrete(MDP<TenhouArray, Integer, DiscreteSpace> mdp, AsyncGlobal<IActorCritic> asyncGlobal,
                                   A3CDiscrete.A3CConfiguration a3cc, int threadNumber, DataManager dataManager) {
        super(mdp, asyncGlobal, a3cc, threadNumber, dataManager);
    }

    protected Policy<TenhouArray, Integer> getPolicy(IActorCritic net) {
        return new ACTenhouPolicy(net, new Random(conf.getSeed()));
    }


    protected SubEpochReturn<TenhouArray> trainRnnSubEpoch(TenhouArray sObs, int nstep) {
        synchronized (getAsyncGlobal()) {
            getCurrent().copy(getAsyncGlobal().getCurrent());
        }

        //Ensure states cleared
        getCurrent().reset();

        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        TenhouArray obs = sObs;
        Policy<TenhouArray, Integer> policy = getPolicy(getCurrent());
        ACTenhouPolicy acPolicy = (ACTenhouPolicy)policy;

        Integer action;
        IHistoryProcessor hp = getHistoryProcessor();
        int skipFrame = hp != null ? hp.getConf().getSkipFrame() : 1;

        double accuReward = 0;
        int i = 0;
        while (!getMdp().isDone() && i < nstep * skipFrame) {

            INDArray input = Learning.getInput(getMdp(), obs);
            INDArray hstack;

            if (hp != null) {
                hp.record(input);
            }

            hstack = processHistory(input);

            INDArray nnInput = Nd4j.create(new int[]{1, 74, 1}, 'f');
            for (int t = 0; t < 74; t ++) {
                double v = input.getDouble(0, t);
                double newV = (double)((int)v & 7) / 4.0 + (v - (int)v) / 4.0;
                nnInput.putScalar(0, t, 0, newV);
            }
            logger.debug("trainRnnSubEpoch " + nnInput);
            INDArray[] output = getCurrent().outputAll(nnInput);

//            INDArray[] output = getCurrent().outputAll(hstack);
            action = acPolicy.nextAction(hstack, output[1]);
            int qAction = Learning.getMaxAction(output[1]);

            StepReply<TenhouArray> stepReply = getMdp().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();
            obs = stepReply.getObservation();

            double qReward = 0;
            if (getMdp().isDone()) {
                if (accuReward > 0) {
                    qReward = accuReward * DqnSettings.A3CWinReward();
                    qReward = Math.min(1, Math.max(qReward, -1));
                } else {
                    qReward = 0;
                }

//                else if (accuReward > 0 && accuReward <= 0.001) {
//                    accuReward = DqnSettings.A3CTiePenalty();
//                }
//                else {
//                    qReward = accuReward;
//                }
            }

//            else {
//                if (action == qAction) {
//                    qReward = DqnSettings.ActionMatchReward();
//                }else {
//                    qReward = DqnSettings.ActionMisMatchReward();
//                }
//            }

            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + output[0]);
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + qReward);
            rewards.add(new MiniTrans<>(hstack, action, output, qReward));

//            reward += stepReply.getReward();

            i++;
        }


        //For Test
        if (! DqnSettings.IsTest()) {
            if (rewards.size() > 1 && accuReward > 0) {
                Gradient[] rc = calcGradient(getCurrent(), rewards);
                logger.debug("+++++++++++++++++++++++ " + rc[0].getGradientFor("lstm0_W").getDouble(0, 0));

                getAsyncGlobal().enqueue(rc, i);
            }
        }
        //END Test


//        for (int j = 0; j < gradients.length; j ++) {
//            logger.debug("Gradient " + gradients[j]);
//        }

        return new SubEpochReturn<>(i, obs, accuReward, getCurrent().getLatestScore());
    }

    protected SubEpochReturn<TenhouArray> trainDenseSubEpoch(TenhouArray sObs, int nstep) {
//        logger.debug("-------------------------------> isRecurrent " + getCurrent().isRecurrent());
        synchronized (getAsyncGlobal()) {
            getCurrent().copy(getAsyncGlobal().getCurrent());
        }
        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        TenhouArray obs = sObs;
        Policy<TenhouArray, Integer> policy = getPolicy(getCurrent());
        ACTenhouPolicy acPolicy = (ACTenhouPolicy)policy;

        Integer action;
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

            hstack = processHistory(input);
            INDArray[] output = getCurrent().outputAll(hstack);
            action = acPolicy.nextAction(hstack, output[1]);

            StepReply<TenhouArray> stepReply = getMdp().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();
            obs = stepReply.getObservation();


            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output" + output);
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> accuReward" + accuReward);
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
        }

        //a bit of a trick usable because of how the stack is treated to init R
        INDArray input = Learning.getInput(getMdp(), obs);

        if (rewards.size() > 1) { //only difference
            getAsyncGlobal().enqueue(calcGradient(getCurrent(), rewards), i);
        }else {
            logger.error("Invalid history size, not to compute");
        }

        return new SubEpochReturn<>(i, obs, reward, getCurrent().getLatestScore());
    }

    public SubEpochReturn<TenhouArray> trainSubEpoch(TenhouArray sObs, int nstep) {
        logger.debug("-------------------------------> isRecurrent " + getCurrent().isRecurrent());

        if (getCurrent().isRecurrent()) {
            return trainRnnSubEpoch(sObs, nstep);
        }else {
            return trainDenseSubEpoch(sObs, nstep);
        }
    }

    @Override
    public Gradient[] calcGradient(IActorCritic iac, Stack<MiniTrans<Integer>> rewards) {
        if (getCurrent().isRecurrent()) {
            MiniTrans<Integer> minTrans = rewards.pop();

            int size = rewards.size();

            int[] shape = getHistoryProcessor() == null ? mdp.getObservationSpace().getShape()
                    : getHistoryProcessor().getConf().getShape();
            int[] nshape = Learning.makeShape(1, shape, size);

            INDArray input = Nd4j.create(nshape, 'c');
            INDArray targets = Nd4j.create(new int[]{1, 1, size}, 'c');
            INDArray logSoftmax = Nd4j.zeros(new int[] {1, mdp.getActionSpace().getSize(), size}, 'c');

            double r = minTrans.getReward();
//            double baseR = r;
//            logger.debug("_-------------------------------------------> baseR " + baseR);
            for (int i = size - 1; i >= 0; i--) {
                minTrans = rewards.pop();

                r = minTrans.getReward() + conf.getGamma() * r;
                input.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)).assign(minTrans.getObs());

//                INDArray rawInput = minTrans.getObs().dup();
//
//
//                Set<Integer> validAction = MessageParseUtils.getAvailableActionJavaSet(rawInput);
//                for (int k = 0; k < 42; k ++) {
//                    if (k == minTrans.getAction()) {
//                        double expectedV = minTrans.getOutput()[0].getDouble(0);
//                        double advantage = r - expectedV;
//                        logSoftmax.putScalar(0, k, i, advantage);
//                    }else if (!validAction.contains(k)) {
//                        logSoftmax.putScalar(0, k, i, DqnSettings.InvalidActionPenalty());
//                    }
//                }

                //the actor
                double expectedV = minTrans.getOutput()[0].getDouble(0);
                double advantage = r - expectedV;

                //the critic
                targets.putScalar(i, r);
                logSoftmax.putScalar(0, minTrans.getAction(), i, advantage);
            }

            INDArray rawInputs = input.dup();
            input.fmodi(7.0).divi(4.0);



            iac.reset();

//            INDArray[] tmpOuts = ((TenhouActorCriticCompGraph)this.getAsyncGlobal().getCurrent()).myOutput(input);
            INDArray[] tmpOuts = ((TenhouActorCriticCompGraph)iac).myOutput(input);



            //Penalty for invalid actions
            for (int i = size - 1; i >= 0; i--) {
                INDArray rawInput = rawInputs.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i));
                Set<Integer> validActions = MessageParseUtilsImpl.getAvailableActionJavaSet(rawInput);
                for (int k = 0; k < 42; k ++) {
                    if (!validActions.contains(k)) {
                        double adv = tmpOuts[1].getDouble(0, k, i) * DqnSettings.InvalidActionPenalty();
                        logSoftmax.putScalar(0, k, i, adv);
                    }
                }
            }


            logger.debug("calcGradient mse " + logSoftmax.norm2Number());
            logger.debug("calcGradient input " + input);
            logger.debug("calcGradient logsoftmax " + logSoftmax);
            logger.debug("calcGradient value " + targets);
            logger.debug("Tmp Outs " + tmpOuts[0]);
            logger.debug("Tmp Outs 1 " + tmpOuts[1]);

            Gradient[] rc = iac.gradient(input, new INDArray[]{targets, logSoftmax});
//            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RC");
//            for (int i = 0; i < rc.length; i++) {
//                logger.debug("" + rc[i]);
//            }
            return rc;
        }else {
            return super.calcGradient(iac, rewards);
        }
    }
}
