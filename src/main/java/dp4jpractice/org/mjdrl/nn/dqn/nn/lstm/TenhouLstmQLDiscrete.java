package dp4jpractice.org.mjdrl.nn.dqn.nn.lstm;

import dp4jpractice.org.mjdrl.config.DqnSettings;
import dp4jpractice.org.mjdrl.nn.dqn.nn.DqnUtils;
import dp4jpractice.org.mjdrl.nn.dqn.nn.lstm.TenhouLstmEpsGreedy;
import dp4jpractice.org.mjdrl.nn.dqn.nn.TenhouSimpleDenseQLDiscrete;
import dp4jpractice.org.mjdrl.nn.dqn.nn.replay.LstmExpReplay;
import dp4jpractice.org.mjdrl.nn.dqn.nn.replay.LstmTransition;
import dp4jpractice.org.mjdrl.nn.dqn.utils.RewardLogger;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tenhouclient.impl.MessageParseUtilsImpl;
import tenhouclient.impl.mdp.TenhouArray;
import tenhouclient.impl.mdp.TenhouIntegerActionSpace;

import java.util.ArrayList;

public class TenhouLstmQLDiscrete extends TenhouSimpleDenseQLDiscrete {
    private Logger logger = LoggerFactory.getLogger(TenhouLstmQLDiscrete.class);
    
    
    private LstmExpReplay replayBuffer = new LstmExpReplay();
    private TenhouLstmEpsGreedy lstmEps;
    private LstmDqn lstmDqn;
    private LstmDqn targetLstmDqn;
    private INDArray dqnInput;
    private int obsColNum;
    private int actionColNum;
    private INDArray qsRowArray;
    private int epochNum;
    private static final int UpdateDqnLimit = 5;

    public TenhouLstmQLDiscrete(QLearning.QLConfiguration conf, MDP<TenhouArray, Integer, TenhouIntegerActionSpace> mdp, LstmDqn dqn, DataManager dataManager) {
        super(conf, mdp, dqn, dataManager);

        this.lstmDqn = dqn;
        this.targetLstmDqn = new LstmDqn(dqn.getNN().clone()); //!!!!!!!!TODO: update target will fail
        this.egPolicy = new TenhouLstmEpsGreedy(dqn, conf.getUpdateStart(), epsilonNbStep, conf.getMinEpsilon(), this);
        this.lstmEps = new TenhouLstmEpsGreedy(dqn, conf.getUpdateStart(), epsilonNbStep, conf.getMinEpsilon(), this);

        obsColNum = mdp.getObservationSpace().getShape()[0];
        dqnInput = Nd4j.create(new int[] {1, obsColNum, 1}, 'f');
        actionColNum = mdp.getActionSpace().getSize();
        qsRowArray = Nd4j.create(1, actionColNum);
    }


    private void updateDqnInput(INDArray input) {
        for (int i = 0; i < obsColNum; i ++) {
            dqnInput.putScalar(0, i, 0, input.getDouble(i));
        }
    }

    private void updateRowQs(INDArray qs) {
        for (int i = 0; i < actionColNum; i ++) {
            qsRowArray.putScalar(0, i, qs.getDouble(0, i, 0));
        }
    }

    public QLStepReturn<TenhouArray>  trainStep(TenhouArray obs){
        int action = 0;
        INDArray rawInput = getInput(obs);
        INDArray input = DqnUtils.getNNInput(rawInput);

        Double maxQ = Double.NaN;

        if(historyInput == null) {
            historyInput = input;
            rawHistoryInput = rawInput;
        }

        logger.debug("input: " + historyInput);
        logger.debug("rawInput: " + rawHistoryInput);


        updateDqnInput(historyInput);
        INDArray qs = lstmDqn.getNN().rnnTimeStep(dqnInput);
//        INDArray qs = getCurrentDQN().output(historyInput);
        int maxAction = qs.argMax(1).getInt(0, 0);
//        int maxAction = Learning.getMaxAction(qs);
        maxQ = qs.getDouble(0, maxAction, 0);
        updateRowQs(qs);
        action = lstmEps.nextAction(rawHistoryInput, qsRowArray);
        logger.debug("qs " + qs);
        logger.debug("maxAction " + maxAction);
        logger.debug("maxQ " + maxQ);
        logger.debug("action " + action);

        lastAction = action;
        StepReply<TenhouArray> stepReply = getMdp().step(action);
        accuReward += stepReply.getReward() * conf.getRewardFactor();
        logger.debug("The reward " + accuReward);

        RewardLogger.writeLog("maxQ: " + maxQ);
        RewardLogger.writeLog("reward: " + accuReward);

        INDArray rawNInput = getInput(stepReply.getObservation());
        INDArray nInput = DqnUtils.getNNInput(rawNInput).sub(DqnUtils.getNNInput(rawHistoryInput));

        //qs not been modified afterwards
        LstmTransition trans = new LstmTransition(rawInput, rawNInput, qs, action, accuReward, stepReply.isDone());
        replayBuffer.store(trans);

        if (stepReply.isDone()){
            lstmDqn.getNN().rnnClearPreviousState();

            updateDqn();
        }

        historyInput = nInput;
        rawHistoryInput = rawNInput;
        accuReward = 0;

        return new QLStepReturn<>(maxQ, getCurrentDQN().getLatestScore(), stepReply);
    }

    protected void printShape(INDArray data, String comment) {
        logger.debug(comment);
        int[] shape = data.shape();
        logger.debug("------------------>" + shape.length);
        for (int i = 0; i < shape.length; i ++) {
            logger.debug(shape[i] + ",");
        }
    }

    private void updateObsArray(INDArray src, INDArray dst, int exampleNum) {
        for (int i = 0; i < obsColNum; i ++) {
            dst.putScalar(0, i, exampleNum, src.getDouble(i));
        }
    }

    protected void updateDqn() {

        logger.debug("---------------------------------------> setTarget");
        ArrayList<LstmTransition> transitions = replayBuffer.getBatch();
        if (transitions.size() == 0) {
            throw new IllegalArgumentException("too few transitions");
        }
        int size = transitions.size();


        INDArray obs = Nd4j.create(new int[]{1, obsColNum, size}, 'f');
        INDArray nextObs = Nd4j.create(new int[] {1, obsColNum, size}, 'f');
        int[] actions = new int[size];

        INDArray dqnOutputAr = Nd4j.create(new int[]{1, actionColNum, size}, 'f');
//        boolean[] areTerminal = new boolean[size];
//
//        INDArray dqnOutputAr = Nd4j.create(size, getMdp().getActionSpace().getSize());

        for (int i = 0; i < size; i++) {
            LstmTransition trans = transitions.get(i);
//            areTerminal[i] = trans.isTerminal();
            actions[i] = trans.action();
            logger.debug("qs " + trans.qs());
//            printShape(trans.qs(), "trans.qs");
            for (int k = 0; k < actionColNum; k ++) {
                dqnOutputAr.putScalar(0, k, i, trans.qs().getDouble(0, k, 0));
            }

            if (i <= 0) {
                INDArray obsInput = DqnUtils.getNNFInput(trans.rawInput());
                INDArray nextObsInput = DqnUtils.getNNFInput(trans.rawNextInput());
                updateObsArray(obsInput, obs, i);
                updateObsArray(nextObsInput, nextObs, i);

//                obs.putRow(i, DqnUtils.getNNInput(obsArray));

//                nextObs.putRow(i, DqnUtils.getNNInput(trans.rawNextInput()));
            }else {
                LstmTransition lastTrans = transitions.get(i - 1);
                INDArray obsArray = DqnUtils.getNNInput(trans.rawInput());
                INDArray lastObsArray = DqnUtils.getNNInput(lastTrans.rawInput());
                INDArray obsInput = obsArray.sub(lastObsArray);
                INDArray nextArray = DqnUtils.getNNInput(trans.rawNextInput());
                INDArray nextObsInput = nextArray.sub(obsArray);

                //TODO: Don't know if the calculate nextobj this way is OK
                updateObsArray(obsInput, obs, i);
                updateObsArray(nextObsInput, nextObs, i);
            }
        }

        targetLstmDqn.getNN().rnnClearPreviousState();
        INDArray dqnOutputNext = targetLstmDqn.output(nextObs);

        logger.debug("before dqnOutputAr " + dqnOutputAr);

        logger.debug("obs " + obs);
        logger.debug("nextObs " + nextObs);



        double[] tempQ = new double[size];
        INDArray tmpQs = Nd4j.create(1, obsColNum);
        for (int i = 0; i < size; i ++) {
            for (int k = 0; k < actionColNum; k ++) {
                tmpQs.putScalar(0, k, dqnOutputNext.getDouble(0, k, i));
            }
            org.nd4j.linalg.primitives.Pair<Double, Integer> tempPair = MessageParseUtilsImpl.getLegalQAction(transitions.get(i).rawNextInput(), tmpQs);
            tempQ[i] = tempPair.getFirst();
        }

//        printShape(getMaxAction, "getMaxAction");
//        printShape(tempPair.getKey(), "tempQ");

        for (int i = 0; i < size; i++) {
            double yTar = transitions.get(i).reward();
            logger.debug("reward " + yTar);
            if (!transitions.get(i).isTerminal()) {
                double q = tempQ[i];
                yTar += getConfiguration().getGamma() * q;
                logger.debug("q = " + q + ", yTar = " + yTar);
            }


            if(dqnOutputAr.shape().length > 2) {
                double previousV = dqnOutputAr.getDouble(0, actions[i], i);

                double lowB = previousV - getConfiguration().getErrorClamp();
                double highB = previousV + getConfiguration().getErrorClamp();
                double clamped = Math.min(highB, Math.max(yTar, lowB));

                dqnOutputAr.putScalar(0, actions[i], i, clamped);
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

        lstmDqn.getNN().rnnClearPreviousState();
        lstmDqn.fit(obs, dqnOutputAr);

        epochNum ++;
        if ((epochNum % UpdateDqnLimit) == 0) {
            targetLstmDqn = new LstmDqn(lstmDqn.getNN().clone());
        }

        if ((epochNum % DqnSettings.UpdateTillSave()) == 0) {
            String currFileName = DqnSettings.ModelLstmFileName() + System.currentTimeMillis() + ".xml";
            try {
//                    getCurrentDQN()
//                    currentDQN.
                logger.debug("Save model into "+ currFileName);
                ModelSerializer.writeModel(lstmDqn.getNN(), currFileName, true);
            }catch (Exception e) {
                e.printStackTrace();
            }
        }

        replayBuffer.clear();
        lstmDqn.getNN().rnnClearPreviousState();
        targetLstmDqn.getNN().rnnClearPreviousState();

//        return new Pair<>(obs, dqnOutputAr);
    }
}
