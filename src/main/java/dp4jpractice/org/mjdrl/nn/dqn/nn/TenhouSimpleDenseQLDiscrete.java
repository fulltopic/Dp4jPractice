package dp4jpractice.org.mjdrl.nn.dqn.nn;

import dp4jpractice.org.mjdrl.config.DqnSettings;
import dp4jpractice.org.mjdrl.nn.dqn.utils.RewardLogger;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.util.Constants;
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


public class TenhouSimpleDenseQLDiscrete extends QLearning<TenhouArray, Integer, TenhouIntegerActionSpace> {
    private Logger logger = LoggerFactory.getLogger(TenhouSimpleDenseQLDiscrete.class);
    
    protected QLConfiguration conf;
    private DataManager dataManager;
    protected MDP<TenhouArray, Integer, TenhouIntegerActionSpace> mdp;
    private DQNPolicy<TenhouArray> policy;

    protected int epsilonNbStep;
    protected EpsGreedy<TenhouArray, Integer, TenhouIntegerActionSpace> egPolicy;
    private IDQN currentDQN;
    private IDQN targetDQN;
    protected int lastAction = 0;
    protected INDArray rawHistory[] = null;

    protected INDArray historyInput = null;
    protected INDArray rawHistoryInput = null;

    protected double accuReward = 0;
    private int lastMonitor = Constants.MONITOR_FREQ * (-1);

//    private String modelBaseName = "/home/zf/workspaces/workspace_java/tenhoulogs/testdqnmodels/model";
    protected int updatedCounter = 0;
    protected static int UpdateTillSave = DqnSettings.UpdateTillSave();

    public TenhouSimpleDenseQLDiscrete(QLearning.QLConfiguration conf, MDP<TenhouArray, Integer, TenhouIntegerActionSpace> mdp, IDQN dqn, DataManager dataManager) {
        super(conf);
        this.conf = conf;
        this.dataManager = dataManager;
        this.mdp = mdp;
        this.policy = null;

        this.epsilonNbStep = conf.getEpsilonNbStep();
        this.currentDQN = dqn;
        this.targetDQN = dqn.clone();
        mdp.getActionSpace().setSeed(conf.getSeed());

        this.egPolicy = new TenhouEpsGreedy(dqn, conf.getUpdateStart(), epsilonNbStep, conf.getMinEpsilon(), this);
    }

//    public void setModelFileName(String fileName) {
//        this.modelBaseName = fileName;
//    }

    public MDP<TenhouArray, Integer, TenhouIntegerActionSpace> getMdp(){
      return mdp;
    }

    public IDQN getTargetDQN() {
        return targetDQN;
    }

    public IDQN getCurrentDQN() {
        return currentDQN;
    }

    public Policy<TenhouArray, Integer> getPolicy() {
        return null;
    }

    public DataManager getDataManager() {
        return dataManager;
    }

    public EpsGreedy<TenhouArray, Integer, TenhouIntegerActionSpace> getEgPolicy() {
        return egPolicy;
    }

    public QLConfiguration getConfiguration() {
        return conf;
    }

    public void setTargetDQN(IDQN dqn) {
        targetDQN = dqn;
    }

    public void postEpoch() {

        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    public void preEpoch() {
        historyInput = null;
        rawHistoryInput = null;
        rawHistory = null;
        lastAction = 0;
        accuReward = 0;

        if (getStepCounter() - lastMonitor >= Constants.MONITOR_FREQ && getHistoryProcessor() != null
                && getDataManager().isSaveData()) {
            lastMonitor = getStepCounter();
            int[] shape = getMdp().getObservationSpace().getShape();
            getHistoryProcessor().startMonitor(getDataManager().getVideoDir() + "/video-" + getEpochCounter() + "-"
                    + getStepCounter() + ".mp4", shape);
        }
    }



    protected int transSize = 0;
    public QLStepReturn<TenhouArray>  trainStep(TenhouArray obs){
        int action = 0;
        INDArray rawInput = getInput(obs);
        INDArray input = DqnUtils.getNNInput(rawInput);

//        boolean isHistoryProcessor = false;

        int skipFrame = 1;
        int historyLength = 1;
        int updateStart = getConfiguration().getUpdateStart()
                + ((getConfiguration().getBatchSize() + historyLength) * skipFrame);

        Double maxQ = Double.NaN;

        if(historyInput == null) {
            historyInput = input;
            rawHistoryInput = rawInput;
            rawHistory = new INDArray[] {rawInput};
        }

        logger.debug("input: " + historyInput);
        logger.debug("rawInput: " + rawHistoryInput);


        INDArray qs = getCurrentDQN().output(historyInput);
        int maxAction = Learning.getMaxAction(qs);
        maxQ = qs.getDouble(maxAction);
        action = getEgPolicy().nextAction(rawHistoryInput);
        logger.debug("qs " + qs);
        logger.debug("maxAction " + maxAction);
        logger.debug("maxQ " + maxQ);
        logger.debug("action " + action);

        if (maxAction != action) {
            Transition<Integer> paneltyTrans = new Transition<>(rawHistory, maxAction, -1, false, rawInput);
            getExpReplay().store(paneltyTrans);
            transSize ++;
        }

        lastAction = action;
        StepReply<TenhouArray> stepReply = getMdp().step(action);
        accuReward += stepReply.getReward() * conf.getRewardFactor();
        logger.debug("The reward " + accuReward);

        RewardLogger.writeLog("maxQ: " + maxQ);
        RewardLogger.writeLog("reward: " + accuReward);

        INDArray rawNInput = getInput(stepReply.getObservation());
        INDArray nInput = DqnUtils.getNNInput(rawNInput);
        INDArray[] rawNHistory = new INDArray[] {rawNInput};

        Transition<Integer> trans = new Transition<>(rawHistory, action, accuReward, stepReply.isDone(), rawNInput);
        getExpReplay().store(trans);
        transSize ++;

//        logger.debug("Get counter " + getStepCounter() + ", " + updateStart);
//        if (getStepCounter() > updateStart) {
        if (stepReply.isDone()){
            logger.info("UPDATE tenhou simple dense dqn NN");
            Pair<INDArray, INDArray> targets = setTarget(((ExpReplay)getExpReplay()).getBatch(transSize));
            transSize = 0;

            getCurrentDQN().fit(targets.getFirst(), targets.getSecond());

            updatedCounter ++;
            if (updatedCounter > UpdateTillSave) {
                updatedCounter = 0;

                String currFileName = DqnSettings.ModelFileName() + System.currentTimeMillis() + ".xml";
                try {
//                    getCurrentDQN()
//                    currentDQN.
                    logger.info("Save model into "+ currFileName);
                    getCurrentDQN().save(currFileName);
                }catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        historyInput = nInput;
        rawHistoryInput = rawNInput;
        accuReward = 0;
        rawHistory = rawNHistory;

        return new QLStepReturn<>(maxQ, getCurrentDQN().getLatestScore(), stepReply);
    }

    protected void printShape(INDArray data, String comment) {
        logger.debug(comment);
        int[] shape = data.shape();
        logger.debug("------------------>" + shape.length);
        for (int i = 0; i < shape.length; i ++) {
            logger.debug(shape[i] + ", ");
        }
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

//        List transIndex = new ArrayList<Integer>();
//        for (int i = 0; i < size; i ++) {
//            transIndex.add(i);
//        }
//        Collections.shuffle(transIndex);

        for (int i = 0; i < size; i++) {
            Transition<Integer> trans = transitions.get(i);
            areTerminal[i] = trans.isTerminal();
            actions[i] = trans.getAction();

            INDArray[] obsArray = trans.getObservation();
            obs.putRow(i, DqnUtils.getNNInput(obsArray[0]));

            nextObs.putRow(i, DqnUtils.getNNInput(trans.getNextObservation()));
        }

        INDArray dqnOutputAr = dqnOutput(obs);
        INDArray dqnOutputNext = dqnOutput(nextObs);

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
            if (!areTerminal[i]) {
                double q = tempQ[i];
                yTar += getConfiguration().getGamma() * q;
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

        printShape(dqnOutputAr, "dqnOutputAr");


        return new Pair<>(obs, dqnOutputAr);
    }


}
