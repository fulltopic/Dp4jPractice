package rl.dqn.supervised.utils;

import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public class DbTransition implements Serializable{

    private double reward;
    private boolean isTerminal;
    private INDArray nextObservation;
    private INDArray[] observation;
    private int action;


    public INDArray[] getObservation() {
        return observation;
    }

    public int getAction() {
        return action;
    }

    public double getReward() {
        return reward;
    }

    public boolean isTerminal() {
        return isTerminal;
    }

    public INDArray getNextObservation() {
        return nextObservation;
    }

    // No copy
    public DbTransition(Transition<Integer> tran) {
        this.reward = tran.getReward();
        this.isTerminal = tran.isTerminal();
        this.nextObservation = tran.getNextObservation();
        this.observation = tran.getObservation();
        this.action = tran.getAction();

        for (int i = 0; i < this.observation.length; i ++) {
            //TODO: Replace 8 with package defined constant
            this.observation[i].fmodi(8);
            this.observation[i].divi(4); //normalization
        }
    }

    public Transition<Integer> toTransition() {
        return new Transition<>(observation, action, reward, isTerminal, nextObservation);
    }

}
