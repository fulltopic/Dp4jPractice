package rl.dqn.supervised.utils;


import org.deeplearning4j.rl4j.learning.sync.Transition;

import java.io.Serializable;
import java.util.Vector;

public class ReplayDB implements Serializable {
    public Vector<DbTransition> datas = null;

    public ReplayDB() {
        datas = new Vector<>();
    }

    public void add(Transition<Integer> tran) {
        datas.add(new DbTransition(tran));
    }

    public void add(DbTransition tran) { datas.add(tran);}
}
