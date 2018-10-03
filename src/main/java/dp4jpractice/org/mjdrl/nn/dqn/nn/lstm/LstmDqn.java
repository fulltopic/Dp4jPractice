package dp4jpractice.org.mjdrl.nn.dqn.nn.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.network.dqn.DQN;

public class LstmDqn<NN extends DQN> extends DQN<NN> {
    public LstmDqn(MultiLayerNetwork mln) {
        super(mln);
    }

    public MultiLayerNetwork getNN() {
        return mln;
    }

}
