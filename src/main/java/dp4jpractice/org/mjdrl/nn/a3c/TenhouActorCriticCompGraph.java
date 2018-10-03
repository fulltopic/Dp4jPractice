package dp4jpractice.org.mjdrl.nn.a3c;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticCompGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TenhouActorCriticCompGraph<NN extends TenhouActorCriticCompGraph> extends ActorCriticCompGraph<NN> {
    public TenhouActorCriticCompGraph(ComputationGraph cg) {
        super(cg);
    }

    public ComputationGraph getCg() {
        return this.cg;
    }

    public INDArray[] myOutput(INDArray... input) {
        return this.cg.output(false, input);
    }

    public NN clone() {
        System.out.println("TenhouActorCriticCompGraph clone");
        NN nn = (NN)new TenhouActorCriticCompGraph(cg.clone());
        nn.cg.setListeners(cg.getListeners());
        return nn;
    }

    public void copy(NN from) {
        System.out.println("TenhouActorCriticCompGraph copy");
        cg.setParams(from.cg.params());
    }
}
