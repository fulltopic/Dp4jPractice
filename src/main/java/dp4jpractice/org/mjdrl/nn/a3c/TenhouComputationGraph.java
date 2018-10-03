package dp4jpractice.org.mjdrl.nn.a3c;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//For deep clone
public class TenhouComputationGraph extends ComputationGraph {
    Logger logger = LoggerFactory.getLogger(TenhouComputationGraph.class);

    public TenhouComputationGraph(ComputationGraphConfiguration configuration) {
        super(configuration);
    }

    @Override
    public void computeGradientAndScore() {
        if (configuration.getBackpropType() == BackpropType.TruncatedBPTT) {
            logger.debug("++++++++++++++++++++++++++++++++ " + "Truncated");
        }else {
            logger.debug("++++++++++++++++++++++++++++++++ " + "Standard");
        }

        super.computeGradientAndScore();
    }

    @Override
    public TenhouComputationGraph clone() {
        System.out.println("----------------------------------> TenhouComputationGraph clone");
        TenhouComputationGraph cg = new TenhouComputationGraph(configuration.clone());
        cg.init(params().dup(), false);
        if (solver != null) {
            //If  solver is null: updater hasn't been initialized -> getUpdater call will force initialization, however
            ComputationGraphUpdater u = this.getUpdater();
            INDArray updaterState = u.getStateViewArray();
            if (updaterState != null) {
                cg.getUpdater().setStateViewArray(updaterState.dup());
            }
        }
//        cg.listeners = this.listeners;
//        for (int i = 0; i < topologicalOrder.length; i++) {
//            if (!vertices[topologicalOrder[i]].hasLayer())
//                continue;
//            String layerName = vertices[topologicalOrder[i]].getVertexName();
//            if (getLayer(layerName) instanceof FrozenLayer) {
//                cg.getVertex(layerName).setLayerAsFrozen();
//            }
//        }
        return cg;
    }
}
