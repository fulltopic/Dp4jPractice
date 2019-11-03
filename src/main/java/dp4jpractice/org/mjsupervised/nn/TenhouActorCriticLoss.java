package dp4jpractice.org.mjsupervised.nn;

import org.deeplearning4j.rl4j.network.ac.ActorCriticLoss;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TenhouActorCriticLoss extends ActorCriticLoss {
    private Logger logger = LoggerFactory.getLogger(TenhouActorCriticLoss.class);
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
//        logger.debug("computeGradient labels" + labels);
//        logger.debug("computeGradient preOutput" + preOutput);

        INDArray output = activationFn.getActivation(preOutput.dup(), true).addi(1e-5);

//        logger.debug("computeGradient output " + output);
        INDArray logOutput = Transforms.log(output, true);
        INDArray entropyDev = logOutput.addi(1);
        INDArray dLda = output.rdivi(labels).subi(entropyDev.muli(BETA)).negi();
        INDArray grad = activationFn.backprop(preOutput, dLda).getFirst();

        if (mask != null) {
            LossUtil.applyMask(grad, mask);
        }

        return grad;
    }
}
