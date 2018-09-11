package rl.dqn.reinforcement.dqn.nn.a3c;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TenhouInputPreProcessor implements InputPreProcessor {
    Logger logger = LoggerFactory.getLogger(TenhouInputPreProcessor.class);

    public INDArray preProcess(INDArray input, int miniBatchSize) {
        INDArray output = input.dup();
        int[] shape = input.shape();
        logger.debug("Preprocessor----------------------> shape " + shape.length);
//        StackTraceElement[] elements = Thread.currentThread().getStackTrace();
//        for (int i = 0; i < elements.length; i ++) {
//            StackTraceElement s = elements[i];
//            System.out.println("\tat " + s.getClassName() + "." + s.getMethodName()
//                    + "(" + s.getFileName() + ":" + s.getLineNumber() + ")");
//        }

//        for (int i = 0; i < shape.length; i ++) {
//            System.out.print(shape[i] + ", ");
//        }
//        System.out.println(" ");

        for (int i = 0; i < shape[0]; i ++) {
            for (int k = 0; k < shape[1]; k ++) {
                double v = (int)(Math.floor(input.getDouble(i, k))) & 7;
                output.putScalar(i, k, v / 4);
            }
        }

//        logger.debug("Original " + input);
//        logger.debug("Processed " + output);

        return output;
    }

    public INDArray backprop(INDArray output, int miniBatchSize) {
        return output;
    }

    public InputPreProcessor clone() {
        return new TenhouInputPreProcessor();
    }

    public InputType getOutputType(InputType inputType) {
        return inputType;
    }

    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        //Assume mask array is 2d for time series (1 value per time step)
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else if (maskArray.rank() == 2) {
            //Need to reshape mask array from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1]
            return new Pair<>(TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray), currentMaskState);
        } else {
            throw new IllegalArgumentException("Received mask array of rank " + maskArray.rank()
                    + "; expected rank 2 mask array. Mask array shape: " + Arrays.toString(maskArray.shape()));
        }
    }

}
