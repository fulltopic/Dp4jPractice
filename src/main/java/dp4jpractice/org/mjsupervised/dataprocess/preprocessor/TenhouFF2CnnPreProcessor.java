package dp4jpractice.org.mjsupervised.dataprocess.preprocessor;

import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class TenhouFF2CnnPreProcessor extends FeedForwardToCnnPreProcessor {
    Logger logger = LoggerFactory.getLogger(TenhouFF2CnnPreProcessor.class);

    public TenhouFF2CnnPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                                    @JsonProperty("inputWidth") int inputWidth, @JsonProperty("numChannels") int numChannels) {
        super (inputHeight, inputWidth, numChannels);
    }

    public INDArray preProcess(INDArray input, int miniBatchSize) {
        System.out.println("input shape " + Arrays.toString(input.shape()));
        System.out.println("input" + input);

        INDArray output = Nd4j.zeros(input.shape()[0], input.shape()[1]);

        for (int i = 0; i < input.shape()[0]; i ++) {
            for (int j = 0; j < input.shape()[1]; j ++) {
                int v = input.getInt(i, j);
                double newV = ((double)(v & 7)) / 4.0;
                output.putScalar(i, j, newV);
            }
        }

        logger.debug("Processed input");
        logger.debug("output " + output);
       return super.preProcess(output, miniBatchSize);
    }
}
