package rl.dqn.reinforcement.dqn.nn.a3c;

import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class TenhouRnnToCnnPreProcessor extends RnnToCnnPreProcessor {
    Logger logger = LoggerFactory.getLogger(TenhouRnnToCnnPreProcessor.class);

    public TenhouRnnToCnnPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                                @JsonProperty("inputWidth") int inputWidth, @JsonProperty("numChannels") int numChannels) {
        super(inputHeight, inputWidth, numChannels);
        System.out.println("----------------------------------------------> TenhouRnnToCnnPreProcessor constructor");
        logger.debug("----------------------------------------------> TenhouRnnToCnnPreProcessor constructor");
    }

    public INDArray preProcess(INDArray input, int miniBatchSize) {
//        System.out.println("----------------------------------------------> rnntocnn preprocess");
//        logger.debug("----------------------------------------------> rnntocnn preprocess " + input.shape().length + ": " + input.shape()[0] + " " + input.shape()[1]);



        INDArray input2 = input;
        if (input.shape().length < 3) {
            input2 = input.dup().reshape(input.shape()[0], input.shape()[1], 1);
        }
//        logger.debug("----------------------------------------------> rnntocnn after preprocess " + input2.shape().length + ": " + input2.shape()[0] + " " + input2.shape()[1] + " " + input2.shape()[2]);


        INDArray output = super.preProcess(input2, miniBatchSize);
//        logger.debug("----------------------------------------------> rnntocnn output " + output.shape().length + ": " + output.shape()[0] + " " + output.shape()[1] + " " + output.shape()[2] + " " + output.shape()[3]);

//        output.fmodi(7).divi(4);

//        for (int i = 0; i < output.shape()[0]; i ++) {
//            for (int j = 0; j < output.shape()[1]; j ++) {
//                for (int k = 0; k < output.shape()[2]; k ++) {
//                    for (int t = 33; t < 35; t ++) {
//                        if (output.getDouble(i, j, k, t) > 0) {
//                            output.putScalar(i, j, k, t, 1.0);
//                        }
//                    }
//                }
//            }
//        }

//        for (int i = 0; i < output.shape()[0]; i ++) {
//            for (int j = 0; j < output.shape()[1]; j ++) {
//                for (int k = 0; k < 1; k ++) {
//                    int v = output.getInt(i, j, k);
//                    double newV = (double)(v & 7) / 4.0;
//                    output.putScalar(i, j, k, newV);
//                }
//            }
//        }

//        logger.debug("Preprocessor " + output);

        return output;
    }

}
