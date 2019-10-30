package dp4jpractice.org.mjdrl.nn.dqn.nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DqnUtils {
    static Logger logger = LoggerFactory.getLogger(DqnUtils.class);

    public static INDArray getNNInput(INDArray input)  {
        INDArray nnInput = input.dup();

        for (int i = 0; i < nnInput.shape()[1]; i ++) {
            double v = nnInput.getDouble(0, i);
            double remain = v - (double)((int)v);
            double nv = ((double)((int)v & 7) + remain) / 4.0;
            nnInput.putScalar(0, i, nv);
        }

//        System.out.println("Shape " + nnInput.shape().length);
//        for(int i = 0; i < nnInput.shape().length; i ++) {
//            System.out.print(nnInput.shape()[i] + ", ");
//        }
//        System.out.println("");

        return nnInput;
    }

    public static INDArray getNNFInput(INDArray input)  {
        INDArray nnInput = Nd4j.create(new long[]{input.shape()[1], input.shape()[0]}, 'f');

        for (int i = 0; i < nnInput.shape()[1]; i ++) {
            double v = nnInput.getDouble(0, i);
            double remain = v - (double)((int)v);
            double nv = ((double)((int)v & 7) + remain) / 4.0;
            nnInput.putScalar(i, 0, nv);
        }

        StringBuilder logContent = new StringBuilder();
        logContent.append("Shape ");
        logContent.append(nnInput.shape().length);
        logContent.append("\n");
        for(int i = 0; i < nnInput.shape().length; i ++) {
            logContent.append(nnInput.shape()[i]);
            logContent.append(", ");
        }
        logContent.append("\n");
        logger.debug(logContent.toString());

        return nnInput;
    }

}
