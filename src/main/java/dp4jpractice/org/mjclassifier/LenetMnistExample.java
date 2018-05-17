package dp4jpractice.org.mjclassifier;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;

public class LenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(LenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 16; // Test batch size
        int nEpochs = 1; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; //

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        /*
            Construct the neural network
         */
        log.info("Build model....");

        // learning rate schedule in the form of <Iteration #, Learning Rate>
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        ComputationGraphConfiguration graphConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(.01)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .graphBuilder()
                .addInputs("in")
                .addLayer("conv_1", new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .name("conv_1")
                        .build(), "in")
                .addLayer("pool_1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .name("pool_1")
                        .build(), "conv_1")
                .addLayer("conv_2", new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .name("conv_2")
                        .build(), "pool_1")
                .addLayer("pool_2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .name("pool_2")
                        .build(), "conv_2")
                .addLayer("fc", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).name("fc").build(), "pool_2")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .name("output")
                        .build(), "fc")
                .setInputTypes(InputType.convolutionalFlat(28,28,1)) //See note below
                .setOutputs("output")
                .backprop(true).pretrain(false).build();


//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(iterations) // Training iterations as above
//                .regularization(true).l2(0.0005)
//                /*
//                    Uncomment the following for learning decay and bias
//                 */
//                .learningRate(.01)//.biasLearningRate(0.02)
//                /*
//                    Alternatively, you can use a learning rate schedule.
//
//                    NOTE: this LR schedule defined here overrides the rate set in .learningRate(). Also,
//                    if you're using the Transfer Learning API, this same override will carry over to
//                    your new model configuration.
//                */
//                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
//                .learningRateSchedule(lrSchedule)
//                /*
//                    Below is an example of using inverse policy rate decay for learning rate
//                */
//                //.learningRateDecayPolicy(LearningRatePolicy.Inverse)
//                //.lrPolicyDecayRate(0.001)
//                //.lrPolicyPower(0.75)
//                .weightInit(WeightInit.XAVIER)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
//                .list()
//                .layer(0, new ConvolutionLayer.Builder(5, 5)
//                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
//                        .nIn(nChannels)
//                        .stride(1, 1)
//                        .nOut(20)
//                        .activation(Activation.IDENTITY)
//                        .name("conv_1")
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .name("pool_1")
//                        .build())
//                .layer(2, new ConvolutionLayer.Builder(5, 5)
//                        //Note that nIn need not be specified in later layers
//                        .stride(1, 1)
//                        .nOut(50)
//                        .activation(Activation.IDENTITY)
//                        .name("conv_2")
//                        .build())
//                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .name("pool_2")
//                        .build())
//                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
//                        .nOut(500).name("fc").build())
//                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(outputNum)
//                        .activation(Activation.SOFTMAX)
//                        .name("output")
//                        .build())
//                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
//                .backprop(true).pretrain(false).build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        ComputationGraph model = new ComputationGraph(graphConfiguration);
        model.init();


//        log.info("Train model....");
//        model.setListeners(new ScoreIterationListener(1));

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            mnistTest.reset();
        }
        log.info("****************Example finished********************");


//        String fileName = "/home/zf/books/DeepLearning-master/majiong/datasets/models/lenet.zip";
//        File MJModelFile = new File(fileName);
        File MJModelFile = new File("datasets/models/lenet.zip");
        FileOutputStream fos = new FileOutputStream(MJModelFile);
        ModelSerializer.writeModel(model, fos, true);
        log.info("End of model saving");
    }

}
