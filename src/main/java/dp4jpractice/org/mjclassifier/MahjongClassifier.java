package dp4jpractice.org.mjclassifier;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MahjongClassifier {
    private static final int Seed = 71;
    private static final int ParamSeed = 89;
    private static Random ParamRandom = new Random(ParamSeed);
    private static Random MjRandom = new Random(Seed);
    private static final int ChannelNum = 1;
    private static final int ClassNum = 3;
    private static final int ScratchHeight = 30;
    private static final int ScratchWidth = 20;
    private static final int TestBatchSize = 9;
    private static final int TestEpochNum = 1;

    private static final String ResourcePath = "datasets/";
    private static final String ResourceTrainPath = "/dataset6/";
    private static final String ResourceTestPath = "/datasettest/";
    private static final String ResourceScratchReportPath = "/reports/";
    private static final String ResourceModelPath = "/models/lenet.zip";
    private static final String ReportPrefix = "/scratchreport";
    private String trainPath = "";
    private String testPath = "";
    private String scratchReportPath = "";
    private String transferModelPath = "";

    private static int TransferHeight = 28;
    private static int TransferWidth = 28;
    private static int TransferBatchSize = 16;

    private BufferedWriter reportWriter;
    private DataNormalization normalizer;

    private static final Logger log = LoggerFactory.getLogger(MahjongClassifier.class);


    private abstract class Model {
        abstract public void fit(DataSetIterator iterator, int epochNum);
        abstract protected void addListener(TrainingListener listener);

        public void setListeners(boolean setUIiListener, boolean setEvalListener, DataSetIterator validIte)
        {
            if (setUIiListener) {
                UIServer uiServer = UIServer.getInstance();
                InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
                uiServer.attach(statsStorage);

                addListener(new StatsListener(statsStorage));
            }
            if (setEvalListener) {
                addListener(new EvaluativeListener(validIte, 1, InvocationType.EPOCH_END));
            }
        }
    }

    private class NetworkModel extends Model{
        private MultiLayerNetwork model;

        NetworkModel(MultiLayerNetwork model) {
            this.model = model;
        }

        public void fit(DataSetIterator iterator, int epochNum) {
            model.fit(iterator, epochNum);
        }

        protected void addListener(TrainingListener listener) {
            model.addListeners(listener);
        }
    }

    private class GraphModel extends Model {
        private ComputationGraph graph;

        GraphModel(ComputationGraph graph) {
            this.graph = graph;
        }

        public void fit(DataSetIterator iterator, int epochNum) {
            graph.fit(iterator, epochNum);
        }

        protected  void addListener(TrainingListener listener) {
            graph.addListeners(listener);
        }
    }

    public MahjongClassifier() throws Exception {
        File resourceDir = new File(ResourcePath);
        String resourcePath = resourceDir.getAbsolutePath();
        trainPath = resourcePath + ResourceTrainPath;
        testPath = resourcePath + ResourceTestPath;
        scratchReportPath = resourcePath + ResourceScratchReportPath;
        transferModelPath = resourcePath + ResourceModelPath;

        this.normalizer = new ImagePreProcessingScaler(0, 1);
        String fileName = scratchReportPath + ReportPrefix + System.currentTimeMillis() + ".txt";
        reportWriter = new BufferedWriter(new FileWriter(fileName));
    }

    public void close() {
        try {
            reportWriter.close();
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    private DataSetIterator createDataSetIterator(String path, int height, int width, int batchSize) throws Exception {
        log.info("Prepare dataset");
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        File parentFile = new File(path);
        FileSplit split = new FileSplit(parentFile, NativeImageLoader.ALLOWED_FORMATS, MjRandom);

        ImageRecordReader recordReader = new ImageRecordReader(height, width, ChannelNum, labelGenerator);
        recordReader.initialize(split);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, ClassNum);
        dataIter.setPreProcessor(normalizer);
        normalizer.fit(dataIter);

        return dataIter;
    }

    private MultiLayerNetwork createScratchModel(double l2NormBeta, double learningRate, int l1KernelSize) throws Exception{
        log.info("==========================================================> Get params " + l2NormBeta + ", " + learningRate + ", " + l1KernelSize);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(Seed)
                .updater(new AdaGrad(learningRate))
                .l2(l2NormBeta)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                //Layers
                .layer(0, new ConvolutionLayer.Builder(l1KernelSize, l1KernelSize)
                        .padding(1, 1)
                        .stride(1, 1)
                        .nIn(ChannelNum)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(ClassNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                //Input
                .setInputType(InputType.convolutionalFlat(ScratchHeight, ScratchWidth, ChannelNum)).build();
        config.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);

        List<NeuralNetConfiguration> confs = config.getConfs();
        log.info("===================================================> Get config size " + confs.size());
        for (NeuralNetConfiguration conf : confs) {
            log.info("Conf : \n" + conf.toJson());
        }

        //Give JVM enough time to deal with warning in MultiLayerConfiguration building
        Thread.sleep(10000L, 0);


        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        return model;
    }

    private ComputationGraph createTransferModel(double learningRate) throws Exception {
        ComputationGraph pretrainedNet = (ComputationGraph) ModelSerializer.restoreComputationGraph(
                new File(transferModelPath));


        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new AdaGrad(learningRate))
                .seed(Seed)
                .build();

        return new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc")
                .removeVertexAndConnections("output")
                .addLayer("output",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(500)
                                .nOut(ClassNum)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc")
                .setOutputs("output")
                .build();
    }

    private void reportParams(int batchSize, int epochNum, double l2NormBeta, double learningRate, int l1KernelSize) {
        report("\n\n\n/**************************************** Case ***************************************/");
        StringBuffer buffer = new StringBuffer();
        double[] param = {batchSize, epochNum, l2NormBeta, learningRate, l1KernelSize};
        for(int i = 0; i < param.length; i ++) {
            buffer.append(param[i]);
            buffer.append(", ");
        }
        report(buffer.toString());
        report("");
    }

    public void runScratchModel(int batchSize, int epochNum, double l2NormBeta, double learningRate, int l1KernelSize) throws Exception {
        log.info("******************************* Get params " + l2NormBeta + ", " + learningRate + ", " + l1KernelSize);
        reportParams(batchSize, epochNum, l2NormBeta, learningRate, l1KernelSize);

        MultiLayerNetwork network = createScratchModel(l2NormBeta, learningRate, l1KernelSize);
        NetworkModel model = new NetworkModel(network);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage));

        runModel(model, batchSize, epochNum, ScratchHeight, ScratchWidth);
    }


    public void runTransferModel(int epochNum, double learningRate) throws Exception {
        double l2NormBeta = 0.0005;
        int l1KernelSize = 5;
        int batchSize = TransferBatchSize;
        log.info("******************************* Get params " + l2NormBeta + ", " + learningRate + ", " + l1KernelSize);
        reportParams(batchSize, epochNum, l2NormBeta, learningRate, l1KernelSize);

        ComputationGraph graph = createTransferModel(learningRate);
        GraphModel model = new GraphModel(graph);

        runModel(model, batchSize, epochNum, TransferHeight, TransferWidth);
    }

    private void runModel(Model model, int batchSize, int epochNum, //double l2NormBeta, double learningRate, int l1KernelSize,
                          int height, int width)
            throws Exception
    {
        DataNormalization normalizer = new ImagePreProcessingScaler(0, 1);

        log.info("Prepare dataset");
        ParentPathLabelGenerator trainLabelGenerator = new ParentPathLabelGenerator();
        File trainParentFile = new File(trainPath);
        FileSplit trainSplit = new FileSplit(trainParentFile, NativeImageLoader.ALLOWED_FORMATS, MjRandom);

        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, ChannelNum, trainLabelGenerator);
        trainRecordReader.initialize(trainSplit);
        DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, ClassNum);
        trainDataIter.setPreProcessor(normalizer);
        normalizer.fit(trainDataIter);


        model.fit(trainDataIter, epochNum);

        DataSetIterator testDataIter = createDataSetIterator(testPath, height, width, TestBatchSize);
        model.setListeners(true, true, testDataIter);

        ImageTransform flipTransform = new FlipImageTransform(new Random(123));
        ImageTransform flipTransform1 = new FlipImageTransform(Seed);

        List<ImageTransform> transforms = Arrays.asList(
                flipTransform,
                flipTransform1);

        for (ImageTransform transform: transforms) {
            trainRecordReader.initialize(trainSplit, transform);
            trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, ClassNum);
            trainDataIter.setPreProcessor(normalizer);
            normalizer.fit(trainDataIter);
            model.fit(trainDataIter, epochNum);
        }

        DataSetIterator testEpochIte = new MultipleEpochsIterator(TestEpochNum, testDataIter);


//        Evaluation eval = model.evaluate(testEpochIte);
//        log.info("-------------> " + eval.stats());
//        System.out.println("-----------------> " + eval.stats());
//
//        reportWriter.write(eval.stats());
//        reportWriter.flush();

        System.out.println("End of tests");
        log.info("End of test");

    }

    private void runAllScratchModels(double[][] params, int paramIndex, double[] param) throws Exception{
        if (paramIndex >= params.length) {
            runScratchModel((int)param[0], (int)param[1], param[2], param[3], (int)param[4]);
            return;
        }

        for(int i = 0; i < params[paramIndex].length; i ++) {
            param[paramIndex] = params[paramIndex][i];
            runAllScratchModels(params, paramIndex + 1, param);
        }
    }

    private void report(String content) {
        try {
            reportWriter.write(content + "\n");
            reportWriter.flush();
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public void runScratchModels() throws Exception {
        report("Running scratch models");
//        runScratchModel(16, 20, 0.0005, 0.01, 2);

        double[] batchSizes = {16,};
        double[] epochNums = {15, 20, 25, 30};
        double[] l2NormBetas = {0.0005, 0.0002, 0.0001};
        double[] learningRates = {0.02, 0.01, 0.005, 0.001};
        double[] l1KernelSizes = {2, 3, 4, 5};

        double[][] params = {batchSizes, epochNums, l2NormBetas, learningRates, l1KernelSizes};
        double[] modelParam = new double[params.length];
        Arrays.fill(modelParam, 0);

        runAllScratchModels(params, 0, modelParam);
    }

    private void runAllTransferModels(double[][] params, int paramIndex, double[] param) throws Exception {
        report("Running transfer models ");

        if (paramIndex >= params.length) {
            runTransferModel((int)param[0], param[1]);
            return;
        }

        for(int i = 0; i < params[paramIndex].length; i ++) {
            param[paramIndex] = params[paramIndex][i];
            runAllTransferModels(params, paramIndex + 1, param);
        }
    }

    public void runTransferModels() throws Exception {
        double[] epochNums = {15, 20, 25, 30};
        double[] learningRates = {0.02, 0.01, 0.005, 0.001};

        double[][] params = { epochNums, learningRates};
        double[] modelParam = new double[params.length];
        Arrays.fill(modelParam, 0);

        runAllTransferModels(params, 0, modelParam);
    }

    private int getBatchSize() {
        return 16;
//        int r = ParamRandom.nextInt(3) + 4;
//        return (int)Math.pow(2, r);
    }

    private int getEpochNum1() {
        int r = ParamRandom.nextInt(20);
        return r + 15;
    }

    private double getL2NormBeta1() {
        //0.0001 ~ 0.0005
        return ParamRandom.nextInt(11) * 0.0001;
    }

    private double getLearningRate1() {
        double r = ParamRandom.nextInt(3)  + 1;
        return Math.pow(10, (-1) * r);
    }

    private int getKernelSize1() {
        int r = ParamRandom.nextInt(5);
        return r + 2;
    }


    private int getEpochNum2() {
        return 22;
    }

    private double getL2NormBeta2() {
        //0.0001 ~ 0.0005
        double[] betas = {0.0001, 0.0003, 0.0008};
        int index = ParamRandom.nextInt(betas.length);
        return betas[index];
    }

    private double getLearningRate2() {
        double r = ParamRandom.nextInt(10) * 0.001;
        return 0.005 + r;
    }

    private int getKernelSize2() {
        return 3;
    }


    private void runRandomSearchScratchModel1() throws Exception{
        runScratchModel(getBatchSize(),
                getEpochNum1(),
                getL2NormBeta1(),
                getLearningRate1(),
                getKernelSize1());
    }

    private void runRandomSearchScratchModel2() throws Exception{
        runScratchModel(getBatchSize(),
                getEpochNum2(),
                getL2NormBeta2(),
                getLearningRate2(),
                getKernelSize2());
    }

    public void runRamdomSearchScratchModels(int sampleNum) throws Exception {
        report("Run random search scratch models");
        if (sampleNum <= 0) {
            throw new IllegalStateException("Sample number should be > 0");
        }

        for(int i = 0; i < sampleNum; i ++) {
            runRandomSearchScratchModel2();
        }
    }


    public static void main(String[] args) {
        try {
            MahjongClassifier classifier = new MahjongClassifier();
//            classifier.runScratchModels();
            classifier.runScratchModel(16, 80, 0.0005, 0.01, 3);
//            classifier.runTransferModel(32, 0.02);
//            classifier.runTransferModels();
//            classifier.runRamdomSearchScratchModels(1);

            classifier.close();
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

}
