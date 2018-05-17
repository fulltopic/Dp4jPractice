package dp4jpractice.org.utils;

import java.io.*;

public class Report2CSV {
    private String srcFileName;
    private String dstFileName;
    private BufferedReader reader = null;
    private BufferedWriter writer = null;

    public Report2CSV(String srcFileName, String dstFileName) {
        this.srcFileName = srcFileName;
        this.dstFileName = dstFileName;
    }

    public void convert() throws Exception {
        try {
            reader = new BufferedReader(new FileReader(srcFileName));
            writer = new BufferedWriter(new FileWriter(dstFileName));

            writer.write("Batch, Epoch, L2Reg, LearningRate, L1KernelSize, Accuracy, Precision, Recall, F1 Score \n");
            writer.flush();

            String dstContent = "";
            String srcContent = reader.readLine();
            while (srcContent != null) {
                if (srcContent.contains("Case")) {
                    dstContent = reader.readLine();
                }

                if (srcContent.contains("Accuracy")) {
                    dstContent += srcContent.split(":")[1].split("\\(")[0].trim() + ", ";
                }
                if (srcContent.contains("Precision")) {
                    dstContent += srcContent.split(":")[1].split("\\(")[0].trim() + ", ";
                }
                if (srcContent.contains("Recall")) {
                    dstContent += srcContent.split(":")[1].split("\\(")[0].trim() + ", ";
                }
                if (srcContent.contains("F1 Score")) {
                    dstContent += srcContent.split(":")[1].split("\\(")[0].trim() + ", ";

                    writer.write(dstContent + "\n");
                    writer.flush();
                }

                srcContent = reader.readLine();
            }
        }finally {
            if(reader != null) {
                reader.close();
            }
            if(writer != null) {
                writer.close();
            }
        }
    }

    public static void main(String[] args) {
        File reportsDir = new File("datasets/reports");
        String srcPath = reportsDir.getAbsolutePath() + "/scratchreport1526541748551.txt";
        String dstPath = reportsDir.getAbsolutePath() + "/scratchreport1526541748551.csv";

        Report2CSV converter = new Report2CSV(srcPath, dstPath);
        try {
            converter.convert();
        }catch(Exception e) {
            e.printStackTrace();
        }

    }

}
