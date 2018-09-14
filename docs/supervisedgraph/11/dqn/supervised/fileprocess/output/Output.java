package rl.dqn.supervised.fileprocess.output;

import org.jfree.chart.*;
import org.jfree.chart.plot.PiePlot3D;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.general.PieDataset;
import org.jfree.util.Rotation;

import javax.swing.JFrame;
import java.io.BufferedReader;
import java.io.FileReader;

public class Output extends JFrame {
    private static final long serialVersionUID = 1L;

    public Output(String applicationTitle, String chartTitle) {
        super(applicationTitle);

        try {
            JFreeChart lineChart = ChartFactory.createLineChart(
                    chartTitle,
                    "acc ", "f1",
                    createCsvDataset(),
                    PlotOrientation.VERTICAL,
                    true, true, false
            );

            ChartPanel chartPanel = new ChartPanel(lineChart);
            chartPanel.setPreferredSize(new java.awt.Dimension(560 * 5, 367));
            setContentPane(chartPanel);
        }catch(Exception e) {
            e.printStackTrace();
        }

    }


    private DefaultCategoryDataset createCsvDataset() throws Exception {
        DefaultCategoryDataset ds = new DefaultCategoryDataset();
        String fileName = "/home/zf/workspaces/workspace_java/tenhoulogs/logs/db4/dqrn/player/output8.csv";
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line;

        String headers = reader.readLine();
        System.out.println(headers);
        int seq = 0;

        while ((line = reader.readLine()) != null) {
            String[] params = line.split(",");
            System.out.println(params[7]);
            ds.addValue(Double.valueOf(params[13]), "dnk", params[7]); //Double.valueOf(params[7]).toString());
            seq += 1;
        }

//        ds.addValue( 15 , "schools" , "1970" );
//        ds.addValue( 30 , "schools" , "1980" );
//        ds.addValue( 60 , "schools" ,  "1990" );
//        ds.addValue( 120 , "schools" , "2000" );
//        ds.addValue( 240 , "schools" , "2010" );
//        ds.addValue( 300 , "schools" , "2014" );

        return ds;
    }




    public static void main(String[] args) {
        Output demo = new Output("Comp", "so?");
        demo.pack();
        demo.setVisible(true);
    }
}
