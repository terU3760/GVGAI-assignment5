package miscellaneous.util;

import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.util.List;

public class Utility {


    public static INDArray asMatrix(BufferedImage image, ImageLoader imageLoader, int channels) {
        if (channels == 3L) {
            return imageLoader.toBgr(image);
        } else {

            int w = image.getWidth();
            int h = image.getHeight();
            INDArray ret = Nd4j.create(new int[]{h, w});

            for(int i = 0; i < h; ++i) {
                for(int j = 0; j < w; ++j) {
                    ret.putScalar(new int[]{i, j}, image.getRGB(j, i) & 0xFF );
                }
            }

            return ret;
        }
    }





    public static double sum(List<Double> list) {
        double sum = 0;
        for (double i: list) {
            sum += i;
        }
        return sum;
    }




    
}
