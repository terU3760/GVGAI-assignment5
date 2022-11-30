package miscellaneous.util;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Processor {
    public double[][] process_state_batch(int[][] batch);
    public double[][][] process_state_batch(int[][][] batch);

    public INDArray processStateBatch(INDArray batch);

}
