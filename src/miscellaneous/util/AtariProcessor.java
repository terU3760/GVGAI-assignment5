package miscellaneous.util;

import org.nd4j.linalg.api.ndarray.INDArray;

public class AtariProcessor implements Processor {


    public double[][] process_state_batch(int[][] batch)
    {

        double[][] processed_batch = new double[batch.length][batch[0].length];

        for( int i = 0 ; i < batch.length ; i++ )
        {

            for( int j = 0; j < batch[0].length; j++ )
            {

                double temp = batch[ i ][ j ];
                processed_batch[ i ][ j ] = temp / 255.0 ;


            }


        }

        return processed_batch;

    }


    public double[][][] process_state_batch(int[][][] batch)
    {


        double[][][] processed_batch = new double[batch.length][batch[0].length][batch[0][0].length];

        for( int i = 0 ; i < batch.length ; i++ )
        {

            for( int j = 0 ; j < batch[0].length ; j++ )
            {

                for( int k = 0 ; k < batch[0][0].length ; k++ )
                {

                    double temp = batch[i][j][k];
                    processed_batch[i][j][k] = temp / 255.0;


                }

            }

        }


        return processed_batch;


    }


    public INDArray processStateBatch(INDArray batch)
    {
        return batch.divi(255.0);
    }



}
