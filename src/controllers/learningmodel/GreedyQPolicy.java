package controllers.learningmodel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GreedyQPolicy extends Policy {


    public GreedyQPolicy()
    {
    }


    public int selectAction(INDArray qValues)
    {

        long[] qValuesShape = qValues.shape();
        assert( ( ( qValuesShape.length == 1 ) || ( qValuesShape.length == 2 ) ) && ( qValues.rank() == 1 ) );

        int action = Nd4j.argMax(qValues, -1).getInt(0);
        return action;

    }


}
