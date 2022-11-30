package controllers.learningmodel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class EpsGreedyQPolicy extends Policy {

    public double eps = 0.1;
    public Random rd1 = new Random();
    public Random rd2 = new Random();

    public EpsGreedyQPolicy(double eps)
    {
        this.eps = eps;
    }

    public EpsGreedyQPolicy()
    {
    }

    public void setEps(double eps)
    {
        this.eps = eps;
    }

    public double getEps()
    {
        return this.eps;
    }

    public int selectAction(INDArray qValues)
    {

        assert(qValues.rank() == 1);
        int nbActions = (int)qValues.size(0);
        assert(qValues.rows() == 1);
        nbActions = qValues.columns();

        int action;
        if( rd1.nextDouble() < this.eps )
        {
            action = rd2.nextInt(nbActions);
        }
        else
        {
            action = Nd4j.argMax(qValues, 0).getInt(0);

        }
        return action;


    }

}
