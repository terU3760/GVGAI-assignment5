package controllers.learningmodel;

import jdk.jshell.spi.ExecutionControl;
import org.apache.commons.lang.NotImplementedException;
import org.nd4j.linalg.api.ndarray.INDArray;

public class LinearAnnealedPolicy extends Policy {


    public Policy innerPolicy;
    public double valueMax;
    public double valueMin;
    public double valueTest;
    public int nbSteps;





    public int accumulatedGameTicks = 0;





    public LinearAnnealedPolicy(Policy innerPolicy, double valueMax, double valueMin, double valueTest, int nbSteps)
    {

        this.innerPolicy = innerPolicy;
        this.valueMax = valueMax;
        this.valueMin = valueMin;
        this.valueTest = valueTest;
        this.nbSteps = nbSteps;





    }


    public double getCurrentValue() throws NotImplementedException
    {

        if( !(this.getAgent() instanceof DQNVanillaAgent) )
        {
            throw new NotImplementedException();
        }
        else
        {
            double value;
            DQNVanillaAgent agent = (DQNVanillaAgent) (this.getAgent());
            if(agent.training) {
                // Linear annealed:f(x) = ax + b.
                double a = -( this.valueMax - this.valueMin ) / this.nbSteps;
                double b = this.valueMax;
                value = Math.max(this.valueMin, a * ( agent.game.getGameTick() + this.accumulatedGameTicks ) + b );
            }
            else
            {
                value = this.valueTest;
            }
            return value;
        }

    }


    public int selectAction(INDArray qValues)
    {

        if( !( this.innerPolicy instanceof EpsGreedyQPolicy) )
        {
            throw new NotImplementedException();
        }
        else
        {
            ((EpsGreedyQPolicy)(this.innerPolicy)).setEps(this.getCurrentValue());
            return ((EpsGreedyQPolicy)(this.innerPolicy)).selectAction(qValues);
        }

    }


    public double metrics()
    {

        return ((EpsGreedyQPolicy)(this.innerPolicy)).getEps();

    }

    public String[] metricNames()
    {

        return new String[]{ "mean_eps" };

    }





}
