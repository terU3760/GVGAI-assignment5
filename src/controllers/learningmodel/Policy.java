package controllers.learningmodel;

import core.player.AbstractPlayer;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Policy {


    public AbstractPlayer getAgent() {
        return agent;
    }

    public void setAgent(AbstractPlayer agent) {
        this.agent = agent;
    }

    public abstract int selectAction(INDArray qValues);

    private AbstractPlayer agent;


}
