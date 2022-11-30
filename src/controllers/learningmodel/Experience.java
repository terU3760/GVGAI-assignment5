package controllers.learningmodel;

import ontology.Types;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedList;

public class Experience {


    private INDArray[] state0;
    private INDArray[] state1;

    public Experience(INDArray[] state0, INDArray[] state1, Types.ACTIONS action, double reward, boolean terminal) {
        this.state0 = state0;
        this.state1 = state1;
        this.action = action;
        this.reward = reward;
        this.terminal = terminal;
    }

    public INDArray[] getState0() {
        return state0;
    }

    public void setState0(INDArray[] state0) {
        this.state0 = state0;
    }

    public INDArray[] getState1() {
        return state1;
    }

    public void setState1(INDArray[] state1) {
        this.state1 = state1;
    }

    public Types.ACTIONS getAction() {
        return action;
    }

    public void setAction(Types.ACTIONS action) {
        this.action = action;
    }

    public double getReward() {
        return reward;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }

    public boolean isTerminal() {
        return terminal;
    }

    public void setTerminal(boolean terminal) {
        this.terminal = terminal;
    }

    private Types.ACTIONS action;
    private double reward;
    private boolean terminal;


}
