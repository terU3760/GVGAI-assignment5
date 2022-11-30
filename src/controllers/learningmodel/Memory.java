package controllers.learningmodel;

import core.game.StateObservation;
import ontology.Types;
import org.nd4j.linalg.api.ndarray.INDArray;


import java.util.concurrent.ThreadLocalRandom;

public class Memory {

    private INDArray stateObservation;
    private Types.ACTIONS action;

    public Memory(INDArray stateObservation, Types.ACTIONS action, double reward, boolean terminal, boolean training) {
        this.stateObservation = stateObservation;
        this.action = action;
        this.reward = reward;
        this.terminal = terminal;
        this.training = training;
    }

    public INDArray getStateObservation() {
        return stateObservation;
    }

    public void setStateObservation(INDArray stateObservation) {
        this.stateObservation = stateObservation;
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

    public boolean isTraining() {
        return training;
    }

    public void setTraining(boolean training) {
        this.training = training;
    }

    private double reward;
    private boolean terminal;
    private boolean training;





}
