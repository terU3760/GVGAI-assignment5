package controllers.learningmodel;

import controllers.Heuristics.StateHeuristic;
import controllers.Heuristics.WinScoreHeuristic;
import core.SpriteGroup;
import core.VGDLSprite;
import core.game.ForwardModel;
import core.game.StateObservation;
import core.player.AbstractPlayer;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;

import ontology.Types;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.MathUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tools.ElapsedCpuTimer;
import weka.classifiers.Classifier;
import weka.core.Instances;

import miscellaneous.util.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import core.game.Game;


import org.nd4j.linalg.api.shape.Shape;


public class DQNVanillaAgent extends AbstractPlayer {

    protected Classifier m_model;
    protected Random m_rnd;
    private static int SIMULATION_DEPTH = 20;
    private final HashMap<Integer, Types.ACTIONS> action_mapping;
    public Policy mPolicy;
    protected Policy testPolicy;
    protected int N_ACTIONS;
    protected static Instances m_dataset;
    protected int m_maxPoolSize = 1000;
    protected double m_gamma = 0.99;





    private static long seed = 123L;

    private int mInputChannels = 4;
    private int mInputWidth = 84 * 3;
    private int mInputHeight = 84 * 3;



    protected Processor processor;
    protected MultiLayerNetwork model;

    protected MultiLayerNetwork targetModel;
    protected MultiLayerNetwork trainableModel;





    private INDArray mRecentObservation;
    private int mRecentAction;





    public Game game;
    public LinkedList<INDArray> statesAsINDArray;
    public boolean training;
    public INDArray recentObservation;





    private LinkedList<Memory> memoryOfPast = new LinkedList<Memory>();
    private int memoryInterval;
    private int memoryLimit;
    private int memoryWindowLength;
    private boolean memoryIgnoreEpisodeBoundaries;





    private int backwardBatchSize;
    private int trainInterval;





    private int targetModelUpdate;





    private String dqnVanillaModelSavePath;
    private int modelSaveInterval;





    private int episode = 0;
    private int episodeStep = 0;
    private int nbMaxEpisodeSteps;





    private int currentStep;
    private int nbStepsWarmup;





    private LinkedList<Double> rewardsOfEpisode = new LinkedList<Double>();
    private LinkedList<Double> lossesOfEpisode = new LinkedList<Double>();
    private LinkedList<Double> meanQsOfEpisode = new LinkedList<Double>();
    private LinkedList<Double> epsilonsOfEpisode = new LinkedList<Double>();





    private boolean outputEveryStep;





    private double optimizerAdamLearningRate;





    public int accumulatedGameTicks = 0;





    public DQNVanillaAgent(StateObservation stateObs, ElapsedCpuTimer elapsedTimer, int memoryInterval, int memoryLimit, int memoryWindowLength, boolean memoryIgnoreEpisodeBoundaries, double gamma, int targetModelUpdate, String dqnVanillaModelSavePath, int modelSaveInterval, int nbMaxEpisodeSteps, int nbStepsWarmup, int trainInterval, int backwardBatchSize, double optimizerAdamLearningRate, int currentStep, boolean outputEveryStep) {
        this.m_rnd = new Random();

        // convert numbers to actions
        this.action_mapping = new HashMap<Integer, Types.ACTIONS>();
        int i = 0;
        for (Types.ACTIONS action : stateObs.getAvailableActions()) {
            this.action_mapping.put(i, action);
            i++;
        }

        this.N_ACTIONS = stateObs.getAvailableActions().size();

        this.mPolicy = new LinearAnnealedPolicy(new EpsGreedyQPolicy(), 1.0, 0.1, 0.05, 1250000);
        this.mPolicy.setAgent(this);
        this.testPolicy = new GreedyQPolicy();
        this.testPolicy.setAgent(this);





        this.memoryInterval = memoryInterval;
        this.memoryLimit = memoryLimit;
        this.memoryWindowLength = memoryWindowLength;
        this.memoryIgnoreEpisodeBoundaries = memoryIgnoreEpisodeBoundaries;
        this.m_gamma = gamma;





        this.dqnVanillaModelSavePath = dqnVanillaModelSavePath;
        this.modelSaveInterval = modelSaveInterval;




        this.nbMaxEpisodeSteps = nbMaxEpisodeSteps;





        this.currentStep = currentStep;
        this.nbStepsWarmup = nbStepsWarmup;
        this.trainInterval = trainInterval;
        this.backwardBatchSize = backwardBatchSize;





        this.game = stateObs.getModel().game;
        this.processor = new AtariProcessor();





        this.optimizerAdamLearningRate = optimizerAdamLearningRate;




        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new AdaDelta())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(optimizerAdamLearningRate))
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(8,8).stride(4,4).padding(0,0).activation(Activation.RELU)
                        .nIn(mInputChannels).nOut(32).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(4,4).stride(2,2).padding(0,0).activation(Activation.RELU)
                        .nOut(64).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(0,0).activation(Activation.RELU)
                        .nOut(64).build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(this.N_ACTIONS)
                        .activation(Activation.IDENTITY)
                        .build())
                .setInputType(InputType.convolutional(mInputHeight, mInputWidth, mInputChannels))
                .build();

        this.model = new MultiLayerNetwork(conf);
        this.model.init();


        this.targetModel = this.model.clone();
        this.trainableModel = this.model.clone();
        this.targetModelUpdate = targetModelUpdate;





        this.outputEveryStep = outputEveryStep;





    }





    public void setStateObs(StateObservation stateObs)
    {

        // convert numbers to actions
        this.action_mapping.clear();;
        int i = 0;
        for (Types.ACTIONS action : stateObs.getAvailableActions()) {
            this.action_mapping.put(i, action);
            i++;
        }

        this.N_ACTIONS = stateObs.getAvailableActions().size();

        this.game = stateObs.getModel().game;

    }





    public int actionMappingGetKey(Types.ACTIONS action)
    {

        for(Map.Entry<Integer,Types.ACTIONS> e: this.action_mapping.entrySet())
        {
            if(e.getValue() == action)
            {

                return e.getKey().intValue();

            }
        }
        return -1;

    }






    /**
     *
     * Learning based agent.
     *
     * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {





        learnPolicy(stateObs, SIMULATION_DEPTH, new WinScoreHeuristic(stateObs));
        Types.ACTIONS bestAction = null;
        try {
            int actionNum = forward();
            bestAction = action_mapping.get(actionNum);
        } catch(Exception exc) {
            exc.printStackTrace();
        }
        return bestAction;





    }





    private void learnPolicy(StateObservation stateObs, int actionRepitition, StateHeuristic heuristic) {





        fit(stateObs, heuristic, actionRepitition, false);





    }





    public double[][][] process_state_batch(int[][][] batch)
    {

        return processor.process_state_batch(batch);

    }


    public INDArray processStateBatch(INDArray batch)
    {
        return processor.processStateBatch(batch);
    }


    public double[][] compute_batch_q_values(int[][][] state_batch)
    {


        double[][][] batch = process_state_batch(state_batch);

        INDArray input_batch = Nd4j.create(batch);
        INDArray predicted = model.output( input_batch , false );

        assert(predicted.rows() == state_batch.length);
        assert(predicted.columns() == N_ACTIONS);

        double[][] q_values = new double[predicted.rows()][predicted.columns()];
        for( int i = 0 ; i < predicted.rows(); i++ )
        {

            for( int j = 0; j < predicted.columns(); j++ )
            {

                q_values[i][j] = predicted.getDouble( i , j );

            }

        }


        return q_values;


    }


    public INDArray computeBatchQValues(INDArray state_batch)
    {


        INDArray batch = processStateBatch(state_batch);

        INDArray predicted = model.output( batch , false );

        assert(predicted.rows() == state_batch.size(0));
        assert(predicted.columns() == N_ACTIONS);


        return predicted;


    }





    public double[] compute_q_values(int[][] state)
    {

        int[][][] temp_state = new int[1][state.length][state[0].length];
        double[][] q_values = compute_batch_q_values(temp_state);
        assert( q_values.length == 1 );
        assert( q_values[0].length == N_ACTIONS );
        return q_values[0];



    }



    public INDArray computeQValues(INDArray state)
    {
        state = state.reshape(new long[]{ 1 , state.size(0) , state.size(1) });
        INDArray q_values = computeBatchQValues( state );
        assert(q_values.size(0) == 1);
        assert(q_values.size(1) == N_ACTIONS);
        return q_values;

    }





    public INDArray getImageOutputOfCurrentState() {
        return  getImageOutputOfAGame(this.game);
    }





    public INDArray getImageOutputOfAGame(Game game)
    {


        BufferedImage gameScreenImage = new BufferedImage(game.getScreenSize().width, game.getScreenSize().height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = (Graphics2D) gameScreenImage.getGraphics();

        SpriteGroup[] spriteGroupsGame = game.getSpriteGroups();
        //this.spriteGroups = spriteGroupsGame;
        SpriteGroup[] spriteGroups = new SpriteGroup[spriteGroupsGame.length];
        for(int i = 0; i < spriteGroups.length; ++i)
        {
            spriteGroups[i] = new SpriteGroup(spriteGroupsGame[i].getItype());
            spriteGroups[i].copyAllSprites(spriteGroupsGame[i].getSprites().values());
        }

        int[] gameSpriteOrder = game.getSpriteOrder();

        if(spriteGroups != null) for(Integer spriteTypeInt : gameSpriteOrder)
        {
            if(spriteGroups[spriteTypeInt] != null) {
                ConcurrentHashMap<Integer, VGDLSprite> cMap =spriteGroups[spriteTypeInt].getSprites();
                Set<Integer> s = cMap.keySet();
                for (Integer key : s) {
                    VGDLSprite sp = cMap.get(key);
                    if (sp != null)
                        sp.draw(graphics, game);
                }
            }
        }


        // Convert color image to gray scale image
        BufferedImage grayGameScreenImage = new BufferedImage(game.getScreenSize().width, game.getScreenSize().height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = grayGameScreenImage.getGraphics();
        g.drawImage(gameScreenImage, 0, 0, null);
        g.dispose();





        // resize the image
        Image tmp = grayGameScreenImage.getScaledInstance(mInputWidth, mInputHeight, Image.SCALE_SMOOTH);
        BufferedImage resizedGrayGameScreenImage = new BufferedImage(mInputWidth, mInputHeight, grayGameScreenImage.getType());

        Graphics2D g2d = resizedGrayGameScreenImage.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();


        // convert the resized gray scale image to ndarray
        int channels = 1;
        ImageLoader loader = new ImageLoader(mInputWidth, mInputHeight, 1);
        INDArray inputImageNDArray = Utility.asMatrix(resizedGrayGameScreenImage, loader, channels);

        return inputImageNDArray;


    }





    INDArray getRecentStates(INDArray currentStateAsINDArray)
    {


        INDArray[] recentStatesINDArray = new INDArray[this.memoryWindowLength];


        recentStatesINDArray[this.memoryWindowLength - 1] = currentStateAsINDArray;

        Iterator iter = this.memoryOfPast.descendingIterator();
        int i = this.memoryWindowLength - 2;

        while(iter.hasNext())
        {

            Memory m = (Memory)(iter.next());
            if(!(m.isTerminal())) {
                recentStatesINDArray[i--] = m.getStateObservation();


                if( i < 0 )
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        while( i >= 0 )
        {
            recentStatesINDArray[i] = recentStatesINDArray[i+1].dup();
            i--;
        }

        return Nd4j.stack(0, recentStatesINDArray );


    }





    public int forward()
    {





        INDArray currentStateAsINDArray = getImageOutputOfCurrentState();



        INDArray recentStateAsINDArray = getRecentStates(currentStateAsINDArray);



        long[] recentStateAsINDArrayShape = recentStateAsINDArray.shape();
        assert( recentStateAsINDArrayShape.length == 3 );
        recentStateAsINDArray = recentStateAsINDArray.reshape( new long[]{ 1, recentStateAsINDArrayShape[0], recentStateAsINDArrayShape[1], recentStateAsINDArrayShape[2] } );


        INDArray qValues = computeBatchQValues(recentStateAsINDArray);


        int action;
        if(this.training)
        {
            action = this.mPolicy.selectAction(qValues);
        }
        else
        {
            action = this.testPolicy.selectAction(qValues);
        }

        // Book-keeping.
        this.mRecentObservation = currentStateAsINDArray;
        this.mRecentAction = action;

        return action;



    }



    public int forward(Game game)
    {


        INDArray currentStateAsINDArray = getImageOutputOfAGame(game);


        INDArray recentStateAsINDArray = getRecentStates(currentStateAsINDArray);



        long[] recentStateAsINDArrayShape = recentStateAsINDArray.shape();
        assert( recentStateAsINDArrayShape.length == 2 );
        recentStateAsINDArray = recentStateAsINDArray.reshape( new long[]{ 1, recentStateAsINDArrayShape[0], recentStateAsINDArrayShape[1], recentStateAsINDArrayShape[2] } );


        INDArray qValues = computeBatchQValues(recentStateAsINDArray);


        int action;
        if(this.training)
        {
            action = this.mPolicy.selectAction(qValues);
        }
        else
        {
            action = this.testPolicy.selectAction(qValues);
        }


        return action;


    }





    public static int[] sampleBatchIndexes(int low, int high, int size)
    {


        assert( ( high - low ) >= size );
        int[] batchIds = new int[size];
        for(int i = 0; i < size; i++)
        {
            batchIds[i] = ThreadLocalRandom.current().nextInt(low, high);
        }
        return batchIds;


    }





    public LinkedList<Experience> sample(int batchSize, int[] batchIdxs)
    {


        assert( this.memoryOfPast.size() >= this.memoryWindowLength + 2);


        if(batchIdxs == null) {
            batchIdxs = sampleBatchIndexes(this.memoryWindowLength, this.memoryOfPast.size() - 1, batchSize);
        }
        for(int i = 0; i < batchIdxs.length; i++)
        {
            batchIdxs[i]+=1;
        }
        assert( Collections.min( Arrays.asList( Arrays.stream( batchIdxs ).boxed().toArray( Integer[]::new ) ) ) >= this.memoryWindowLength + 1 );
        assert( Collections.max( Arrays.asList( Arrays.stream( batchIdxs ).boxed().toArray( Integer[]::new ) ) ) < this.memoryOfPast.size() );
        assert( batchIdxs.length == this.backwardBatchSize);


        LinkedList<Experience> experiences = new LinkedList<Experience>();

        for(int idx: batchIdxs)
        {
            boolean terminal0 = this.memoryOfPast.get(idx - 2).isTerminal();
            int newIdx = idx;
            while(terminal0) {
                newIdx = sampleBatchIndexes(this.memoryWindowLength + 1, this.memoryOfPast.size(), 1)[0];
                terminal0 = this.memoryOfPast.get(newIdx - 2).isTerminal();
            }
            assert( this.memoryWindowLength <= newIdx );
            assert( newIdx < this.memoryOfPast.size() );


            LinkedList<INDArray> state0 = new LinkedList<INDArray>();
            state0.add( this.memoryOfPast.get(newIdx - 1).getStateObservation() );
            for(int offset = 0; offset < this.memoryWindowLength - 1 ; offset++)
            {
                int currentIdx = newIdx - 2 - offset;
                assert(currentIdx >= 1);
                boolean currentTerminal = this.memoryOfPast.get(newIdx - 1).isTerminal();

                if(currentTerminal && (!(this.memoryIgnoreEpisodeBoundaries)))
                {
                    break;
                }

                state0.addFirst( this.memoryOfPast.get(currentIdx).getStateObservation() );
            }
            while(state0.size() < this.memoryWindowLength)
                state0.addFirst(Nd4j.zerosLike(state0.getFirst()));


            Types.ACTIONS action = this.memoryOfPast.get(newIdx - 1).getAction();
            double reward = this.memoryOfPast.get(newIdx - 1).getReward();
            boolean terminal = this.memoryOfPast.get(newIdx - 1).isTerminal();

            LinkedList<INDArray> state1=new LinkedList<INDArray>();
            Iterator it = state0.iterator();
            it.next();
            while(it.hasNext())
            {
                state1.addLast( ( (INDArray)(it.next()) ).dup() );
            }
            state1.addLast( this.memoryOfPast.get(newIdx).getStateObservation() );

            assert( state0.size() == this.memoryWindowLength );
            assert( state1.size() == state0.size() );


            experiences.addLast( new Experience( state0.toArray(new INDArray[state0.size()]), state1.toArray(new INDArray[state1.size()]), action, reward, terminal ) );


        }


        assert( experiences.size() == this.backwardBatchSize);
        return experiences;

    }





    public void backward(double reward, boolean terminal)
    {

        if(this.game.getGameTick() % this.memoryInterval == 0) {





            this.memoryOfPast.addLast(new Memory(this.mRecentObservation.reshape(Shape.squeeze(this.mRecentObservation.shape())), this.action_mapping.get(this.mRecentAction), reward, terminal, true));





            while(this.memoryOfPast.size() < this.backwardBatchSize)
            {
                this.memoryOfPast.addLast(new Memory(this.mRecentObservation.reshape(Shape.squeeze(this.mRecentObservation.shape())), this.action_mapping.get(this.mRecentAction), reward, terminal, true));
            }





            while(this.memoryOfPast.size() > this.memoryLimit) {
                this.memoryOfPast.removeFirst();
            }
        }

        if(!this.training) {
            return;
        }





        if((this.currentStep >= this.nbStepsWarmup)&&(this.currentStep % this.trainInterval == 0)) {





            LinkedList<Experience> experiences = sample(this.backwardBatchSize,null);
            LinkedList<INDArray> state0Batch = new LinkedList<INDArray>();
            LinkedList<Double> rewardBatch = new LinkedList<Double>();
            LinkedList<Types.ACTIONS> actionBatch = new LinkedList<Types.ACTIONS>();
            LinkedList<Double> terminal1Batch = new LinkedList<Double>();
            LinkedList<INDArray> state1Batch = new LinkedList<INDArray>();





            for(Experience experience : experiences)
            {





                state0Batch.addLast(Nd4j.stack(0,experience.getState0()));
                state1Batch.addLast(Nd4j.stack(0,experience.getState1()));
                rewardBatch.addLast(experience.getReward());
                actionBatch.addLast(experience.getAction());
                terminal1Batch.addLast( experience.isTerminal() ? 0.0: 1.0);

            }





            INDArray state0BatchProcessed = this.processStateBatch(Nd4j.stack(0, state0Batch.toArray(new INDArray[state0Batch.size()])));
            INDArray state1BatchProcessed = this.processStateBatch(Nd4j.stack(0, state0Batch.toArray(new INDArray[state1Batch.size()])));
            INDArray terminal1BatchINDArray = Nd4j.create(terminal1Batch);
            INDArray rewardBatchINDArray = Nd4j.create(rewardBatch);
            assert ((rewardBatchINDArray.shape().length == 1) && (rewardBatchINDArray.shape()[0] == this.backwardBatchSize));
            assert ((terminal1BatchINDArray.shape().length == rewardBatchINDArray.shape().length) && (terminal1BatchINDArray.shape()[0] == rewardBatchINDArray.shape()[0]));
            assert (actionBatch.size() == rewardBatchINDArray.shape()[0]);



            INDArray targetQValues = targetModel.output(state1BatchProcessed);
            assert ((targetQValues.shape().length == 2) && (targetQValues.shape()[0] == this.backwardBatchSize) && (targetQValues.shape()[1] == this.N_ACTIONS));
            INDArray qBatch = Nd4j.max(targetQValues, 1).ravel();
            assert ((qBatch.shape().length == 1) && (qBatch.shape()[0] == this.backwardBatchSize));


            INDArray targets = Nd4j.zeros(DataType.DOUBLE, this.backwardBatchSize, this.N_ACTIONS);
            INDArray dummyTargets = Nd4j.zeros(DataType.DOUBLE, this.backwardBatchSize);
            INDArray masks = Nd4j.zeros(DataType.DOUBLE, this.backwardBatchSize, this.N_ACTIONS);


            INDArray discountedRewardBatch = qBatch.mul(this.m_gamma);
            discountedRewardBatch.muli(terminal1BatchINDArray);
            assert ((discountedRewardBatch.shape().length == rewardBatchINDArray.shape().length) && (discountedRewardBatch.shape()[0] == rewardBatchINDArray.shape()[0]));
            INDArray Rs = rewardBatchINDArray.add(discountedRewardBatch);


            ListIterator<Types.ACTIONS> it = actionBatch.listIterator();
            while (it.hasNext()) {
                int index = it.nextIndex();
                Types.ACTIONS action = it.next();
                double R = Rs.getDouble(index);
                targets.putScalar(new int[]{index, actionMappingGetKey(action)}, R);
                dummyTargets.putScalar(new int[]{index}, R);
                masks.putScalar(new int[]{index, actionMappingGetKey(action)}, 1.0);
            }





                trainableModel.setLabels(targets);
                trainableModel.setInput(state0BatchProcessed);
                trainableModel.setLayerMaskArrays(null, masks);





                trainableModel.feedForward(true, false);





                trainableModel.computeGradientAndScore();
                double loss = trainableModel.score();





            Gradient gradient = trainableModel.getGradient();
            trainableModel.getUpdater().update(trainableModel, gradient, this.game.getGameTick(), currentStep, backwardBatchSize, LayerWorkspaceMgr.noWorkspaces());
            INDArray updateVector = gradient.gradient();
            trainableModel.params().subi(updateVector);



            if ((this.targetModelUpdate >= 1) && ((this.game.getGameTick() % this.targetModelUpdate) == 0)) {
                this.targetModel.setParams(this.trainableModel.params());
            }





            INDArray qValues = trainableModel.output(state0BatchProcessed);
            INDArray meanQOfStep = qValues.max( -1 );
            meanQOfStep = meanQOfStep.mean( 0 );
            double meanQOfStepValue = meanQOfStep.getDouble(0);
            this.meanQsOfEpisode.addLast( meanQOfStepValue );
            this.rewardsOfEpisode.addLast( reward );
            this.lossesOfEpisode.addLast( loss );
            double epsOfStep = ((LinearAnnealedPolicy)this.mPolicy).metrics();
            this.epsilonsOfEpisode.addLast( epsOfStep );





        }





        if(this.game.getGameTick() % this.modelSaveInterval == 0)
        {
            try {
                this.trainableModel.save(new File(this.dqnVanillaModelSavePath));
            } catch(IOException e)
            {
                e.printStackTrace();
                System.out.println("Can't save trained model!!!");
            }
        }





    }





    public void fit(StateObservation stateObs, StateHeuristic heuristic,int actionRepitition, boolean outputEveryStep)
    {



        this.training = true;

        stateObs = stateObs.copy();


        int actionNum = forward();





        double score_before = heuristic.evaluateState(stateObs);

        // simulate
        Types.ACTIONS action = action_mapping.get(actionNum);


        boolean done = false;
        double reward = 0;

        Game clonedGame = this.game.cloneExceptForwardModelAndRandomAndPathFinder();
        clonedGame.initForwardModel();
        ForwardModel forwardModel = clonedGame.getFwdModel();
        forwardModel.setFwdModel( forwardModel );
        for(int i = 0; i < actionRepitition; i++) {


            forwardModel.advance( action );
            StateObservation newStateObs = forwardModel.getObservation();

            double score_after = heuristic.evaluateState(newStateObs);
            stateObs = newStateObs;

            double delta_score = score_after - score_before;
            reward += delta_score;
            done = forwardModel.isGameOver();


            if(done)
                break;


            actionNum = forward(forwardModel.game);
            action = action_mapping.get(actionNum);
            score_before = heuristic.evaluateState(stateObs);





        }





        backward(reward, done);





        if(this.outputEveryStep) {
            System.out.println("Episode " + this.episode + ", step " + this.episodeStep + ", reward: " + reward + ", loss: " + this.lossesOfEpisode.getLast() + ", mean q: " + this.meanQsOfEpisode.getLast() + ", epsilon: " + this.epsilonsOfEpisode.getLast());
        }





        if (done || (this.episodeStep >= this.nbMaxEpisodeSteps - 1))
        {

            double episodeReward = Utility.sum( this.rewardsOfEpisode );
            double meanReward = episodeReward / this.rewardsOfEpisode.size();
            double episodeLoss = Utility.sum( this.lossesOfEpisode );
            double meanLoss = episodeLoss / this.lossesOfEpisode.size();
            double episodeMeanQs = Utility.sum( this.meanQsOfEpisode );
            double meanQ = episodeMeanQs / this.meanQsOfEpisode.size();
            double episodeEps = Utility.sum( this.epsilonsOfEpisode );
            double meanEps = episodeEps / this.epsilonsOfEpisode.size();
            System.out.println( "Episode "+ this.episode + " ended, episode reward: " + episodeReward + ", mean reward: " + meanReward + ", loss: " + meanLoss + ", mean q: " + meanQ + ", mean eps: " + meanEps );


            this.rewardsOfEpisode.clear();
            this.lossesOfEpisode.clear();
            this.meanQsOfEpisode.clear();
            this.epsilonsOfEpisode.clear();





        }




        if(this.episodeStep > this.nbMaxEpisodeSteps - 1)
        {
            // Assure the current game ended
            this.game = clonedGame;
            this.game.disqualify();
            boolean fwdModelIsGameOver = this.game.getFwdModel().isGameOver();
            assert( fwdModelIsGameOver );
            boolean gameIsGameOver = this.game.isGameOver();
            assert( gameIsGameOver );


            System.gc();

        }
        if(done)
        {
            episode +=1;
            episodeStep = 0;


            forward();
            backward(0.0, false);



            // Assure the current game ended
            boolean fwdModelIsGameOver = forwardModel.isGameOver();
            assert( fwdModelIsGameOver );
            this.game.setEnded( forwardModel.isEnded());
            this.game.setWinner( forwardModel.getWinner() );


            System.gc();


        }
        else
        {
            episodeStep += 1;


            forwardModel = null;
            clonedGame = null;
            System.gc();


        }





    }





}
