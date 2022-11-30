import controllers.learningmodel.DQNVanillaAgent;
import controllers.learningmodel.LinearAnnealedPolicy;
import core.ArcadeMachineContinuousLearning;
import core.competition.CompetitionParameters;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 04/10/13
 * Time: 16:29
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Assignment5
{

    public static void main(String[] args)
    {
        //Reinforcement learning controllers:
        String rlController = "controllers.learningmodel.DQNVanillaAgent";

        //Only pacman game
        String game = "examples/gridphysics/pacman.txt";
        String level = "examples/gridphysics/pacman_lvl";

        //Other settings
        boolean visuals = true;
        int seed = new Random().nextInt();

        //Game and level to play

        int memoryInterval = 1;
        int memoryLimit = 1000000;
        int memoryWindowLength = 4;
        boolean memoryIgnoreEpisodeBoundaries = false;
        double gamma = 0.99;
        int targetModelUpdate = 50;





        String dqnVanillaModelSavePath = "models/DQNVanillaForPacman.h5";
        int modelSaveInterval = 50;





        int nbMaxEpisodeSteps = 20000;





        int nbSteps = 100000;
        int nbStepsWarmup = 50;
        int trainInterval = 4;
        int backwardBatchSize = 32;





        double optimizerAdamLearningRate = 0.00025;





        boolean outputEveryStep = true;





        // Monte-Carlo RL training
        CompetitionParameters.ACTION_TIME = 1000000;
        //ArcadeMachine.runOneGame(game, level, visuals, rlController, null, seed, false);
        //String level2 = gamesPath + games[gameIdx] + "_lvl" + 1 +".txt";
        for(int i=0; i<nbSteps; i++){
            String levelfile = level + "0.txt";
            ArcadeMachineContinuousLearning.runOneGameButCreatePlayerOnceOnly(game, levelfile, visuals, rlController, memoryInterval, memoryLimit, memoryWindowLength, memoryIgnoreEpisodeBoundaries, gamma, null, targetModelUpdate, dqnVanillaModelSavePath, modelSaveInterval, nbMaxEpisodeSteps, nbStepsWarmup, trainInterval, backwardBatchSize, optimizerAdamLearningRate, i, seed, outputEveryStep, false);
            nbStepsWarmup = 0;




            ((LinearAnnealedPolicy)(((DQNVanillaAgent)(ArcadeMachineContinuousLearning.m_playerOnceCreated)).mPolicy)).accumulatedGameTicks += ((DQNVanillaAgent)(ArcadeMachineContinuousLearning.m_playerOnceCreated)).game.getGameTick();





        }
        
    }
}
