package ontology.effects;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;

import core.VGDLFactory;
import core.VGDLSprite;
import core.content.Content;
import core.content.InteractionContent;
import core.game.Game;
import org.apache.commons.lang3.SerializationUtils;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 23/10/13
 * Time: 15:20
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Effect implements Cloneable{

    //Indicates if this effect kills any sprite
    public boolean is_kill_effect = false;

    //Indicates if this effect has some random element.
    public boolean is_stochastic = false;

    //Change of the score this effect makes.
    public int scoreChange = 0;

    //Probabilty for stochastic effects.
    public double prob = 1;

    //Indicates if this effects changes the score.
    public boolean applyScore = true;

    //Indicates the number of repetitions of this effect. This affects how many times this
    // effect is taken into account at each step. This is useful for chain effects (i.e. pushing
    // boxes in a chain - thecitadel, enemycitadel).
    public int repeat = 1;

    /**
     * 'Unique' hashcode for this effect
     */
    public long hashCode;


    /**
     * Executes the effect
     *
     * @param sprite1 first sprite of the collision
     * @param sprite2 second sprite of the collision
     * @param game    reference to the game object with the current state.
     */
    public void execute(VGDLSprite sprite1, VGDLSprite sprite2, Game game){};

    public void setStochastic() {
        if (prob > 0 && prob < 1)
            is_stochastic = true;
    }

    public void parseParameters(InteractionContent content) {
        //parameters from the object.
        VGDLFactory.GetInstance().parseParameters(content, this);
        hashCode = content.hashCode;
    }

    
    public ArrayList<String> getEffectSprites(){
    	return new ArrayList<String>();
    }





    public Effect() {}

    @Override
    public Effect clone() throws CloneNotSupportedException
    {

        Effect clonedEffect = (Effect)super.clone();

        clonedEffect.is_kill_effect = this.is_kill_effect;
        clonedEffect.is_stochastic = this.is_stochastic;
        clonedEffect.scoreChange = this.scoreChange;
        clonedEffect.prob = this.prob;
        clonedEffect.applyScore = this.applyScore;
        clonedEffect.repeat = this.repeat;
        clonedEffect.hashCode = this.hashCode;


        return clonedEffect;


    }





}
