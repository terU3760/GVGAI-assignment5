package ontology.effects.unary;

import core.VGDLSprite;
import core.content.InteractionContent;
import core.game.Game;
import ontology.effects.Effect;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 23/10/13
 * Time: 15:21
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class KillSprite extends Effect {

    public KillSprite(InteractionContent cnt)
    {
        is_kill_effect = true;
        this.parseParameters(cnt);
    }

    @Override
    public void execute(VGDLSprite sprite1, VGDLSprite sprite2, Game game) {
        game.killSprite(sprite1);
    }





    @Override
    public KillSprite clone() throws CloneNotSupportedException
    {

        Effect clonedKillSprite = super.clone();

        return (KillSprite)clonedKillSprite;


    }





}
