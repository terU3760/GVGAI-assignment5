package ontology.effects.unary;

import core.VGDLRegistry;
import core.VGDLSprite;
import core.content.InteractionContent;
import core.game.Game;
import ontology.effects.Effect;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 04/11/13
 * Time: 15:57
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class KillIfHasLess extends Effect
{
    public String resource;
    public int resourceId;
    public int limit;

    public KillIfHasLess(InteractionContent cnt)
    {
        is_kill_effect = true;
        resourceId = -1;
        this.parseParameters(cnt);
        resourceId = VGDLRegistry.GetInstance().getRegisteredSpriteValue(resource);
    }

    @Override
    public void execute(VGDLSprite sprite1, VGDLSprite sprite2, Game game)
    {
        applyScore = false;
        if(sprite1.getAmountResource(resourceId) <= limit)
        {
            game.killSprite(sprite1);
            applyScore = true;
        }
    }





    public KillIfHasLess() {}

    @Override
    public KillIfHasLess clone() throws CloneNotSupportedException
    {

        KillIfHasLess clonedKillIfHasLess = (KillIfHasLess)super.clone();
        clonedKillIfHasLess.resource = this.resource;
        clonedKillIfHasLess.resourceId = this.resourceId;
        clonedKillIfHasLess.limit = this.limit;

        return clonedKillIfHasLess;


    }





}
