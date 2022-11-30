package ontology.effects.unary;

import java.util.ArrayList;

import core.VGDLRegistry;
import core.VGDLSprite;
import core.content.InteractionContent;
import core.game.Game;
import ontology.effects.Effect;

/**
 * Created by Diego on 18/02/14.
 */
public class SpawnIfHasMore  extends Effect {

    public int spend;
    public String resource;
    public int resourceId;
    public int limit;
    public String stype;
    public int itype;

    public SpawnIfHasMore(InteractionContent cnt)
    {
        resourceId = -1;
        spend = 0;
        this.parseParameters(cnt);
        resourceId = VGDLRegistry.GetInstance().getRegisteredSpriteValue(resource);
        itype = VGDLRegistry.GetInstance().getRegisteredSpriteValue(stype);
    }

    @Override
    public void execute(VGDLSprite sprite1, VGDLSprite sprite2, Game game)
    {
        applyScore = false;

        if(game.getRandomGenerator().nextDouble() >= prob) return;

        if(sprite1.getAmountResource(resourceId) >= limit)
        {
            game.addSprite(itype, sprite1.getPosition());
            applyScore = true;

            sprite1.modifyResource(resourceId, -spend); //0 by default.
        }
    }
    
    @Override
    public ArrayList<String> getEffectSprites(){
    	ArrayList<String> result = new ArrayList<String>();
    	if(stype!=null) result.add(stype);
    	
    	return result;
    }





    public SpawnIfHasMore() {}

    @Override
    public SpawnIfHasMore clone() throws CloneNotSupportedException
    {

        SpawnIfHasMore clonedSpawnIfHasMore = (SpawnIfHasMore)super.clone();
        clonedSpawnIfHasMore.spend = this.spend;
        clonedSpawnIfHasMore.resource = this.resource;
        clonedSpawnIfHasMore.resourceId = this.resourceId;
        clonedSpawnIfHasMore.limit = this.limit;
        clonedSpawnIfHasMore.stype = this.stype;
        clonedSpawnIfHasMore.itype = this .itype;

        return clonedSpawnIfHasMore;


    }





}
