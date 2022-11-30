package core.content;

import core.game.Game;
import org.apache.commons.lang3.SerializationUtils;

import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 16/10/13
 * Time: 14:08
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public abstract class Content implements Cloneable
{
    /**
     * Original line with the content, in VGDL format.
     */
    public String line;

    /**
     * Main definition of the content.
     * It is always the first word of each line in VGDL
     */
    public String identifier;

    /**
     * List of parameters of this content (key => value).
     * List of all pairs of the form key=value on the line.
     */
    public HashMap<String, String> parameters;

    /**
     * Indicates if this content is definition (i.e., includes character ">" in VGDL).
     */
    public boolean is_definition;

    /**
     * Returns the original line of the content.
     * @return original line, in VGDL format.
     */
    public String toString() {  return line; }





    public Content clone()
    {

        try {
            Content clonedContent = getClass().getDeclaredConstructor().newInstance();

            clonedContent.line = this.line;
            clonedContent.identifier = this.identifier;
            clonedContent.parameters = SerializationUtils.clone(this.parameters);
            clonedContent.is_definition = this.is_definition;

            return clonedContent;

        } catch (NoSuchMethodException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (InstantiationException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return null;


    }





}
