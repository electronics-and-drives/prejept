package edlab.eda.prejept;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;

/**
 *  PreceptMinimal
 *  Minimal implementaion for loading and evaluating pytorch models.
 * */
public class PreceptMinimal 
{
    private Module module;

    /**
     * PreceptMinimal
     *  Minimal implementaion for loading and evaluating pytorch models.
     *  @param modePath Path to torchscript model.
     * */
    public PreceptMinimal(String modPath)
    {
        this.module = Module.load(modPath);
    }

    /**
     * predict
     *  Method for making prediction with the given torchscript model.
     *  @param data     Input data float array.
     *  @param shape    Shape of the input tensor.
     *  @return         Prediction as float array.
     * */
    public float[] predict(float[] data, long[] shape)
    {
        Tensor input  = Tensor.fromBlob(data, shape);
        IValue result = this.module.forward(IValue.from(input)); 
        Tensor output = result.toTensor();

        return output.getDataAsFloatArray();
    }

    /**
     * predict
     *  Overloaded for convenience, if only one sample is given.
     *  @param data     Input data float array.
     *  @return         Prediction as float array.
     * */
    public float[] predict(float[] data)
    {
        return this.predict(data, new long[] {1, data.length});
    }
}
