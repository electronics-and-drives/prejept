package edlab.eda.prejept;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;

import edlab.eda.prejept.PreceptModule;

public class Demo 
{
    public static void main(String[] args) 
    {
        String modPath = "/home/ynk/workspace/PySpectre/char/models/xh035-nmos-20210727-085807/xh035-nmos-model.pt";
        String cfgPath = "/home/ynk/workspace/PySpectre/char/models/xh035-nmos-20210727-085807/xh035-nmos-model.yml";

        //Module mod = Module.load(modPath);

        //Tensor data = Tensor.fromBlob( new float[] {0.5f, 0.5f, 0.5f, 0.5f}
        //                             , new long[] {1, 4} );

        //IValue result = mod.forward(IValue.from(data)); 

        //Tensor output = result.toTensor();

        //System.out.println("shape: " + Arrays.toString(output.shape()));
        //System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));

        PreceptModule module = new PreceptModule(modPath, cfgPath);
        double[] result = module.predict(new double[] {0.5d, 0.5d, 0.5d, 0.5d});

        System.out.println(Arrays.toString(result));

        //PreceptMinimal module = new PreceptMinimal(modPath);
        //float[] prediction = module.predict(new float[] {0.5f, 0.5f, 0.5f, 0.5f});

        System.exit(0);
    }
}
