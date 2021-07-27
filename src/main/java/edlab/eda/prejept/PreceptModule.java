package edlab.eda.prejept;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import org.yaml.snakeyaml.Yaml;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.exp;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.analysis.function.*;
import org.apache.commons.math3.util.ArithmeticUtils.*;
/**
 * PreceptModule
 *  Class for loading torchscript models generated with `precept`
 * */
public class PreceptModule 
{   // Path to torchscript model
    private String modelPath;
    // Path to yaml config
    private String configPath;  
    
    // The torchscript module
    private Module module;

    // Number of neurons at the input, this is how long the input array
    // needs to be.
    private int numX;
    // Number of neurons at the output, this is how long the response
    // (prediction) array will be.
    private int numY;

    // Minima and Maxima of training data, this is used to (re-)scale the
    // inputs and outptus
    private RealVector maxX;
    private RealVector minX;
    private RealVector maxY;
    private RealVector minY;
   
    // Type of Transformation
    private String trafoType;

    // Lambdas used for Transformation
    private RealVector lambdaX;
    private RealVector lambdaY;

    // Transformation Mask
    private List<String> maskX;
    private List<String> maskY;
    private List<Integer> maskXidx;
    private List<Integer> maskYidx;

    // Name of Parameters
    private List<String> paramsX;
    private List<String> paramsY;
    
    /**
     * readYAMLcfg
     *  Public Static yaml reader utility. Used by the Constructor to read a
     *  config.
     *  @param fileName     Path to model.y[a]ml as String
     *  @return             YAML config as Map<String,Object>
     * */
    public static Map<String, Object> readYAMLcfg(String fileName)
    {
        String content;
        Yaml yaml;

        try
            {content = new String(Files.readAllBytes(Paths.get(fileName)));}
        catch(IOException ioException)
        {
            throw new RuntimeException( "IO Exception in YAML reader"
                                      , ioException);
        }
        
        yaml = new Yaml();
        Map<String, Object> config = (Map<String, Object>) yaml.load(content);

        return config;
    }

    /**
     * object2double
     *  Helper utility because I'm too stupid for java and YAML. After spending
     *  an eternity fiddeling with both I give up. I will never write another
     *  line of Java. 
     *  @param obj  Some object gottem from snakeyaml.
     *  @return     float array.
     *  */
    private static double[] object2double (Object obj)
    {
        ArrayList<Double> doubleList = (ArrayList<Double>) obj;
        double[] doubleArray = new double[doubleList.size()];
        int i = 0;

        for(Double f: doubleList)
            {doubleArray[i++] = (double) (f != null ? f : Double.NaN);}

        return doubleArray;
    }

    /**
     * float2tensor
     *  Helper uitility for converting float arrays to torch Tensors.
     *  @param array    float array.
     *  @return         Torch tensor.
     *  */
    private static Tensor float2tensor (float[] array)
    {
        if(array.length > 0)
            { return Tensor.fromBlob(array, new long[] {1, array.length}); }
        else
            { return Tensor.fromBlob(new float[] {}, new long[] {0, 0}); }
    }

    /**
     * PreceptModule
     *  Full implementaion for loading and evaluating pytorch models generated
     *  with precept.
     *  @param modPath  Path to torchscript model.pt as String.
     *  @param cfgPath  Path to YAML config as String.
     * */
    public PreceptModule(String modPath, String cfgPath)
    {
        this.modelPath = modPath;
        this.configPath = cfgPath;

        Map<String, Object> cfg = readYAMLcfg(this.configPath);

        this.numX = Integer.parseInt(cfg.get("num_x").toString());
        this.numY = Integer.parseInt(cfg.get("num_y").toString());

        this.paramsX = (List<String>) cfg.get("params_x");
        this.paramsY = (List<String>) cfg.get("params_y");

        this.maxX = MatrixUtils.createRealVector(object2double(cfg.get("max_x")));
        this.minX = MatrixUtils.createRealVector(object2double(cfg.get("min_x")));
        this.maxY = MatrixUtils.createRealVector(object2double(cfg.get("max_y")));
        this.minY = MatrixUtils.createRealVector(object2double(cfg.get("min_y")));
        
        this.trafoType = cfg.get("trafo_type").toString();

        if(this.trafoType == "box")
        {
            this.lambdaX = (cfg.get("lambda_x") != null) ? 
                           MatrixUtils.createRealVector(object2double(cfg.get("lambda_x"))) : null;
            this.lambdaX = (cfg.get("lambda_y") != null) ? 
                           MatrixUtils.createRealVector(object2double(cfg.get("lambda_y"))) : null;
        }
        else
        {
            this.lambdaX = null;
            this.lambdaY = null;
        }
        
        this.maskX = (cfg.get("mask_x") != null) ? 
                     (List<String>) cfg.get("mask_x") : null;
        this.maskY = (cfg.get("mask_x") != null) ? 
                     (List<String>) cfg.get("mask_y") : null;

        this.maskXidx = new ArrayList<Integer>();
        if(this.maskX != null)
        {
            for(String elem: this.maskX)
                {this.maskXidx.add(this.paramsX.indexOf(elem));}
        }

        this.maskYidx = new ArrayList<Integer>();
        if(this.maskY != null)
        {
            for(String elem: this.maskY)
                {this.maskYidx.add(this.paramsY.indexOf(elem));}
        }

        this.module = Module.load(this.modelPath);
    }

    /**
     * predict
     *  make predictions.
     *  @param data     Input data float array.
     *  @return         Prediction as float array.
     * */
    public double[] predict(double[] input)
    {
        if(input.length != this.numX)
            {throw new ArithmeticException();}

        //RealVector X = boxCox(input, this.lambdaX);
        RealVector X_ = scale1d( MatrixUtils.createRealVector(input) //X
                               , this.minX, this.maxX );

        double[] Xd = X_.toArray();
        float[] Xf = new float[Xd.length];
        for(int i = 0; i < Xd.length; i++)
            {Xf[i] = (float) Xd[i];}

        Tensor Xt  = Tensor.fromBlob(Xf, new long[] {1, this.numX});
        Tensor Yt = this.module.forward(IValue.from(Xt)).toTensor();

        float[] Yf = Yt.getDataAsFloatArray();
        double[] Yd = new double[Yf.length];
        for(int i = 0; i < Yf.length; i++)
            {Yd[i] = (double) Yf[i];}

        RealVector Y_ = MatrixUtils.createRealVector(Yd);
        
        RealVector Y = unscale1d(Y_, this.minY, this.maxY);
        //RealVector output = coxBox(Y, this.lambdaY)

        return Y.toArray();
        //return output.toArray();
    }

    /** 
     * scale
     *  Scale the input data according to the transformation used during
     *  training. Takes a Tensor, and returns one of the same size with values
     *  scaled [0;1] based on the minima and maxima specified in the config.
     *  
     *        ⎛  x-min(x)   ⎞
     *   x' = ⎜―――――――――――――⎟
     *        ⎝max(x)-min(x)⎠
     *  
     *  @param X    Input Tensor x.
     *  @return     Scaled [0;1] Tensor x'.
     * */
    public static RealVector scale1d(RealVector X, RealVector min, RealVector max)
    {
        return (X.subtract(min)).ebeDivide((max.subtract(min)));
    }

    /** 
     * unscale
     *  Un-scale the output of the model back to raw/real data
     *  according to the transformation used during training.
     *  Takes a Tensor, and returns one of the same size with values scaled [0;1]
     *  based on the minima and maxima specified in the config.
     *  
     *   x = x' ∙ (max(x) - min(x)) + min(x)
     *  
     * @param X  Output Tensor x'.
     * @return   Unscaled Tensor x.
     * */
    public static RealVector unscale1d(RealVector X, RealVector min, RealVector max)
    {
        return X.ebeMultiply(max.subtract(min)).add(min);
    }


    /** Box-Cox transformation
     *
     * For λ ≠ 0:
     * 
     *       λ  
     *      y -1
     *   y'=――――
     *       λ  
     *       
     * otherwise:
     * 
     *   y' = ln(y)
     *
     *  boxCox :: Tensor -> Tensor
     *
     *  @param y    Input
     *  @return     Transformed
     * */
    double boxCox(double y, double lambda)
    { 
        if(lambda != 0.0)
            { return ((pow(y, lambda) - 1.0) / lambda); }
        else
            { return log(y); }
    }
    
    /** Inverse Box-Cox Transformation
     *
     * For λ ≠ 0:
     *
     *      ⎛ln(y'∙λ+1)⎞
     *      ⎜――――――――――⎟
     *      ⎝    λ     ⎠
     *   y=e            
     * otherwise:
     *
     *      y'
     *   y=e  
     *
     * coxBox :: Tensor -> Tensor
     *
     * @param y     Input
     * @return      Transformed
     * */
    double coxBox(double y, double lambda)
    {
        if(lambda != 0.0)
            { return exp(log(y * lambda + 1) / lambda); }
        else
            { return exp(y); }
    }
}
