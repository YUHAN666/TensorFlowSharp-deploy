using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using TensorFlow;
using OpenCvSharp;
using HalconDotNet;
using System.Text.RegularExpressions;
using OpenCvSharp.XImgProc;

namespace PAT_AOI
{
    public class YoloProcess
    {

        TFSession sess;                                         //模型前向传播session
        private TFGraph preprocessGraph;              //图片预处理计算图
        private TFSession preprocessSession;         //图片预处理session
        private TFOutput imagePlaceholder;         //左白光图片预处理占位符
        private TFOutput imageBatchPlaceholder;    //图片batch占位符
        string inputImageTensorname;                            //模型输入Tensor名
        string inputImageShapeTensorname;                            //模型输入Tensor名
        string inputInputShapeTensorname;                            //模型输入Tensor名
        string outputScorename;                            //模型输出类别Tensor名
        string outputBoxname;                          //模型输出左Mask名
        string outputClassname;                         //模型输出右Mask名
        int[] inputTensorshape;                             //模型输入图片尺寸
        string saveNgDir;                                     //保存NG样品mask文件夹

        public string exceptionString;                             //错误类型字符串

        public bool Initial(string modelFilename, string inputImageTensorName,  string inputImageShapeTensorName, string inputInputShapeTensorName, string outputScorelName, string outputBoxName, string outputClassName, int[] inputTensorShape, string NgDir)
        {
            // 初始化函数
            //  输入参数：
            //               modelFilename: 模型路径
            //               inputTensorName: 模型输入张量名
            //               outputLabelName: 模型输出标签张量名
            //               outputMaskName_l: 左图输出Mask张量名
            //               outputMaskName_r: 右图输出Mask张量名
            //              inputTensorShape: 模型输入张量尺寸
            //              NgDir: 保存NG样本输出Mask的路径

            var graph = new TFGraph();
            inputImageTensorname = inputImageTensorName;                          
            inputImageShapeTensorname = inputImageShapeTensorName;                         
            inputInputShapeTensorname = inputInputShapeTensorName;                          
            outputScorename = outputScorelName;                           
            outputBoxname = outputBoxName;                         
            outputClassname = outputClassName;                      
            inputTensorshape = inputTensorShape;
            saveNgDir = NgDir;

            try
            {
                var model = File.ReadAllBytes(modelFilename);       // 读取模型文件
                graph.Import(model, "");        // 从模型中导入计算图
                sess = new TFSession(graph);        // 创建session

                ConstructPreprocessGraph(out preprocessGraph, out imagePlaceholder, out imageBatchPlaceholder);       //创建图片预处理计算图
                preprocessSession = new TFSession(preprocessGraph);

                return true;
            }

            catch (System.IO.IOException)
            {
                exceptionString = "Cannot find modelfile";
                return false;
            }
            catch (TensorFlow.TFException)
            {
                exceptionString = " Error in Preprocessing";
                return false;
            }

        }

        private void ConstructPreprocessGraph(out TFGraph graph, out TFOutput input_lw, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            graph = new TFGraph();
            input_lw = graph.Placeholder(TFDataType.String);
            //float scale = 255;
            TFOutput output_lw = graph.DecodeJpeg(contents: input_lw, channels: inputTensorshape[2]);  // 解码
            //float value = 128;
            output_lw = graph.Cast(x: output_lw, DstT: TFDataType.Float);     // 转类型
            output_lw = graph.ExpandDims(input: output_lw, dim: graph.Const(0));      // 增加batch维度
            //output_lw = graph.ResizeBilinear(images: output_lw, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // resize
            //output_lw = graph.CropAndResize(image: output_lw, boxes: graph.Const(new[,] { { -0.3, 0, 1.3, 1 } }), box_ind: graph.Const(new[] { 0 }), crop_size: graph.Const(new[] { 320, 320 }), extrapolation_value: value);
            //output_lw = graph.Div(x: output_lw, y: graph.Const(scale));       // 归一化
            output = output_lw;
        }



        public int Process(string imageFilename_lw)
        {
            // 处理函数，输出 0 for OK; 整数for NG classes; -1 for error, 如果NG则保存模型输出Mask
            // 输入参数：
            //              imageFilename_l: 输入左边图片
            //              imageFilename_r: 输入右边图片

            int defect;

            var imageBytes_lw = File.ReadAllBytes(imageFilename_lw);
       
            var imageBytesString_lw = TFTensor.CreateString(imageBytes_lw);


            var inputTensor = preprocessSession.Run(inputs: new[] { imagePlaceholder, },
                                                                                    inputValues: new[] { imageBytesString_lw,  },
                                                                                    outputs: new[] { imageBatchPlaceholder });     // 获取输入张量
            float[]  imageShape = new float[] { 1200, 1920};
            float[] inputShape = new float[] { 320, 320 };
            var runner = sess.GetRunner();
            runner.AddInput(inputImageShapeTensorname, imageShape);
            runner.AddInput(inputInputShapeTensorname, inputShape);
            runner.AddInput(inputImageTensorname, inputTensor[0]);
            runner.Fetch(outputScorename);
            runner.Fetch(outputBoxname);
            runner.Fetch(outputClassname);
            var output = runner.Run();         // 执行计算  incompatible shape 说明模型输入tensor shape错了
            var score = output[0];     // 输出标签(OK or NG)
            var boxes = output[1];      // 左mask输出
            var classes = output[2];     // 右mask输出



            float[][] box = (float[][])boxes.GetValue(true);
            Mat image = Cv2.ImRead(imageFilename_lw);
            Point p1 = new Point(box[0][0], box[0][1]);
            Point p2 = new Point(box[0][2], box[0][3]);
            Scalar red = new Scalar(0, 0, 255);
            Cv2.Rectangle(image, p1, p2, red);

            Point p3 = new Point(box[1][0], box[1][1]);
            Point p4 = new Point(box[1][2], box[1][3]);
            Cv2.Rectangle(image, p3, p4, red);
            Size resize_shape = new Size ( 960, 600 );
            Cv2.Resize(image, image, resize_shape);
            Cv2.ImShow("image", image);
            Cv2.WaitKey();

            //Console.WriteLine(label[0][0]);
            //Console.WriteLine(label[1][0]);

            ////IntPtr resData1 = mask_out.Data;
            ////Mat cvmask1 = new Mat(928, 320, 5, resData1);
            ////cvmask1 = cvmask1 * 255;
            ////Cv2.ImWrite("test.png", cvmask1);

            int result = 0;

            return result;



            //catch (System.IO.IOException)
            //{
            //    exceptionString = " Cannnot find image file ";
            //    return -1;

            //}
            //catch (TensorFlow.TFException)
            //{
            //    exceptionString = " Error in Inference ";
            //    return -1;

            //}
            //catch (System.ArgumentOutOfRangeException)
            //{
            //    exceptionString = " Incorrect input/output Tensorname ";
            //    return -1;
            //}

        }


        private int PostProcessNG(string maskFilename)
        {
            // TODO  对NG样本进行分类，输出1-3作为类别，出错返回-1

            //put halcon codes here
            HObject ho_Image, ho_Region;

            // Local control variables 

            HTuple hv_Area = null, hv_Row = null, hv_Column = null;
            //HTuple hv_defect = new HTuple();
            int defect = -1;
            // Initialize local and output iconic variables 
            HOperatorSet.GenEmptyObj(out ho_Image);
            HOperatorSet.GenEmptyObj(out ho_Region);
            //Image Acquisition 01: Code generated by Image Acquisition 01

            ho_Image.Dispose();
            HOperatorSet.ReadImage(out ho_Image, maskFilename);
            ho_Region.Dispose();
            HOperatorSet.Threshold(ho_Image, out ho_Region, 1, 255);

            HOperatorSet.AreaCenter(ho_Region, out hv_Area, out hv_Row, out hv_Column);
            if ((int)(new HTuple(hv_Area.TupleGreater(10000))) != 0)
            {
                //hv_defect = 0;
                defect = 1;
            }
            else if ((int)(new HTuple(hv_Area.TupleLess(10))) != 0)
            {
                //hv_defect = 1;
                defect = 2;
            }
            else
            {
                //hv_defect = 2;
                defect = 3;
            }

            ho_Image.Dispose();
            ho_Region.Dispose();

            return defect;
        }


        public void Dispose()
        {
            sess.Dispose();
            preprocessGraph.Dispose();
            preprocessSession.Dispose();

        }

    }

}

