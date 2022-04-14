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
using System.Runtime.InteropServices;

namespace PAT_AOI
{
    public class DeepLearningProcess
    {

        TFSession sess;                                         //模型前向传播session
        private TFGraph preprocessGraph;              //图片预处理计算图
        private TFSession preprocessSession;         //图片预处理session
        private TFOutput imagePlaceholder_lw;         //左白光图片预处理占位符
        private TFOutput imagePlaceholder_lr;         //左红光图片预处理占位符
        private TFOutput imagePlaceholder_rw;         //右白光图片预处理占位符
        private TFOutput imagePlaceholder_rr;         //右红光图片预处理占位符
        private TFOutput imageBatchPlaceholder;    //图片batch占位符
        string inputTensorname;                            //模型输入Tensor名
        string outputLabelname;                            //模型输出类别Tensor名
        string outputMaskname_l;                          //模型输出左Mask名
        string outputMaskname_r;                         //模型输出右Mask名
        int[] inputTensorshape;                             //模型输入图片尺寸
        string saveNgDir;                                     //保存NG样品mask文件夹

       public string exceptionString;                             //错误类型字符串

        public bool Initial(string modelFilename, string inputTensorName, string outputLabelName, string outputMaskName_l, string outputMaskName_r, int[] inputTensorShape, string NgDir)
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
            inputTensorname = inputTensorName;
            outputLabelname = outputLabelName;
            outputMaskname_l = outputMaskName_l;
            outputMaskname_r = outputMaskName_r;
            inputTensorshape = inputTensorShape;
            saveNgDir = NgDir;

            try {
                var model = File.ReadAllBytes(modelFilename);       // 读取模型文件
                graph.Import(model, "");        // 从模型中导入计算图

                TFSessionOptions options = new TFSessionOptions();
                byte[] data = { 50, 09, 09, 154, 153, 153, 153, 153, 153, 201, 63 }; //0.

                IntPtr buffer = Marshal.AllocHGlobal(data.Length);
                Marshal.Copy(data, 0, buffer, data.Length);
                TFStatus ts = new TFStatus();
                options.SetConfig(buffer, data.Length, ts);
                Marshal.FreeHGlobal(buffer);
                sess = new TFSession(graph, options);        // 创建session

                //sess = new TFSession(graph);        // 创建session

                ConstructPreprocessGraph(out preprocessGraph, out imagePlaceholder_lw, out imagePlaceholder_lr,
                    out imagePlaceholder_rw, out imagePlaceholder_rr, out imageBatchPlaceholder);       //创建图片预处理计算图
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
        
        private void ConstructPreprocessGraph(out TFGraph graph, out TFOutput input_lw, out TFOutput input_lr, out TFOutput input_rw, out TFOutput input_rr, out TFOutput output)
        {
            // 构建图像归一化，resize等预处理过程的计算图
            graph = new TFGraph();
            input_lw = graph.Placeholder(TFDataType.String);
            input_lr = graph.Placeholder(TFDataType.String);
            input_rw = graph.Placeholder(TFDataType.String);
            input_rr = graph.Placeholder(TFDataType.String);
            float scale = 255;
        
            TFOutput output_lw = graph.DecodeBmp(contents: input_lw, channels: inputTensorshape[2]);  // 解码
            output_lw = graph.Cast(x: output_lw, DstT: TFDataType.Float);     // 转类型
            output_lw = graph.Div(x: output_lw, y: graph.Const(scale));       // 归一化
            output_lw = graph.ExpandDims(input: output_lw, dim: graph.Const(0));      // 增加batch维度
            output_lw = graph.ResizeBilinear(images: output_lw, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // resize

            TFOutput output_lr = graph.DecodeBmp(contents: input_lr, channels: inputTensorshape[2]);  // 解码
            output_lr = graph.Cast(x: output_lr, DstT: TFDataType.Float);     // 转类型
            output_lr = graph.Div(x: output_lr, y: graph.Const(scale));       // 归一化
            output_lr = graph.ExpandDims(input: output_lr, dim: graph.Const(0));      // 增加batch维度
            output_lr = graph.ResizeBilinear(images: output_lr, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));       // resize

            TFOutput output_rw = graph.DecodeBmp(contents: input_rw, channels: inputTensorshape[2]);
            output_rw = graph.Cast(x: output_rw, DstT: TFDataType.Float);
            output_rw = graph.Div(x: output_rw, y: graph.Const(scale));
            output_rw = graph.ExpandDims(input: output_rw, dim: graph.Const(0));
            output_rw = graph.ResizeBilinear(images: output_rw, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));

            TFOutput output_rr = graph.DecodeBmp(contents: input_rr, channels: inputTensorshape[2]);
            output_rr = graph.Cast(x: output_rr, DstT: TFDataType.Float);
            output_rr = graph.Div(x: output_rr, y: graph.Const(scale));
            output_rr = graph.ExpandDims(input: output_rr, dim: graph.Const(0));
            output_rr = graph.ResizeBilinear(images: output_rr, size: graph.Const(new[] { inputTensorshape[0], inputTensorshape[1] }));

            TFOutput output_l = graph.Concat(concat_dim: graph.Const(3), new TFOutput[] { output_lw, output_lr});       // 将两张图片合成一个输入张量
            TFOutput output_r = graph.Concat(concat_dim: graph.Const(3), new TFOutput[] { output_rw, output_rr });       // 将两张图片合成一个输入张量
            output = graph.Concat(concat_dim: graph.Const(0), new TFOutput[] { output_l, output_r });       // 将两张图片合成一个输入张量
        }



        public int Process(string imageFilename_lw, string imageFilename_lr, string imageFilename_rw, string imageFilename_rr)
        {
            // 处理函数，输出 0 for OK; 整数for NG classes; -1 for error, 如果NG则保存模型输出Mask
            // 输入参数：
            //              imageFilename_l: 输入左边图片
            //              imageFilename_r: 输入右边图片

            int defect;

            var imageBytes_lw = File.ReadAllBytes(imageFilename_lw);
            var imageBytes_lr = File.ReadAllBytes(imageFilename_lr);
            var imageBytes_rw = File.ReadAllBytes(imageFilename_rw);
            var imageBytes_rr = File.ReadAllBytes(imageFilename_rr);

            var imageBytesString_lw = TFTensor.CreateString(imageBytes_lw);
            var imageBytesString_lr = TFTensor.CreateString(imageBytes_lr);
            var imageBytesString_rw = TFTensor.CreateString(imageBytes_rw);
            var imageBytesString_rr = TFTensor.CreateString(imageBytes_rr);

            var inputTensor = preprocessSession.Run(inputs: new[] { imagePlaceholder_lw, imagePlaceholder_lr, imagePlaceholder_rw, imagePlaceholder_rr },
                                                                                    inputValues: new[] { imageBytesString_lw, imageBytesString_lr, imageBytesString_rw, imageBytesString_rr },
                                                                                    outputs: new[] { imageBatchPlaceholder });     // 获取输入张量

            var runner = sess.GetRunner();
            runner.AddInput(inputTensorname, inputTensor[0]);
            runner.Fetch(outputLabelname);
            runner.Fetch(outputMaskname_l);
            runner.Fetch(outputMaskname_r);
            var output = runner.Run();         // 执行计算  incompatible shape 说明模型输入tensor shape错了
            var decision_out = output[0];     // 输出标签(OK or NG)
            var mask_out_l = output[0];      // 左mask输出
            var mask_out_r = output[1];     // 右mask输出



            float[][] label = (float[][])decision_out.GetValue(true);

            //Console.WriteLine(label[0][0]);
            //Console.WriteLine(label[1][0]);

            ////IntPtr resData1 = mask_out.Data;
            ////Mat cvmask1 = new Mat(928, 320, 5, resData1);
            ////cvmask1 = cvmask1 * 255;
            ////Cv2.ImWrite("test.png", cvmask1);

            int result = 0;
            // 如果NG则保存模型输出Mask
            if (label[0][0] > 0.5)
            {
                //float[][][][] mask = (float[][][][])mask_out.GetValue(true);
                IntPtr resData_l = mask_out_l.Data;
                //UIntPtr dataSize = mask_out.TensorByteSize;

                //byte[] s_ImageBuffer = new byte[(int)dataSize];
                //System.Runtime.InteropServices.Marshal.Copy(resData, s_ImageBuffer, 0, (int)dataSize);
                Mat cvmask_l = new Mat(inputTensorshape[0], inputTensorshape[1], 5, resData_l);
                cvmask_l = cvmask_l * 255;
                string[] saveNgImage_l = imageFilename_lw.Split(new char[] { '\\' });
                string NgFilename_l = saveNgDir + '\\' + saveNgImage_l[saveNgImage_l.Length - 1];
                Cv2.ImWrite(NgFilename_l, cvmask_l);
                result = 1;
                //Cv2.ImShow("demo", cvmask);
                //Cv2.WaitKey(0);
                //System.IO.File.WriteAllBytes("test.bmp", s_ImageBuffer);

                //byte[][][][] mask = (byte[][][][])mask_out.GetValue(true);
                //defect = PostProcessNG(NgFilename_l);

            }

            if (label[1][0] > 0.5)
            {

                IntPtr resData_r = mask_out_r.Data;
                Mat cvmask_r = new Mat(inputTensorshape[0], inputTensorshape[1], 5, resData_r);
                cvmask_r = cvmask_r * 255;
                string[] saveNgImage_r = imageFilename_rw.Split(new char[] { '\\' });
                string NgFilename_r = saveNgDir + '\\' + saveNgImage_r[saveNgImage_r.Length - 1];
                Cv2.ImWrite(NgFilename_r, cvmask_r);
                //defect =  PostProcessNG(NgFilename_r);
                result = 1;

            }


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

